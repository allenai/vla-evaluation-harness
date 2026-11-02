"""VectorEpisodeRunner: episodes over a VectorStepBenchmark's environments, one server round trip per step."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vla_eval import columns, watchdog
from vla_eval.benchmarks.base import VectorStepBenchmark
from vla_eval.types import Batch, EpisodeResult, Task
from vla_eval.recording import EpisodeRecorder, RecordingError
from vla_eval.runners.base import (
    EnvStartError,
    EnvStepError,
    ModelActError,
    ModelStartError,
    episode_end_payload,
    episode_start_payload,
    phase,
)

logger = logging.getLogger(__name__)


def _try(error: type[EnvStartError] | type[EnvStepError], fn: Callable[..., Any], *args: Any) -> Any:
    """``fn(*args)``, or the exception it raised (tagged with ``error``'s phase); RecordingError propagates."""
    try:
        with phase(error):
            return fn(*args)
    except RecordingError:
        raise
    except Exception as exc:
        return exc


@dataclass
class VectorEpisode:
    """One episode for the runner; ``ref`` is the caller's handle, returned in ``on_end``."""

    task: Task
    recorder: EpisodeRecorder
    ref: Any = None


class OnEnd(Protocol):
    """``on_end(episode, result, error, aborted=...)``: exactly one of ``result`` / ``error`` is set.  ``aborted``:
    the run stops with this episode (the server is unreachable), so the episode is to be redone on a rerun."""

    def __call__(
        self,
        episode: VectorEpisode,
        result: EpisodeResult | None,
        error: BaseException | None,
        *,
        aborted: bool = False,
    ) -> None: ...


@dataclass
class _Running:
    episode: VectorEpisode
    t0: float


class VectorEpisodeRunner:
    """Runs episodes on a :class:`VectorStepBenchmark`'s slots, one ``VECTOR_STEP`` round trip per step.

    Free slots take ``next_episode(slot)`` (``None``: none now; asked again when an episode ends).  A slot's error
    ends its episode; a ``reset``/``step`` exception ends all on the simulator; a failed round trip ends the
    running ones and reconnects, raising ``ConnectionError`` if that fails.  The benchmark runs on the loop thread.
    """

    async def run(
        self,
        benchmark: VectorStepBenchmark,
        conn: Any,  # Connection
        next_episode: Callable[[int], VectorEpisode | None],
        on_end: OnEnd,
        *,
        max_steps: int | None = None,
    ) -> None:
        self._benchmark, self._conn, self._max_steps = benchmark, conn, max_steps
        self._running: dict[int, _Running] = {}
        self._slots: list[int] = []  # the running slots, in the rows' order of
        self._obs: Batch = {}  # their observations
        self._steps = np.zeros(benchmark.num_envs, np.int64)
        self._ends: list[list[Any]] = []  # [slot, EPISODE_END payload] of the server sessions to end next round trip
        exhausted = False

        def ended(episode: VectorEpisode, result: Any, error: Any, *, aborted: bool = False) -> None:
            nonlocal exhausted
            exhausted = False  # the caller may have work again (a held task, a released queue item)
            on_end(episode, result, error, aborted=aborted)

        self._on_end: OnEnd = ended
        try:
            while True:
                starts: list[list[Any]] = []
                if not exhausted and (not self._running or benchmark.partial_reset):
                    new: list[tuple[int, VectorEpisode]] = []
                    for slot in range(benchmark.num_envs):
                        if slot in self._running:
                            continue
                        episode = next_episode(slot)
                        if episode is None:
                            exhausted = True
                            break
                        new.append((slot, episode))
                    if new:
                        starts = await self._start(new)
                if not self._running:
                    if exhausted:
                        break
                    continue
                actions = await self._exchange(starts)
                if actions is not None:
                    await self._step(actions)
        except ConnectionError as exc:
            for r in self._running.values():
                on_end(r.episode, None, exc, aborted=True)
            raise
        if self._ends:  # the last episodes' sessions; their results are booked already
            try:
                await conn.exchange({"ends": self._ends, "starts": [], "slots": [], "observations": {}})
            except Exception as exc:
                logger.warning("Ending the last server sessions failed: %s: %s", type(exc).__name__, exc)

    def _keep(self, rows: list[int]) -> None:
        """Keep only these rows of the running slots (the others' episodes have ended)."""
        if len(rows) < len(self._slots):
            self._slots = [self._slots[i] for i in rows]
            self._obs = columns.take(self._obs, rows)

    def _end_running(self, exc: Exception) -> None:
        for slot, r in self._running.items():
            self._on_end(r.episode, None, exc)
            self._ends.append([slot, episode_end_payload(error=exc)])
        self._running.clear()
        self._slots, self._obs = [], {}

    async def _start(self, new: list[tuple[int, VectorEpisode]]) -> list[list[Any]]:
        """Reset the new slots; return their ``EPISODE_START``s."""
        benchmark = self._benchmark
        slots = [slot for slot, _ in new]
        episodes = [episode for _, episode in new]
        t0 = time.monotonic()

        def reset() -> Batch:
            with phase(EnvStartError):
                obs = benchmark.reset(slots, [e.task for e in episodes], [e.recorder for e in episodes])
                if columns.num_rows(obs) not in (None, len(slots)):
                    raise ValueError(f"reset() returned {columns.num_rows(obs)} observations for {len(slots)} slots")
                return obs

        try:
            obs = reset()
            self._obs = columns.concat([self._obs, obs]) if self._slots else obs  # a new task's columns may differ
        except RecordingError:
            raise
        except Exception as exc:
            for episode in episodes:
                self._on_end(episode, None, exc)
            self._end_running(exc)
            return []
        self._steps[slots] = 0
        self._slots += slots
        for slot, episode in new:
            self._running[slot] = _Running(episode, t0)
        rate = {} if benchmark.control_hz is None else {"control_hz": benchmark.control_hz}
        return [[slot, episode_start_payload(e.task, e.recorder, **rate)] for slot, e in new]

    async def _exchange(self, starts: list[list[Any]]) -> Batch | None:
        """One round trip; the actions of the running slots, after ending those the server failed."""
        step = {"ends": self._ends, "starts": starts, "slots": self._slots, "observations": self._obs}
        self._ends = []
        try:
            reply = await self._conn.exchange(step)
        except (ConnectionError, RecordingError):
            raise
        except Exception as exc:  # closed, timed out or out of sync: the server's sessions are gone
            await self._reconnect(exc)
            return None
        errors = {slot: detail for slot, detail in reply.get("errors", [])}
        if errors:
            for slot in [s for s in self._slots if s in errors]:
                started = not str(errors[slot]).startswith("episode_start_failed")  # else no server episode to end
                error = (ModelActError if started else ModelStartError)(f"Server error: {errors[slot]}")
                self._on_end(self._running.pop(slot).episode, None, error)
                if started:
                    self._ends.append([slot, episode_end_payload(error=error)])
            self._keep([i for i, s in enumerate(self._slots) if s not in errors])
        actions = reply.get("actions", {})
        try:
            rows = columns.num_rows(actions)
        except ValueError as exc:
            rows = f"malformed ({exc})"
        if rows not in (None, len(self._slots)) or (not actions and self._slots):
            self._end_running(ModelActError(f"Server reply has {rows} actions for {len(self._slots)} slots"))
            return None
        return actions if self._slots else None

    async def _step(self, actions: Batch) -> None:
        benchmark, slots, max_steps = self._benchmark, self._slots, self._max_steps
        idx = np.asarray(slots, np.int64)

        def step() -> tuple[Batch, list[int], dict[int, Any]]:
            """Step; return the observations, the rows that go on, and each ended slot's result or error."""
            with phase(EnvStepError):
                obs, done = benchmark.step(slots, actions)
                done = np.asarray(done, bool).reshape(-1)
                if len(done) != len(slots):
                    raise ValueError(f"step() returned {len(done)} done flags for {len(slots)} slots")
            self._steps[idx] += 1
            if max_steps is not None:
                done = done | (self._steps[idx] >= max_steps)
            results = {slots[i]: _try(EnvStepError, benchmark.result, slots[i]) for i in np.flatnonzero(done)}
            return obs, np.flatnonzero(~done).tolist(), results

        try:
            obs, keep, results = step()
        except RecordingError:
            raise
        except Exception as exc:
            self._end_running(exc)
            return
        watchdog.pet()  # a slow simulator's episode can outlast the stall timeout
        self._obs = obs
        self._keep(keep)
        now = time.monotonic()
        for slot, value in results.items():
            r = self._running.pop(slot)
            if isinstance(value, Exception):
                self._on_end(r.episode, None, value)
                self._ends.append([slot, episode_end_payload(error=value)])
                continue
            result: EpisodeResult = {
                "metrics": value,
                "steps": int(self._steps[slot]),
                "elapsed_sec": round(now - r.t0, 3),
            }
            self._on_end(r.episode, result, None)
            self._ends.append([slot, episode_end_payload(result)])

    async def _reconnect(self, exc: Exception) -> None:
        """End every running episode with ``exc`` after reconnecting; aborted when the reconnect fails."""
        try:
            await self._conn.reconnect()
        except Exception as err:
            for r in self._running.values():
                self._on_end(r.episode, None, exc, aborted=True)
            self._running.clear()
            self._slots, self._obs = [], {}
            if isinstance(err, ConnectionError):
                raise
            # e.g. a HELLO timeout: the server cannot take this runner's episodes
            raise ConnectionError(f"reconnect failed: {type(err).__name__}: {err}") from err
        for r in self._running.values():
            self._on_end(r.episode, None, exc)
        self._running.clear()
        self._slots, self._obs = [], {}
