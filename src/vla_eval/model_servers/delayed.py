"""Simulated inference delay for any chunked ``PredictModelServer``, in sync mode: deterministic on any machine.

``DelayedChunkServer`` wraps a server whose ``predict`` / ``predict_batch`` returns a ``[T, D]`` chunk and executes
its chunks the way a policy whose inference takes ``d`` env steps would (naive async execution, as
``KinetixRTCServer`` with ``method="naive"``, ``latency="steps"``):

- every ``s`` steps (``execute_steps``) it infers a chunk from that step's observation, on the model thread;
- that chunk lands ``d`` steps later; until then the previous chunk keeps playing (its actions ``s .. s + d - 1``);
- a chunk landing at step ``t + d`` is served from its action ``d``, the one meant for step ``t + d``.

Each chunk thus serves ``s`` steps, from actions ``d .. s + d - 1`` (``s + d <= T``), and ``d <= s`` keeps one
inference in flight.  The first chunk lands at once (official ``eval_flow.py``: computed before the loop), so ``d = 0``
is exactly the wrapped server with ``chunk_size = s``.  Delay is charged per inference, not per message: a step served
from a landed chunk costs nothing.  The step is the episode's observation index; nothing waits on wall time.

Enable it from a model-server config, with no change to the server script::

    extends: smolvla_libero.yaml
    latency: {delay_steps: 2, execute_steps: 10}   # or delay_s: 0.1, rounded up to whole steps of step_dt

``delay_s`` needs the episode's ``step_dt`` (the benchmark's, sent in ``EPISODE_START``).  Live episodes are refused:
live mode measures real latency.  Episodes infer one at a time (no cross-episode batching).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vla_eval import columns
from vla_eval.model_servers.base import Rows
from vla_eval.model_servers.daemons import DaemonModelServer, EpisodeContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import DimSpec
from vla_eval.types import EpisodeResult, Observation

__all__ = ["DelayedChunkServer", "Latency"]

_EPS = 1e-9  # delay_s / step_dt rounding: 0.1 / 0.05 is 2 steps, not 3


@dataclass(frozen=True)
class Latency:
    """A model-server config's ``latency:`` block.  ``delay_steps`` (d) or ``delay_s`` (seconds of env time, rounded
    up to whole steps); ``execute_steps`` (s): actions executed per chunk, default the episode's ``chunk_size``."""

    delay_steps: int | None = None
    delay_s: float | None = None
    execute_steps: int | None = None

    def __post_init__(self) -> None:
        if (self.delay_steps is None) == (self.delay_s is None):
            raise ValueError("latency needs exactly one of delay_steps and delay_s")
        if self.delay_steps is not None and not (_is_int(self.delay_steps) and self.delay_steps >= 0):
            raise ValueError(f"delay_steps must be an int >= 0, got {self.delay_steps!r}")
        if self.delay_s is not None and not (
            isinstance(self.delay_s, (int, float))
            and not isinstance(self.delay_s, bool)
            and math.isfinite(self.delay_s)
            and self.delay_s >= 0
        ):
            raise ValueError(f"delay_s must be a finite number >= 0, got {self.delay_s!r}")
        if self.execute_steps is not None and not (_is_int(self.execute_steps) and self.execute_steps >= 1):
            raise ValueError(f"execute_steps must be an int >= 1, got {self.execute_steps!r}")

    def steps(self, step_dt: float | None) -> int:
        """d for an episode of ``step_dt`` seconds per step."""
        if self.delay_steps is not None:
            return self.delay_steps
        assert self.delay_s is not None
        if step_dt is None:
            raise ValueError("latency.delay_s needs step_dt: the benchmark declares none; use delay_steps")
        return math.ceil(self.delay_s / step_dt - _EPS)


@dataclass(frozen=True)
class _Schedule:
    s: int  # actions executed per chunk
    d: int  # steps from an observation to its chunk


class DelayedChunkServer(DaemonModelServer):
    """``inner``'s chunks executed under a simulated inference delay (see the module docstring).

    ``inner``'s ``reset_episodes`` / ``finish_episodes`` / ``predict`` (or ``predict_batch``) run on this server's
    model thread; its observation params and specs are this server's.  ``inner``'s own engine (batching, chunk
    buffers) is not used.
    """

    def __init__(
        self,
        inner: PredictModelServer,
        *,
        delay_steps: int | None = None,
        delay_s: float | None = None,
        execute_steps: int | None = None,
    ) -> None:
        if not isinstance(inner, PredictModelServer):
            raise TypeError(f"DelayedChunkServer wraps a PredictModelServer, got {type(inner).__name__}")
        self.latency = Latency(delay_steps, delay_s, execute_steps)
        self.inner = inner
        self._batched = type(inner).predict_batch is not PredictModelServer.predict_batch
        self._schedules: dict[int, _Schedule] = {}  # by row id
        super().__init__(backlog=1, max_rows=inner.max_rows)  # sync: one observation in flight per episode

    def get_observation_params(self) -> dict[str, Any]:
        return self.inner.get_observation_params()

    def get_action_spec(self) -> dict[str, DimSpec]:
        return self.inner.get_action_spec()

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return self.inner.get_observation_spec()

    # -- model thread ----------------------------------------------------------------------------------------------

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        live = {i for i, episode in enumerate(rows.episodes) if episode.mode == "live"}
        for episode in rows.episodes:  # as PredictionEngine: inner's reset may override it (e.g. per suite)
            episode.chunk_size = self.inner.chunk_size
        failed: dict[int, Exception] = dict.fromkeys(
            live, ValueError("DelayedChunkServer simulates delay in sync mode; live mode measures real latency")
        )
        ok = [i for i in range(len(rows)) if i not in failed]
        if ok:
            out = self.inner.reset_episodes(rows if len(ok) == len(rows) else rows.take(ok)) or {}
            failed.update({ok[j]: exc for j, exc in out.items()})
        for i, row in enumerate(rows):
            if i in failed:
                continue
            try:
                self._schedules[row.id] = self._schedule(row.episode.chunk_size, row.episode.config.get("step_dt"))
            except Exception as exc:
                failed[i] = exc
        return failed

    def _schedule(self, chunk_size: int | None, step_dt: float | None) -> _Schedule:
        s = self.latency.execute_steps or chunk_size
        if s is None:
            raise ValueError("latency needs execute_steps, or a chunk_size on the wrapped server")
        d = self.latency.steps(step_dt)
        if d > s:
            raise ValueError(
                f"a delay of {d} steps exceeds execute_steps={s}: one inference would start before "
                "the last landed; raise execute_steps"
            )
        return _Schedule(s, d)

    def _chunk(self, obs: Observation, rows: Rows, need: int) -> np.ndarray:
        if self._batched:
            actions = self.inner.predict_batch(columns.stack([obs]), rows)["actions"][0]
        else:
            actions = self.inner.predict(obs, rows[0])["actions"]
        chunk = np.asarray(actions, np.float32)
        if chunk.ndim != 2 or len(chunk) < need:
            raise ValueError(
                f"{type(self.inner).__name__} returned actions of shape {chunk.shape}: executing "
                f"{need} actions per chunk under the delay needs a [T >= {need}, D] chunk"
            )
        return chunk

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        try:
            self.inner.finish_episodes(rows, results)
        finally:
            for rid in rows.ids.tolist():
                self._schedules.pop(rid, None)

    # -- event loop ------------------------------------------------------------------------------------------------

    async def run(self, ep: EpisodeContext) -> None:
        sched = self._schedules[ep.row_id]
        s, d = sched.s, sched.d
        plan_t, plan = 0, np.zeros((0, 0), np.float32)  # plan[i] is the action for step plan_t + i
        landing: tuple[int, np.ndarray] | None = None  # (its observation's step, chunk), due d steps after it
        async for req in ep.observations():
            j = req.row.step
            if landing is not None and j == landing[0] + d:
                (plan_t, plan), landing = landing, None
            if j % s == 0:
                chunk = await ep.call(self._chunk, req.obs, req.rows, s + d)
                if j == 0 or d == 0:
                    plan_t, plan = j, chunk
                else:
                    landing = (j, chunk)
            await req.reply({"actions": plan[j - plan_t]})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
