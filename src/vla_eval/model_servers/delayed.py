"""Simulated inference delay for a chunked ``PredictModelServer``, in sync mode: deterministic on any machine.

``DelayedChunkServer`` wraps a server whose ``predict`` / ``predict_batch`` returns a ``[T, D]`` chunk and executes
its chunks the way a policy whose inference takes ``d`` env steps would (naive async execution):

- every ``s`` steps (``execute_steps``) it infers a chunk from that step's observation, on the model thread;
- that chunk lands ``d`` steps later; until then the previous chunk keeps playing (its actions ``s .. s + d - 1``);
- a chunk landing at step ``t + d`` is served from its action ``d``, the one meant for step ``t + d``.

The first chunk lands at once (official ``eval_flow.py``: computed before the loop) and serves steps ``0 .. s + d - 1``;
each later chunk serves ``s`` steps, from actions ``d .. s + d - 1`` (``s + d <= T``).  ``d <= s`` keeps one inference
in flight.  So ``d = 0`` is exactly the wrapped server with ``chunk_size = s``, and each step plays the action index
``KinetixRTCServer`` plays with ``method="naive"``, ``latency="steps"``.  Kinetix infers twice from step 0's
observation (an undelayed initial chunk, then chunk 0, landing at ``d``), this server once: for a stochastic policy,
Kinetix's steps ``d .. s + d - 1`` come from a second sample.  Delay is charged per inference, not per message: a step
served from a landed chunk costs nothing.  The step is the episode's observation index; nothing waits on wall time.

The wrapped ``predict`` must not depend on which of its earlier actions ran: under a delay, each chunk after the first
skips its first ``d`` actions.  A server whose state assumes its first ``chunk_size`` actions ran (e.g. X-VLA's
RoboTwin and Google Robot profiles, which carry the last commanded pose to the next call) sets ``assumes_whole_chunks``
and is refused.

Enable it from a model-server config, with no change to the server script::

    extends: smolvla_libero.yaml
    latency: {delay_steps: 2, execute_steps: 10}   # or delay_s: 0.1, rounded up to whole steps of step_dt

``delay_s`` needs the episode's ``step_dt`` (the benchmark's, sent in ``EPISODE_START``).  HELLO reports the wrapped
class (``inner_model_server``) and the ``latency`` block; results save them in ``server_info``.

Limits: live episodes are refused (live mode measures real latency), and so is ``VECTOR_STEP`` (vector benchmarks).
Episodes infer one at a time (no cross-episode batching): the wrapped server's engine is not used, so its
``max_batch_size`` / ``max_wait_time`` (``GET /config``) do not apply.
"""

from __future__ import annotations

import dataclasses
import logging
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

logger = logging.getLogger(__name__)

_EPS = 1e-9  # delay_s / step_dt rounding: 0.1 / 0.05 is 2 steps, not 3
_ABORTED: EpisodeResult = {"status": "aborted", "failure_reason": None}  # a start rolled back after inner's reset


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
        if self.delay_steps is not None and self.execute_steps is not None:
            _check_delay(self.delay_steps, self.execute_steps)

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
        if inner.assumes_whole_chunks:
            raise ValueError(
                f"{type(inner).__name__} assumes each chunk runs whole (assumes_whole_chunks): its predict "
                "would go wrong when a delay skips a chunk's first actions"
            )
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

    def _hello_info(self) -> dict[str, Any]:
        return {"inner_model_server": type(self.inner).__name__, "latency": dataclasses.asdict(self.latency)}

    # -- model thread ----------------------------------------------------------------------------------------------

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        failed: dict[int, Exception] = {}
        delays: dict[int, int] = {}
        for i, episode in enumerate(rows.episodes):  # what needs no chunk size, before inner's reset
            episode.chunk_size = self.inner.chunk_size  # as PredictionEngine: inner's reset may override it
            try:
                if episode.mode == "live":
                    raise ValueError("live mode measures real latency: DelayedChunkServer simulates it in sync")
                delays[i] = self.latency.steps(episode.config.get("step_dt"))
                if self.latency.execute_steps is not None:
                    _check_delay(delays[i], self.latency.execute_steps)
            except Exception as exc:
                failed[i] = exc
        ok = [i for i in range(len(rows)) if i not in failed]
        if not ok:
            return failed
        try:
            out = self.inner.reset_episodes(rows if len(ok) == len(rows) else rows.take(ok)) or {}
        except Exception as exc:
            out = dict.fromkeys(range(len(ok)), exc)
        failed.update({ok[j]: exc for j, exc in out.items()})
        rollback: list[int] = []  # reset by inner, failed here: inner finishes them
        for i in ok:
            if i in failed:
                continue
            row = rows[i]
            try:
                s = self.latency.execute_steps or row.episode.chunk_size
                if s is None:
                    raise ValueError("latency needs execute_steps, or a chunk_size on the wrapped server")
                _check_delay(delays[i], s)
                self._schedules[row.id] = _Schedule(s, delays[i])
            except Exception as exc:
                failed[i] = exc
                rollback.append(i)
        if rollback:
            try:
                self.inner.finish_episodes(rows.take(rollback), [_ABORTED] * len(rollback))
            except Exception:
                logger.exception("finish failed while rolling back a start")
        return failed

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


def _check_delay(d: int, s: int) -> None:
    if d > s:
        raise ValueError(
            f"a delay of {d} steps exceeds execute_steps={s}: one inference would start before the last landed; "
            "raise execute_steps"
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
