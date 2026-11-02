"""DelayedChunkServer: step-counted inference delay for any chunked ``PredictModelServer``, on a sync sim clock.

The naive schedule of ``KinetixRTCServer`` (``method="naive"``, ``latency="steps"``), for models that only implement
``predict``: inference every ``execute_steps`` (s) steps from that step's observation; its chunk lands
``delay_steps`` (d) steps later, served from index d; until then the previous chunk keeps playing.  The initial
chunk lands with no delay (official ``eval_flow.py``).  d = 0 is plain sync with ``chunk_size = s``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import anyio
import numpy as np
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval import columns
from vla_eval.model_servers.base import Rows
from vla_eval.model_servers.daemons import DaemonModelServer, EpisodeContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.recording import StepRecorder
from vla_eval.specs import DimSpec
from vla_eval.types import EpisodeResult, Observation

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class _Inference:
    t: int  # the step whose observation it was inferred from; -1: the initial chunk
    done: anyio.Event = field(default_factory=anyio.Event)
    chunk: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))


@dataclass(eq=False)
class _RowState:
    recorder: StepRecorder | None


class DelayedChunkServer(DaemonModelServer):
    """Wraps ``inner`` and answers each sync observation with the action the schedule gives its step.

    Args:
        inner: The model; its ``predict`` (or ``predict_batch``) must return the full ``[T, D]`` chunk.
        delay_steps: d, ``0 <= d <= s``.
        execute_steps: s, ``s >= 1``.  The first chunk must have ``T >= s + d``, else the episode fails.
    """

    def __init__(self, inner: PredictModelServer, *, delay_steps: int, execute_steps: int) -> None:
        if not (execute_steps >= 1 and 0 <= delay_steps <= execute_steps):
            raise ValueError(
                f"need 0 <= delay_steps <= execute_steps, execute_steps >= 1; got {delay_steps=}, {execute_steps=}"
            )
        super().__init__(backlog=64, max_rows=inner.max_rows)
        self.inner = inner
        self.delay, self.horizon = delay_steps, execute_steps
        self._batched = type(inner).predict_batch is not PredictModelServer.predict_batch
        self._eps_state: dict[int, _RowState] = {}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return self.inner.get_action_spec()

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return self.inner.get_observation_spec()

    def get_observation_params(self) -> dict[str, Any]:
        return self.inner.get_observation_params()

    # -- model thread ----------------------------------------------------------------------------------------------

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        failed = self.inner.reset_episodes(rows)
        for i, (rid, episode) in enumerate(zip(rows.ids.tolist(), rows.episodes)):
            if failed and i in failed:
                continue
            rec = None
            if episode.recording_db_path:
                rec = StepRecorder(episode.recording_db_path, episode.session_id, episode.episode_id)
            self._eps_state[rid] = _RowState(rec)
        return failed

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        self.inner.finish_episodes(rows, results)
        for rid in rows.ids.tolist():
            self._eps_state.pop(rid, None)

    def _infer(self, obs: Observation, rows: Rows) -> np.ndarray:
        """The full ``[T, D]`` chunk for one row's observation (bypasses the inner engine's chunk buffer)."""
        if self._batched:
            actions = self.inner.predict_batch(columns.stack([obs]), rows)["actions"][0]
        else:
            actions = self.inner.predict(obs, rows[0])["actions"]
        return np.asarray(actions)

    def _check_chunk(self, chunk: np.ndarray) -> None:
        """A chunk must be ``[T, D]`` with ``T >= s + d``."""
        if chunk.ndim != 2 or chunk.shape[0] < self.horizon + self.delay:
            raise ValueError(
                f"chunk of shape {chunk.shape} is too short: need T >= s + d = {self.horizon} + {self.delay}"
            )

    # -- event loop ------------------------------------------------------------------------------------------------

    async def run(self, ep: EpisodeContext) -> None:
        # TODO(yue): the schedule.  For each request in ep.observations(), with j = request.row.step:
        #   - j == 0: infer the initial chunk (t=-1) and use it at once;
        #   - j % s == 0: start an inference from this observation (ep.start_soon); it lands at j + d
        #     (d == 0: wait for it now);
        #   - at j: land the inference started at j - d, if any;
        #   - reply {"actions": plan.chunk[j - plan.t']} where t' is the step the plan's index 0 maps to;
        #   - record {"step": j, "chunk_src", "chunk_idx", "obs_step"} when st.recorder is set.
        # ep.call(self._infer, request.obs, request.rows) runs the model; self._check_chunk on every chunk.
        raise NotImplementedError

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        recorders = [st.recorder for rid in rows.ids.tolist() if (st := self._eps_state.get(rid)) and st.recorder]
        await super().on_episode_end(rows, results)
        for rec in recorders:  # SQLite I/O off the model thread
            try:
                await _run_in_thread(rec.close)
            except Exception:
                logger.exception("server step rows were not saved")
