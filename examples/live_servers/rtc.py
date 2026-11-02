"""Real-time chunking (RTC) as an episode-daemon loop: ``RTCServer`` streams each chunk on a time grid (``PUBLISH``)
and refills it ahead of time.  Subclass it and override ``predict``; ``tests/test_example_rtc.py`` covers it."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from vla_eval.model_servers.base import Row, Rows
from vla_eval.model_servers.daemons import DaemonModelServer, EpisodeClosed, EpisodeContext, ObservationRequest
from vla_eval.types import Observation


class RTCServer(DaemonModelServer):
    """Streams each chunk on a ``1/hz`` grid anchored at its observation's admission: slot ``k`` is published only
    within ``[t0 + k/hz, t0 + (k+1)/hz)`` (at a tick, or as its chunk lands after a tick that sent nothing), expired
    slots are skipped, and the newest observation is inferred once ``refill_at`` or fewer actions remain.  Live clients
    with ``publish`` and ``episode_error`` only.  Skipping the slots its latency outlived is LAAS-style selection.

    Override ``predict`` to return a ``[T, D]`` chunk (model thread).  ``warmup_obs``: predicted in the first
    ``reset_episodes`` (call ``super()``): a cold inference longer than ``chunk_size / hz`` is dropped unpublished.
    """

    requires = frozenset({"publish", "episode_error"})

    def __init__(
        self, *, chunk_size: int = 8, hz: float = 10.0, refill_at: int = 3, warmup_obs: Observation | None = None
    ) -> None:
        if not (math.isfinite(hz) and hz > 0 and chunk_size >= 1 and refill_at >= 0):
            raise ValueError("RTCServer needs a finite hz > 0, chunk_size >= 1 and refill_at >= 0")
        super().__init__(backlog="latest", control_hz=hz)
        self.chunk_size, self.hz, self.refill_at = chunk_size, hz, refill_at
        self.warmup_obs = warmup_obs

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        if self.warmup_obs is not None:  # once, before the first episode is ready; a failure retries next start
            self.predict(self.warmup_obs, rows[0])
            self.warmup_obs = None
        return None

    def predict(self, obs: Observation, row: Row) -> np.ndarray:
        """A ``[T, D]`` action chunk for one observation.  Blocking."""
        raise NotImplementedError(f"{type(self).__name__} must override predict()")

    async def run(self, ep: EpisodeContext) -> None:
        plan: deque[np.ndarray] = deque()
        plan_t0, consumed, busy = 0.0, 0, False  # the plan's anchor; slots passed; one refill at a time
        warned = False  # a dropped chunk warns once per episode
        sent = False  # this tick published; else a chunk landing before the next tick publishes its current slot

        def current() -> np.ndarray | None:
            """Drop expired slots (each at most once); take the current interval's slot, never a future one."""
            nonlocal consumed
            due = math.floor((ep.clock.now() - plan_t0) * self.hz)
            while consumed < due and plan:
                plan.popleft()
                consumed += 1
            if plan and consumed == due:
                consumed += 1  # committed before any await
                return plan.popleft()
            return None

        async def infer(request: ObservationRequest) -> None:
            nonlocal busy, plan_t0, consumed, warned, sent
            try:
                chunk = np.asarray(await ep.call(self.predict, request.obs, request.row), np.float32)
                chunk = chunk[: self.chunk_size]
                if chunk.ndim != 2 or not len(chunk):
                    raise ValueError(f"predict must return a [T, D] chunk, got shape {chunk.shape}")
                late = ep.clock.now() - request.admitted_at
                skip = max(0, math.floor(late * self.hz))
                if skip >= len(chunk):  # every slot expired: discarded
                    log = ep.logger.debug if warned else ep.logger.warning
                    warned = True
                    log(
                        "inference %.0f ms exceeds the chunk horizon %.0f ms; chunk dropped",
                        late * 1e3,
                        len(chunk) / self.hz * 1e3,
                    )
                    return
                plan.clear()
                plan.extend(chunk[skip:])
                plan_t0, consumed = request.admitted_at, skip
            except EpisodeClosed:
                return
            except Exception as exc:
                await request.error(exc)  # this observation failed; the episode goes on
                return
            finally:
                busy = False
            if not sent and (action := current()) is not None:
                sent = True
                await ep.publish({"actions": action})

        async for _ in ep.clock.every(1.0 / self.hz, first="now"):
            if ep.closing:
                return
            sent = False
            action = current()
            if not busy and len(plan) + (action is not None) <= self.refill_at:
                if (request := ep.receive_nowait()) is not None:
                    busy = True
                    ep.start_soon(infer, request)
            if action is not None:
                sent = True
                await ep.publish({"actions": action})
