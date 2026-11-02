"""LiveModelServer: continuous inference for real-time episodes.  Each episode's latest observation is inferred in
a loop, one row at a time, and its action sent as soon as it is ready; observations that arrive meanwhile coalesce."""

from __future__ import annotations

import logging
import time
import anyio
import numpy as np
from anyio.abc import TaskGroup

from vla_eval import columns
from vla_eval.types import Action, Batch, EpisodeResult, Observation
from vla_eval.model_servers.base import ModelServer, Row, Rows

logger = logging.getLogger(__name__)


class _Loop:
    """One episode's inference loop: its latest observation not inferred yet, with its arrival time."""

    def __init__(self) -> None:
        self.latest: tuple[Batch, Rows, float] | None = None
        self.wake = anyio.Event()
        self.stopped = False
        self.exited = anyio.Event()


class LiveModelServer(ModelServer):
    """Args:
    chunk_size: Actions of each ``[T, D]`` chunk to choose from; the rest is dropped (no buffer).
    laas: Send the action ``int(latency * hz)`` steps into the chunk, latency counted from the observation's
        arrival (else the first).
    hz: The control rate LAAS assumes.

    Override ``predict`` (and ``select`` to choose the action otherwise).  The hooks run on one model thread;
    one inference at a time, across episodes; no batching.
    """

    supports_vector_step = False

    def __init__(self, *, chunk_size: int | None = None, laas: bool = False, hz: float = 10.0) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.laas = laas
        self.hz = hz
        self._loops: dict[int, _Loop] = {}  # by row id
        self._tg: TaskGroup | None = None

    # ------------------------------------------------------------------
    # Override these
    # ------------------------------------------------------------------

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        """Reset the state of ``rows`` (``episode.config``: the ``EPISODE_START`` payload).  Blocking.
        Return ``{position: exception}`` for starts that failed, or raise to fail them all."""
        return None

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]``: the ``EPISODE_END`` payload).  Blocking."""

    def predict(self, obs: Observation, row: Row) -> Action:
        """One observation of the episode in ``row``; ``"actions"``, if any, is ``[T, D]`` or ``[D]``.  Blocking."""
        raise NotImplementedError(f"{type(self).__name__} must override predict()")

    def select(self, action: Action, latency: float, row: Row) -> Action:
        """The action to send from ``predict``'s output, ``latency`` seconds after its observation arrived:
        the chunk trimmed to ``chunk_size``, float32, its LAAS step ``int(latency * hz)`` (else its first)."""
        if "actions" not in action:
            return action
        a = np.asarray(action["actions"])
        if a.ndim != 2:
            return {**action, "actions": np.asarray(a, np.float32)}
        a = np.asarray(a[: row.episode.chunk_size], np.float32)
        steps = 0
        if self.laas and self.hz > 0:
            steps = int(latency * self.hz)
            if steps >= len(a):
                logger.warning("LAAS: %d steps late for a %d-step chunk; sending its last action", steps, len(a))
        return {**action, "actions": a[min(steps, len(a) - 1)]}

    # ------------------------------------------------------------------
    # ModelServer callbacks
    # ------------------------------------------------------------------

    async def on_serve_start(self, tg: TaskGroup) -> None:
        await super().on_serve_start(tg)
        self._tg = tg

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        for episode in rows.episodes:
            episode.chunk_size = self.chunk_size
        failed = await self.run_on_model_thread(self.reset_episodes, rows) or {}
        assert self._tg is not None, "serve_async calls on_serve_start first"
        for i, rid in enumerate(rows.ids.tolist()):
            if i not in failed:
                self._loops[rid] = loop = _Loop()
                self._tg.start_soon(self._run, loop, rid)
        return failed

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        t = time.monotonic()  # admission: LAAS counts the model queue too
        for i in range(len(rows)):
            o, r = (obs, rows) if len(rows) == 1 else (columns.take(obs, [i]), rows.take([i]))
            loop = self._loops[int(r.ids[0])]
            loop.latest = (o, r, t)
            loop.wake.set()

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        loops = [loop for rid in rows.ids.tolist() if (loop := self._loops.pop(rid, None)) is not None]
        for loop in loops:
            loop.stopped = True
            loop.wake.set()
        for loop in loops:  # an inference in flight finishes before finish_episodes; its action is dropped
            await loop.exited.wait()
        await self.run_on_model_thread(self.finish_episodes, rows, results)

    # ------------------------------------------------------------------

    async def _run(self, loop: _Loop, rid: int) -> None:
        """Infer the row's latest observation, send its action, repeat."""
        try:
            while True:
                await loop.wake.wait()
                loop.wake = anyio.Event()  # before inferring: an observation arriving meanwhile wakes the next
                if loop.stopped:
                    return
                if loop.latest is None:
                    continue
                (obs, rows, t), loop.latest = loop.latest, None
                try:
                    action = await self.run_on_model_thread(self._infer, columns.split(obs, 1)[0], rows[0], t)
                    if loop.stopped:
                        return
                    await self.send_action(rows, columns.stack([action]))
                except Exception as exc:  # inference or send: this observation fails; the loop goes on
                    if loop.stopped:
                        return
                    await self.send_error(rows, exc)
        except Exception:
            logger.exception("continuous inference stopped row=%d", rid)
        finally:
            loop.exited.set()

    def _infer(self, obs: Observation, row: Row, t: float) -> Action:
        action = self.predict(obs, row)
        return self.select(action, time.monotonic() - t, row)
