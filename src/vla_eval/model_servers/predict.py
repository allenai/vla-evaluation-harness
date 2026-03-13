"""PredictModelServer: unified model server for single and batched inference.

Subclass and override ``predict()`` for single-observation inference, or
``predict_batch()`` for GPU-batched multi-session inference, or both.

.. warning::

    **DRAFT** – Continuous Inference (CI) and Latency-Aware Action Selection
    (LAAS) support is implemented but **has not been tested against a real
    model or environment**.  The request-response (non-CI) path is unaffected.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anyio
import numpy as np
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval.model_servers.base import ModelServer, SessionContext
from vla_eval.model_servers.chunking import ActionChunkBuffer, get_ensemble_fn
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_QUEUE_DEPTH_COOLDOWN: float = 10.0  # minimum seconds between queue depth warnings


@dataclass
class _PendingRequest:
    obs: Observation
    ctx: SessionContext
    _event: anyio.Event = field(default_factory=anyio.Event)
    _result: Action | None = field(default=None, init=False)
    _exception: BaseException | None = field(default=None, init=False)

    def set_result(self, result: Action) -> None:
        self._result = result
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exception = exc
        self._event.set()

    async def wait(self) -> Action:
        await self._event.wait()
        if self._exception is not None:
            raise self._exception
        assert self._result is not None
        return self._result

    @property
    def done(self) -> bool:
        return self._event.is_set()


class PredictModelServer(ModelServer):
    """Model server with blocking ``predict()`` / ``predict_batch()`` interface.

    Override ``predict()`` for single-observation inference, or
    ``predict_batch()`` for GPU-batched inference across concurrent sessions.

    Single vs batched dispatch:
        When ``max_batch_size == 1`` (default), each observation is dispatched
        directly via ``predict()`` in a thread-pool executor.

        When ``max_batch_size > 1``, observations are collected into a queue
        and dispatched as a batch via ``predict_batch()``.  This requires
        ``predict_batch()`` to be overridden; otherwise a ``NotImplementedError``
        is raised at runtime.

    Relationship between predict() and predict_batch():
        - Override ``predict()`` only → single inference.  Setting
          ``max_batch_size > 1`` raises ``NotImplementedError``.
        - Override ``predict_batch()`` only → ``predict()`` automatically
          delegates to ``predict_batch([obs], [ctx])[0]``.
        - Override both → ``predict()`` is used for CI mode and single
          dispatch; ``predict_batch()`` is used for batched dispatch.

    Action chunking:
        When ``chunk_size > 1``, the first call to ``predict()`` should return
        a 2-D array of shape ``(chunk_size, action_dim)``.  Subsequent
        observations consume buffered actions without calling ``predict()``
        again until the buffer is empty.

        If ``predict()`` returns a 1-D array ``(action_dim,)``, chunking is
        bypassed regardless of ``chunk_size``.

    Action ensemble (for overlapping chunks):
        - ``"newest"`` (default): use the latest chunk's action.
        - ``"average"``: element-wise mean of old and new.
        - ``"ema"``: exponential moving average (controlled by ``ema_alpha``).
        - A callable ``(old, new) -> blended`` for custom logic.

    The chunk buffer is **deleted** on each ``episode_start``, allowing
    ``chunk_size`` to change between episodes (see CogACT's ``chunk_size_map``).

    Continuous Inference (CI) – **DRAFT, untested**:
        When ``continuous_inference=True``, the server runs a background loop
        that continuously calls ``predict()`` with the latest observation.
        ``on_observation()`` merely buffers the obs and returns immediately.
        The chunk buffer is **not used** in CI mode; each inference produces
        a single action (or the temporally appropriate one via LAAS).

    Latency-Aware Action Selection (LAAS) – **DRAFT, untested**:
        When ``laas=True`` (requires ``continuous_inference=True``), the server
        skips stale actions in a chunk based on inference latency.  If
        inference took *d* seconds and the environment runs at ``hz`` Hz,
        ``delay_steps = int(d * hz)`` actions are skipped.

    Args:
        chunk_size: Number of actions per inference call (default None = no
            chunking or trimming; the model's raw output is used as-is).
        action_ensemble: Strategy for blending overlapping action chunks.
        ema_alpha: Blend ratio for "ema" ensemble (higher = more weight on new).
        max_batch_size: Maximum observations per batch (default 1 = no batching).
            Setting > 1 requires ``predict_batch()`` to be overridden.
        max_wait_time: Seconds to wait for a full batch before dispatching a
            partial one (default 0.01).  Only used when ``max_batch_size > 1``.
        continuous_inference: Enable CI mode (default False). **DRAFT**.
        laas: Enable LAAS (default False, requires CI). **DRAFT**.
        hz: Environment step frequency for LAAS delay computation.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        action_ensemble: str | Callable[[np.ndarray, np.ndarray], np.ndarray] = "newest",
        ema_alpha: float = 0.5,
        max_batch_size: int = 1,
        max_wait_time: float = 0.01,
        continuous_inference: bool = False,
        laas: bool = False,
        hz: float = 10.0,
    ) -> None:
        self.chunk_size = chunk_size
        self.action_ensemble = action_ensemble
        self.ema_alpha = ema_alpha
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.continuous_inference = continuous_inference
        self.laas = laas
        self.hz = hz

        self._chunk_buffers: dict[str, ActionChunkBuffer] = {}
        self._session_chunk_sizes: dict[str, int] = {}
        # Serialise predict() calls so only one runs on the GPU at a time.
        # The batched path (_dispatch_loop) is already single-threaded; this
        # lock protects the non-batched and CI paths.
        self._predict_lock: asyncio.Lock = asyncio.Lock()
        # Dedicated thread limiter so inference is never starved by
        # unrelated run_sync consumers (e.g. process waits) sharing the
        # default 40-token pool.
        self._thread_limiter: anyio.CapacityLimiter = anyio.CapacityLimiter(1)
        # Batch dispatch state
        self._send_stream: MemoryObjectSendStream[_PendingRequest] | None = None
        self._receive_stream: MemoryObjectReceiveStream[_PendingRequest] | None = None
        self._dispatch_task: asyncio.Task | None = None
        # CI state (per session)
        self._ci_tasks: dict[str, asyncio.Task[None]] = {}
        self._obs_slots: dict[str, tuple[Observation, SessionContext, float]] = {}
        self._obs_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Inference methods — override one or both
    # ------------------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        """Run inference on a single observation.  Blocking call.

        Default implementation delegates to ``predict_batch([obs], [ctx])[0]``.
        Override this for single-observation inference, or leave it to use
        the default delegation when only ``predict_batch()`` is overridden.

        Returns:
            dict with ``"actions"`` key containing the raw model output.
            The base class normalizes the result (float32 cast, chunk trim)
            via ``_normalize_result()`` before further processing.
        """
        if type(self).predict_batch is not PredictModelServer.predict_batch:
            return self.predict_batch([obs], [ctx])[0]
        raise NotImplementedError(f"{type(self).__name__} must override predict() or predict_batch()")

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[Action]:
        """Run batched inference.  Blocking call.

        Required when ``max_batch_size > 1``.  ``len(result)`` must equal
        ``len(obs_batch)``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override predict_batch() to use max_batch_size > 1")

    # ------------------------------------------------------------------
    # Chunking helpers (shared by single / batch / CI paths)
    # ------------------------------------------------------------------

    def _get_chunk_size(self, ctx: SessionContext) -> int | None:
        """Return the effective chunk_size for a session."""
        return self._session_chunk_sizes.get(ctx.session_id, self.chunk_size)

    def _try_serve_from_buffer(self, ctx: SessionContext) -> np.ndarray | None:
        """Return a buffered action if available, else ``None``."""
        cs = self._get_chunk_size(ctx)
        if cs is None:
            return None
        sid = ctx.session_id
        if sid not in self._chunk_buffers:
            ensemble_fn = get_ensemble_fn(self.action_ensemble, self.ema_alpha)
            self._chunk_buffers[sid] = ActionChunkBuffer(cs, ensemble_fn)
        buf = self._chunk_buffers[sid]
        if not buf.empty:
            return buf.pop()
        return None

    def _normalize_result(self, result: Action, ctx: SessionContext) -> Action:
        """Normalize predict() output: cast to float32 and trim to chunk_size."""
        actions = result.get("actions")
        if actions is None:
            return result
        actions = np.asarray(actions, dtype=np.float32)
        cs = self._get_chunk_size(ctx)
        if cs is not None and actions.ndim == 2:
            actions = actions[:cs]
        return {**result, "actions": actions}

    async def _process_and_send(self, result: dict[str, Any], ctx: SessionContext) -> None:
        """Post-process inference result (chunking) and send action."""
        actions = result.get("actions")
        if actions is None:
            await ctx.send_action(result)
            return

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions)

        cs = self._get_chunk_size(ctx)
        if cs is None or actions.ndim == 1:
            await ctx.send_action(result)
            return

        # Push chunk and pop first action
        buf = self._chunk_buffers[ctx.session_id]
        buf.push_chunk(actions)
        action = buf.pop()
        if action is not None:
            await ctx.send_action({"actions": action})

    # ------------------------------------------------------------------
    # on_observation: CI vs single vs batch dispatch
    # ------------------------------------------------------------------

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        """Framework calls this on each observation.

        Dispatch strategy:
        - CI mode: buffer obs for background loop, return immediately.
        - ``max_batch_size > 1``: enqueue for batched dispatch via
          ``predict_batch()``.
        - Otherwise: direct dispatch via ``predict()`` in executor.
        """
        if self.continuous_inference:
            sid = ctx.session_id
            event = self._obs_events.get(sid)
            if event is None:
                logger.warning("CI observation before EPISODE_START session=%s — dropping", sid[:8])
                return
            self._obs_slots[sid] = (obs, ctx, time.monotonic())
            event.set()
            return

        # Serve from chunk buffer if available (skip inference)
        buffered = self._try_serve_from_buffer(ctx)
        if buffered is not None:
            await ctx.send_action({"actions": buffered})
            return

        t0 = time.monotonic()
        if self.max_batch_size > 1:
            result = await self._dispatch_batched(obs, ctx)
        else:
            async with self._predict_lock:
                result = await _run_in_thread(self.predict, obs, ctx, limiter=self._thread_limiter)
        logger.debug(
            "inference step=%d %.1fms session=%s", ctx.step, (time.monotonic() - t0) * 1000, ctx.session_id[:8]
        )

        result = self._normalize_result(result, ctx)
        await self._process_and_send(result, ctx)

    # ------------------------------------------------------------------
    # Batch dispatch (queue + background loop)
    # ------------------------------------------------------------------

    async def _dispatch_batched(self, obs: Observation, ctx: SessionContext) -> Action:
        """Enqueue observation and await batched result."""
        self._ensure_dispatch_loop()
        assert self._send_stream is not None
        req = _PendingRequest(obs=obs, ctx=ctx)
        await self._send_stream.send(req)
        return await req.wait()

    def _ensure_dispatch_loop(self) -> None:
        """Lazily start the dispatch loop, restarting if it crashed."""
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        self._send_stream, self._receive_stream = anyio.create_memory_object_stream()  # element type: _PendingRequest
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        self._dispatch_task.add_done_callback(self._on_dispatch_done)

    def _on_dispatch_done(self, task: asyncio.Task[None]) -> None:
        """Log unexpected dispatch loop termination."""
        if not task.cancelled() and task.exception() is not None:
            logger.error("Batch dispatch loop crashed", exc_info=task.exception())

    async def _dispatch_loop(self) -> None:
        """Pipelined batch dispatch: collect next batch while GPU runs.

        Without pipelining the loop is serial:
            collect → GPU → collect → GPU → ...
        Fast GPUs (e.g. H100) finish before many requests arrive, so
        batches stay small and throughput is low.

        With pipelining we overlap collection and inference:
            collect₀ → GPU₀ + collect₁ → GPU₁ + collect₂ → ...
        _collect continuously receives from the stream while _infer
        processes batches.  After GPU finishes, if pending isn't full
        we wait up to max_wait_time for more.
        """
        assert self._receive_stream is not None
        stream = self._receive_stream
        pending: list[_PendingRequest] = []
        has_items = anyio.Event()
        batch_full = anyio.Event()
        stream_closed = False

        async with anyio.create_task_group() as tg:

            async def _collect() -> None:
                nonlocal has_items, batch_full, stream_closed
                async for req in stream:
                    pending.append(req)
                    has_items.set()
                    if len(pending) >= self.max_batch_size:
                        batch_full.set()
                stream_closed = True
                has_items.set()

            async def _infer() -> None:
                nonlocal pending, has_items, batch_full
                _last_depth_warning = 0.0
                try:
                    while True:
                        await has_items.wait()
                        has_items = anyio.Event()

                        if not pending:
                            if stream_closed:
                                break
                            continue

                        batch = pending[: self.max_batch_size]
                        pending = pending[self.max_batch_size :]

                        if len(pending) > self.max_batch_size * 2:
                            now = time.monotonic()
                            if now - _last_depth_warning >= _QUEUE_DEPTH_COOLDOWN:
                                _last_depth_warning = now
                                logger.warning(
                                    "Batch queue depth %d exceeds 2x max_batch_size (%d). "
                                    "Consider increasing max_batch_size or reducing shard count.",
                                    len(pending),
                                    self.max_batch_size,
                                )

                        obs_batch = [r.obs for r in batch]
                        ctx_batch = [r.ctx for r in batch]
                        try:
                            results = await _run_in_thread(
                                self.predict_batch,
                                obs_batch,
                                ctx_batch,
                                limiter=self._thread_limiter,
                            )
                            if len(results) != len(batch):
                                raise RuntimeError(
                                    f"predict_batch returned {len(results)} results for {len(batch)} inputs"
                                )
                            for req, result in zip(batch, results):
                                req.set_result(result)
                        except Exception as exc:
                            for req in batch:
                                if not req.done:
                                    req.set_exception(exc)

                        # Grace period AFTER GPU: accumulate items for next batch
                        if not stream_closed and len(pending) < self.max_batch_size:
                            batch_full = anyio.Event()
                            with anyio.move_on_after(self.max_wait_time):
                                await batch_full.wait()

                        if pending:
                            has_items.set()
                        elif stream_closed:
                            break
                finally:
                    err = RuntimeError("Dispatch loop terminated")
                    for req in pending:
                        if not req.done:
                            req.set_exception(err)

            tg.start_soon(_collect)
            tg.start_soon(_infer)

    # ------------------------------------------------------------------
    # CI loop — DRAFT, untested
    # ------------------------------------------------------------------

    async def _ci_loop(self, session_id: str) -> None:
        """Background loop: wait for obs → infer → send action → repeat.

        DRAFT — not tested against a real model or environment.
        """
        event = self._obs_events[session_id]
        while True:
            await event.wait()
            event.clear()

            slot = self._obs_slots.get(session_id)
            if slot is None:
                continue

            obs, ctx, obs_time = slot

            try:
                async with self._predict_lock:
                    result = await _run_in_thread(self.predict, obs, ctx, limiter=self._thread_limiter)
            except Exception:
                logger.exception("CI inference error session=%s", session_id)
                continue

            result = self._normalize_result(result, ctx)

            actions = result.get("actions")
            try:
                if actions is None:
                    await ctx.send_action(result)
                    continue

                action = self._pick_action(actions, obs_time)
                await ctx.send_action({"actions": action})
            except Exception:
                logger.exception("CI send_action error session=%s", session_id)
                break

    def _pick_action(self, actions: np.ndarray, obs_time: float) -> np.ndarray:
        """Select a single action from inference output, applying LAAS if enabled.

        DRAFT — not tested.
        """
        if actions.ndim == 1:
            return actions

        if self.laas and self.hz > 0:
            delay = time.monotonic() - obs_time
            delay_steps = int(delay * self.hz)
            idx = min(delay_steps, len(actions) - 1)
            if delay_steps >= len(actions):
                logger.warning(
                    "LAAS: delay_steps=%d >= chunk_size=%d (%.0fms latency at %.0fHz). "
                    "Entire chunk is stale; sending last action.",
                    delay_steps,
                    len(actions),
                    delay * 1000,
                    self.hz,
                )
            return actions[idx]

        # CI without LAAS: use first action in chunk
        return actions[0]

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Reset chunk buffer on episode start and start CI loop if enabled.

        Deletes (not clears) the buffer so it is recreated with the current
        ``chunk_size`` on the next observation.  Subclasses can override
        this and call ``self._session_chunk_sizes[ctx.session_id] = N``
        before ``super()`` to set per-session chunk sizes safely.
        """
        sid = ctx.session_id
        self._chunk_buffers.pop(sid, None)

        if self.continuous_inference:
            await self._stop_ci(sid)
            self._obs_events[sid] = asyncio.Event()
            self._obs_slots.pop(sid, None)
            self._ci_tasks[sid] = asyncio.create_task(self._ci_loop(sid))
            logger.info("CI loop started session=%s laas=%s hz=%.1f", sid, self.laas, self.hz)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        """Clean up chunk buffer and stop CI loop."""
        sid = ctx.session_id
        self._chunk_buffers.pop(sid, None)
        self._session_chunk_sizes.pop(sid, None)

        if self.continuous_inference:
            await self._stop_ci(sid)

    async def _stop_ci(self, session_id: str) -> None:
        """Cancel and await the CI loop for a session."""
        task = self._ci_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, anyio.get_cancelled_exc_class()):
                await task
        self._obs_slots.pop(session_id, None)
        self._obs_events.pop(session_id, None)
