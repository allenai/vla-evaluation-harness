"""PredictModelServer: override ``predict(obs, row)`` (one episode's observation), or ``predict_batch(obs, rows)``
(column batch, a row per episode) when batched inference beats per-row; the default calls ``predict`` per row."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import anyio
import numpy as np
from anyio.abc import TaskGroup
from anyio.lowlevel import checkpoint

from vla_eval import columns
from vla_eval.types import Action, Batch, EpisodeResult, Observation
from vla_eval.model_servers.base import ModelServer, Row, Rows

logger = logging.getLogger(__name__)

_ENDED = RuntimeError("the episode ended")
_STOPPED = RuntimeError("model server stopped")


@dataclass(eq=False)
class _Request:
    obs: Batch
    rows: Rows
    wait: bool  # may wait max_wait_time for other connections' requests
    done: anyio.Event = field(default_factory=anyio.Event)
    actions: Batch = field(default_factory=dict)
    errors: dict[int, Exception] = field(default_factory=dict)
    queued_at: float = field(default_factory=anyio.current_time)


class PredictModelServer(ModelServer):
    """Args:
    chunk_size: Actions per inference; the rest of each ``[T, D]`` chunk is served from a per-episode buffer.
    max_batch_size: Most rows per ``predict_batch`` call (``None``: no limit); raise it to batch episodes.
    max_wait_time: Seconds to wait for other connections' requests before inferring a partial batch.
    max_rows: Most concurrent episodes (``None``: no limit); 1 for a model whose state is not per episode.

    The hooks below run on one model thread, in arrival order; ``__init__`` runs on the caller's thread, so
    create thread-bound state (CUDA graphs, JAX) in the first ``reset_episodes`` / ``predict``.
    """

    def __init__(
        self,
        *,
        chunk_size: int | None = None,
        max_batch_size: int | None = 1,
        max_wait_time: float = 0.01,
        max_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_rows = max_rows
        self._chunked = np.zeros(0, bool)  # by row id: the episode has a chunk size
        self._buffers: dict[int, deque[Any]] = {}  # chunked rows' remaining actions
        self._queue: list[_Request] = []
        self._wake = anyio.Event()

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
        """One observation of the episode in ``row``.  Blocking."""
        raise NotImplementedError(f"{type(self).__name__} must override predict() or predict_batch()")

    def predict_batch(self, obs: Batch, rows: Rows) -> Batch:
        """Observation columns of ``rows`` to action columns, a row each.  Blocking."""
        actions, errors = self._predict_rows(obs, rows)
        if errors:
            raise next(iter(errors.values()))
        return actions

    # ------------------------------------------------------------------
    # ModelServer callbacks
    # ------------------------------------------------------------------

    async def on_serve_start(self, tg: TaskGroup) -> None:
        await super().on_serve_start(tg)
        self._queue, self._wake = [], anyio.Event()
        tg.start_soon(self._dispatch)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        for episode in rows.episodes:
            episode.chunk_size = self.chunk_size
        failed = await self.run_on_model_thread(self.reset_episodes, rows)
        if len(self._chunked) < len(self._table):
            self._chunked = np.concatenate([self._chunked, np.zeros(len(self._table) - len(self._chunked), bool)])
        for rid, episode in zip(rows.ids.tolist(), rows.episodes):
            self._chunked[rid] = episode.chunk_size is not None
        return failed

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        idx = rows.ids
        if self._chunked[idx].all() and all(self._buffers.get(i) for i in idx.tolist()):  # no inference to wait for
            await self.send_action(
                rows, columns.stack([{"actions": self._buffers[i].popleft()} for i in idx.tolist()])
            )
            return
        vector = rows._pending is not None and rows._pending.vector
        req = _Request(obs, rows, wait=len(rows) == 1 and not vector)
        self._queue.append(req)
        self._wake.set()
        await req.done.wait()
        await self._reply(rows, req.actions, req.errors)

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        try:
            await self.run_on_model_thread(self.finish_episodes, rows, results)
        finally:
            for rid in rows.ids.tolist():
                self._buffers.pop(rid, None)

    async def _reply(self, rows: Rows, actions: Batch, errors: dict[int, Exception]) -> None:
        if not errors:
            await self.send_action(rows, actions)
            return
        ok = [i for i in range(len(rows)) if i not in errors]
        if ok:
            await self.send_action(rows.take(ok), actions)
        by_error: dict[int, tuple[Exception, list[int]]] = {}
        for i, exc in errors.items():
            by_error.setdefault(id(exc), (exc, []))[1].append(i)
        for exc, positions in by_error.values():
            await self.send_error(rows.take(positions), exc)

    # ------------------------------------------------------------------
    # Dispatch: coalesce requests, serve chunk buffers, infer
    # ------------------------------------------------------------------

    async def _dispatch(self) -> None:
        try:
            await self._dispatch_loop()
        finally:  # shutdown: no request waits forever
            _fail(self._queue, _STOPPED)
            self._queue = []

    async def _dispatch_loop(self) -> None:
        while True:
            await self._wake.wait()
            self._wake = anyio.Event()
            while self._queue:
                # wait for single-episode requests of other connections; a vector step is a whole step already
                if self.max_wait_time > 0 and not self._full() and all(r.wait for r in self._queue):
                    deadline = self._queue[0].queued_at + self.max_wait_time  # from the oldest request's arrival
                    with anyio.CancelScope(deadline=deadline):
                        while not self._full() and all(r.wait for r in self._queue):
                            await self._wake.wait()
                            self._wake = anyio.Event()
                batch = [self._queue.pop(0)]
                rows = len(batch[0].rows)
                limit = self.max_batch_size
                while self._queue and (limit is None or rows + len(self._queue[0].rows) <= limit):
                    rows += len(self._queue[0].rows)
                    batch.append(self._queue.pop(0))
                await self._run(batch)

    def _full(self) -> bool:
        return self.max_batch_size is not None and sum(len(r.rows) for r in self._queue) >= self.max_batch_size

    async def _run(self, batch: list[_Request]) -> None:
        if len(batch) > 1:
            try:
                obs = columns.concat([r.obs for r in batch])
            except Exception:  # requests of different shapes: one at a time
                try:
                    for r in batch:
                        await self._run([r])
                finally:  # cancelled: the rest fail
                    _fail(batch, _STOPPED)
                return
            rows = _merge([r.rows for r in batch])
        else:
            obs, rows = batch[0].obs, batch[0].rows
        try:
            actions, errors = await self._infer(obs, rows)
            start = ok_seen = 0
            for r in batch:
                n = len(r.rows)
                r.errors = {i - start: errors[i] for i in range(start, start + n) if i in errors}
                ok = n - len(r.errors)
                r.actions = actions if len(batch) == 1 else columns.take(actions, range(ok_seen, ok_seen + ok))
                ok_seen += ok
                start += n
            for r in batch:
                r.done.set()
        except Exception as exc:
            logger.exception("dispatch failed")
            _fail(batch, exc)
        finally:  # cancelled
            _fail(batch, _STOPPED)

    async def _infer(self, obs: Batch, rows: Rows) -> tuple[Batch, dict[int, Exception]]:
        """The actions of every position that did not fail, in order, and the failures by position.  One
        model-thread call."""
        n = len(rows)
        ids = rows.ids
        await checkpoint()  # the last await before the model call, so a row ending meanwhile is filtered here
        current = self._current(rows)
        errors: dict[int, Exception] = {}
        if not current.all():  # ended while queued: not inferred
            errors = dict.fromkeys(np.flatnonzero(~current).tolist(), _ENDED)
        chunked = np.flatnonzero(self._chunked[ids] & current).tolist()
        served: dict[int, Any] = {}
        for i in chunked:
            buffer = self._buffers.get(int(ids[i]))
            if buffer:
                served[i] = buffer.popleft()
        todo = [i for i in range(n) if i not in served and i not in errors] if served or errors else list(range(n))
        parts: list[tuple[list[int], Batch]] = []
        if todo:
            failed, parts = await self._run_submitted(self._work, obs, rows, todo)
            errors.update(failed)
        if not chunked:
            return columns.concat([p for _, p in parts]), errors
        # A chunked row returns its chunk's first action and keeps the rest; every row returns "actions" only.
        current = self._current(rows)  # a row that ended meanwhile keeps no buffer: its id may run another episode
        action: dict[int, Any] = {i: a for i, a in served.items() if i not in errors}
        for positions, out in parts:
            for j, i in enumerate(positions):
                try:
                    a = out["actions"][j]
                    cs = rows._episodes[i].chunk_size
                    if cs is not None and np.ndim(a) == 2:
                        a = a[:cs]
                        if current[i]:
                            self._buffers[int(ids[i])] = deque(a[1:])
                        a = a[0]
                    action[i] = a
                except Exception as exc:  # e.g. an empty chunk: this row alone
                    errors[i] = exc
        return columns.stack([{"actions": action[i]} for i in range(n) if i not in errors]), errors

    def _work(
        self, obs: Batch, rows: Rows, todo: list[int]
    ) -> tuple[dict[int, Exception], list[tuple[list[int], Batch]]]:
        """On the model thread: predict the ``todo`` positions, in slices of ``max_batch_size``."""
        errors: dict[int, Exception] = {}
        n = len(rows)
        parts: list[tuple[list[int], Batch]] = []
        batched = type(self).predict_batch is not PredictModelServer.predict_batch
        size = (self.max_batch_size if batched else None) or max(len(todo), 1)  # predict runs row by row anyway
        for start in range(0, len(todo), size):
            group = todo[start : start + size]
            sub, sub_rows = (obs, rows) if len(group) == n else (columns.take(obs, group), rows.take(group))
            try:
                if not batched:
                    out, failed = self._predict_rows(sub, sub_rows)
                else:
                    out, failed = self.predict_batch(sub, sub_rows), {}
                    if not isinstance(out, dict) or columns.num_rows(out) != len(group):
                        raise RuntimeError(
                            f"predict_batch returned {type(out).__name__} for {len(group)} rows: it takes and returns "
                            "column batches (a row per episode), no longer lists of dicts"
                        )
                    out = _float32(out)
            except Exception as exc:
                errors.update(dict.fromkeys(group, exc))
                continue
            errors.update({group[j]: exc for j, exc in failed.items()})
            parts.append(([i for j, i in enumerate(group) if j not in failed], out))
        return errors, parts

    def _predict_rows(self, obs: Batch, rows: Rows) -> tuple[Batch, dict[int, Exception]]:
        """:meth:`predict` per row; a row's exception fails it alone."""
        out: list[Action] = []
        failed: dict[int, Exception] = {}
        for i, row_obs in enumerate(columns.split(obs, len(rows))):
            try:
                action = _float32(self.predict(row_obs, rows[i]))
                if out and action.keys() != out[0].keys():
                    raise RuntimeError(f"action keys {sorted(action)} differ from the batch's {sorted(out[0])}")
                out.append(action)
            except Exception as exc:
                failed[i] = exc
        return columns.stack(out), failed


def _fail(requests: list[_Request], exc: Exception) -> None:
    """Complete the requests not completed yet with ``exc`` for every row."""
    for r in requests:
        if not r.done.is_set():
            r.actions, r.errors = {}, dict.fromkeys(range(len(r.rows)), exc)
            r.done.set()


def _merge(parts: list[Rows]) -> Rows:
    """The rows of several requests, one after another (for inference: no replies through them)."""
    return Rows(
        np.concatenate([r.ids for r in parts]),
        np.concatenate([r.step for r in parts]),
        np.concatenate([r._episodes for r in parts]),
        np.concatenate([r._gen for r in parts]),
    )


def _float32(actions: Batch) -> Batch:
    """``actions`` as float32 (an array, or each item of a list column), as every model's actions travel."""
    a = actions.get("actions")
    if a is None:
        return actions
    if isinstance(a, list):
        return {**actions, "actions": [np.asarray(x, np.float32) for x in a]}
    return {**actions, "actions": np.asarray(a, np.float32)}
