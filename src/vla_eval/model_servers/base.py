"""Model-server contract: every running episode is a row.  ``serve`` calls the async callbacks with batches of rows;
the server replies with ``send_action`` / ``send_error``.  ``PredictModelServer`` (predict.py, batched
request/response) and ``LiveModelServer`` (live.py, continuous inference) build on it."""

from __future__ import annotations

import heapq
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import anyio
from anyio.lowlevel import checkpoint
import numpy as np
import numpy.typing as npt
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval import columns
from vla_eval.specs import DimSpec
from vla_eval.types import Batch, EpisodeResult

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(eq=False)
class Episode:
    """One running episode.  ``session_id``/``episode_id``/``recording_db_path`` open a ``StepRecorder`` on
    its recording; ``reset_episodes`` may override ``chunk_size`` (e.g. per task suite)."""

    config: dict[str, Any]
    episode_id: str
    session_id: str
    eval_id: str = ""
    recording_db_path: str = ""
    chunk_size: int | None = None


class Row(NamedTuple):
    id: int
    step: int
    is_first: bool
    episode: Episode


class _Pending:
    """The replies one ``OBSERVATION`` / ``VECTOR_STEP`` waits for (framework-private).  ``forward``: a live
    episode's sink, each reply sent on at once as ``(action, None)`` or ``(None, error)``."""

    def __init__(
        self,
        n: int,
        *,
        vector: bool = False,
        forward: Callable[[Any, Exception | None], Awaitable[None]] | None = None,
    ) -> None:
        self.pos = np.arange(n)  # the positions of the rows on_observation gets
        self.answered = np.zeros(n, bool)
        self.left = n
        self.parts: list[tuple[np.ndarray, Batch]] = []  # (positions, their actions)
        self.errors: dict[int, Exception] = {}
        self.done = anyio.Event()
        self.vector = vector
        self.forward = forward

    def result(self) -> Batch:
        """The answered positions' actions, in position order."""
        if not self.parts:
            return {}
        if len(self.parts) == 1 and self.parts[0][0] is self.pos:  # one reply for every row
            return self.parts[0][1]
        pos = np.concatenate([p for p, _ in self.parts])
        actions = self.parts[0][1] if len(self.parts) == 1 else columns.concat([a for _, a in self.parts])
        return actions if (np.diff(pos) > 0).all() else columns.take(actions, np.argsort(pos))


class Rows:
    """The rows of one call, as of that call.  ``ids``: stable while an episode runs, reused after, below the peak
    episode count, so per-episode state fits in arrays indexed by them.  ``step``: the episode's prior observations."""

    def __init__(
        self,
        ids: npt.NDArray[np.int64],
        step: npt.NDArray[np.int64],
        episodes: np.ndarray,
        gen: npt.NDArray[np.int64],
        pending: _Pending | None = None,
        pos: npt.NDArray[np.int64] | None = None,
    ) -> None:
        self.ids = ids
        self.step = step
        self._episodes = episodes  # object array: each row's Episode
        self._gen = gen  # each row's incarnation, to tell a reused row id apart
        self._pending = pending  # the observation these rows came with
        self._pos = pos  # their positions in it

    @property
    def is_first(self) -> npt.NDArray[np.bool_]:
        return self.step == 0

    @property
    def episodes(self) -> list[Episode]:
        return list(self._episodes)

    def take(self, positions: Sequence[int] | np.ndarray) -> Rows:
        """These rows, in this order, still bound to the observation they came with."""
        idx = np.asarray(positions, np.int64)
        pos = None if self._pos is None else self._pos[idx]
        return Rows(self.ids[idx], self.step[idx], self._episodes[idx], self._gen[idx], self._pending, pos)

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[Row]:
        return (self[i] for i in range(len(self)))

    def __getitem__(self, i: int) -> Row:
        return Row(int(self.ids[i]), int(self.step[i]), bool(self.step[i] == 0), self._episodes[i])


class ModelServer(ABC):
    """What ``serve`` drives: implement ``on_observation`` and answer each row with ``send_action`` /
    ``send_error``, now or later.  Load weights and warm up in ``__init__`` (call ``super().__init__()``):
    connections are accepted as soon as it returns."""

    max_rows: int | None = None
    """Most concurrent episodes (``None``: no limit); starts beyond it fail."""

    supports_vector_step: bool = True
    """Whether clients may send ``VECTOR_STEP`` (HELLO capability ``vector_step``)."""

    def __init__(self) -> None:
        # Every model call on one thread, one at a time: CUDA graphs (torch.compile) and JAX are thread-bound.
        # The server owns it for its whole life (serve_async may run it again); serve() shuts it down.
        self._model_thread = ThreadPoolExecutor(1, thread_name_prefix="model")
        self._table = np.empty(0, object)  # by row id: its Episode, None when free
        self._gen = np.zeros(0, np.int64)  # by row id: bumped at start and end, so stale Rows are told apart
        self._steps = np.zeros(0, np.int64)  # by row id: observations received
        self._closing = np.zeros(0, bool)
        self._free: list[int] = []  # ended rows' ids, reused smallest first
        self._size = 0  # row ids ever used

    async def run_on_model_thread(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` on the server's one model thread, after the calls submitted before it."""
        await checkpoint()  # a cancelled caller submits nothing
        return await self._run_submitted(fn, *args)

    async def _run_submitted(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` submitted with no await first, so what the caller checked still holds at submission."""
        future = self._model_thread.submit(fn, *args)
        with anyio.CancelScope(shield=True):  # once submitted, returns only after fn finishes
            return await _run_in_thread(future.result)

    async def on_serve_start(self, tg: TaskGroup) -> None:
        """Called by ``serve_async`` before accepting connections; *tg* lives as long as the server.
        Overrides call ``super()`` first."""

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        """Episodes start in ``rows`` (``episode.config``: the ``EPISODE_START`` payload).  Return
        ``{position: exception}`` for starts that failed, or raise to fail them all."""
        return None

    @abstractmethod
    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        """Row ``i`` of ``obs`` is ``rows[i]``'s next observation; every observation comes once, in order.
        Answer each row with ``send_action`` / ``send_error``; an exception fails the rows not answered yet."""

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]``: the ``EPISODE_END`` payload, ``status`` and
        ``failure_reason`` included); their replies are dropped from now on.  Errors are logged."""

    async def send_action(self, rows: Rows, actions: Batch) -> None:
        """Answer ``rows`` (of an ``on_observation`` call, or taken from them) with ``actions``, a row each.
        A row already answered, or whose episode ended, is dropped."""
        if columns.num_rows(actions) not in (None, len(rows)):
            raise ValueError(f"{columns.num_rows(actions)} actions for {len(rows)} rows")
        keep = self._claim(rows)
        if keep is None:
            return
        if len(keep) < len(rows):
            actions = columns.take(actions, keep)
        assert rows._pending is not None and rows._pos is not None
        p = rows._pending
        if p.forward is not None:
            for action in columns.split(actions, len(keep)):
                await p.forward(action, None)
            return
        p.parts.append((rows._pos if len(keep) == len(rows) else rows._pos[keep], actions))
        if not p.left:
            p.done.set()

    async def send_error(self, rows: Rows, error: Exception) -> None:
        """Fail ``rows``' observations with ``error``; as ``send_action``, answered or ended rows drop."""
        keep = self._claim(rows)
        if keep is None:
            return
        assert rows._pending is not None and rows._pos is not None
        p = rows._pending
        if p.forward is not None:
            for _ in keep:
                await p.forward(None, error)
            return
        p.errors.update(dict.fromkeys(rows._pos[keep].tolist(), error))
        if not p.left:
            p.done.set()

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Declare the action output format of this model server.

        Returns a ``{component_name: DimSpec}`` dict describing what this
        server produces.  The orchestrator compares this against the
        benchmark's action spec and warns on mismatches.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_action_spec()")

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation input format this model server expects.

        Returns a ``{component_name: DimSpec}`` dict describing what this
        server needs from the benchmark.  The orchestrator warns when the
        benchmark doesn't provide a declared component.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_observation_spec()")

    def get_observation_params(self) -> dict[str, Any]:
        """Declare observation requirements for this model.

        Returned params are sent in the HELLO response and auto-merged into
        benchmark params by the orchestrator. This lets the model server
        tell the benchmark what observation data it needs (e.g. wrist images,
        proprioceptive state) without requiring manual ``--param`` flags.

        Override in subclasses to auto-detect from model config.
        """
        return {}

    # ------------------------------------------------------------------
    # Framework side (serve.py): rows, their episodes and observations
    # ------------------------------------------------------------------

    def _claim(self, rows: Rows) -> np.ndarray | None:
        """Positions of ``rows`` to answer (not answered yet, episode still running), marked answered."""
        p, pos = rows._pending, rows._pos
        if p is None or pos is None:
            raise RuntimeError("send_action/send_error take the rows of an on_observation call")
        if pos is p.pos and p.left == len(pos) and (self._gen[rows.ids] == rows._gen).all():  # all rows, first reply
            p.answered[:] = True
            p.left = 0
            return pos
        ok = ~p.answered[pos] & (self._gen[rows.ids] == rows._gen)
        if not ok.all():
            if (dup := p.answered[pos]).any():
                logger.warning("dropped %d replies to rows answered already", int(dup.sum()))
            if not ok.any():
                return None
        keep = np.flatnonzero(ok)
        p.answered[pos[keep]] = True
        left = len(p.answered) - int(np.count_nonzero(p.answered))
        if p.left - left < len(keep):  # a position listed twice: its first listing answers it
            keep = keep[np.sort(np.unique(pos[keep], return_index=True)[1])]
        p.left = left
        return keep

    def _current(self, rows: Rows) -> npt.NDArray[np.bool_]:
        """Which of ``rows`` still run the episode they had."""
        return self._gen[rows.ids] == rows._gen

    def _rows(self, ids: Sequence[int] | np.ndarray, pending: _Pending | None = None) -> Rows:
        idx = np.asarray(ids, np.int64)
        pos = None if pending is None else pending.pos
        return Rows(idx, self._steps[idx], self._table[idx], self._gen[idx], pending, pos)

    def _alloc(self) -> int:
        if self._free:
            return heapq.heappop(self._free)
        rid, self._size = self._size, self._size + 1
        if rid == len(self._table):
            grow = max(16, len(self._table))
            self._table = np.concatenate([self._table, np.full(grow, None, object)])
            self._gen = np.concatenate([self._gen, np.zeros(grow, np.int64)])
            self._steps = np.concatenate([self._steps, np.zeros(grow, np.int64)])
            self._closing = np.concatenate([self._closing, np.zeros(grow, bool)])
        return rid

    def _release(self, rid: int) -> None:
        self._table[rid] = None
        self._closing[rid] = False
        heapq.heappush(self._free, rid)

    async def _start_episodes(self, starts: list[tuple[dict[str, Any], str]]) -> list[int | Exception]:
        """Start an episode per ``(EPISODE_START payload, default session id)``; its row id or why it failed."""
        out: list[int | Exception] = []
        ids: list[int] = []
        for config, session_id in starts:
            if self.max_rows is not None and self._running() >= self.max_rows:
                out.append(RuntimeError(f"{type(self).__name__} runs at most {self.max_rows} episodes at once"))
                continue
            rec = config.get("recording") or {}
            rid = self._alloc()
            self._table[rid] = Episode(
                config=config,
                episode_id=rec.get("eid") or str(uuid.uuid4()),
                session_id=rec.get("sid") or session_id,
                eval_id=rec.get("eval_id") or "",
                recording_db_path=rec.get("db_path") or "",
            )
            self._gen[rid] += 1
            self._steps[rid] = 0
            ids.append(rid)
            out.append(rid)
        if not ids:
            return out
        try:
            failed = await self.on_episode_start(self._rows(ids)) or {}
        except Exception as exc:
            failed = dict.fromkeys(range(len(ids)), exc)
        except BaseException:  # cancelled: none of them started
            for rid in ids:
                self._release(rid)
            raise
        bad = {ids[i]: exc for i, exc in failed.items()}
        for rid, exc in bad.items():
            logger.warning("on_episode_start failed for row %d: %s: %s", rid, type(exc).__name__, exc)
            self._release(rid)
        return [bad.get(x, x) if isinstance(x, int) else x for x in out]

    async def _end_episodes(self, ids: Sequence[int], results: list[EpisodeResult]) -> None:
        """End these rows' episodes and free the rows; a row ends once, even if listed twice."""
        n = self._size
        ending = {i: r for i, r in zip(ids, results) if 0 <= i < n and self._table[i] is not None}
        ending = {i: r for i, r in ending.items() if not self._closing[i]}
        if not ending:
            return
        rows = self._rows(list(ending))
        self._closing[rows.ids] = True
        self._gen[rows.ids] += 1  # replies from now on drop
        with anyio.CancelScope(shield=True):  # once begun, an end runs on_episode_end before the rows are freed
            try:
                await self.on_episode_end(rows, list(ending.values()))
            except Exception:
                logger.exception("on_episode_end failed")
            finally:
                for rid in ending:
                    self._release(rid)

    async def _observe(self, obs: Batch, ids: Sequence[int] | np.ndarray, pending: _Pending) -> None:
        """Hand ``obs`` (a row per id) to ``on_observation``; its replies go to ``pending``."""
        rows = self._rows(ids, pending)
        self._steps[rows.ids] += 1
        try:
            await self.on_observation(obs, rows)
        except Exception as exc:
            logger.exception("on_observation failed")
            await self.send_error(rows, exc)

    def _running(self) -> int:
        return self._size - len(self._free)
