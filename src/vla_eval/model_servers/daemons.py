"""Episode daemons: ``run(ep)`` once per episode owns its loop (when to infer, reply or publish).  A sync episode's
clock is sim time, advanced ``step_dt`` per env step; a live one's is wall time.  Guard rails, not a scheduler: one
mailbox, one model thread, drain, fail."""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypeVar

import anyio
from anyio.lowlevel import checkpoint
from chrono_daemon import Clock, SimClock, WallClock

from vla_eval import columns
from vla_eval.model_servers.base import (
    CAPABILITIES,
    Episode,
    EpisodeClosed,
    MissingCapabilities,
    ModelServer,
    Row,
    Rows,
    _leaves,
    _Link,
)
from vla_eval.types import Action, Batch, EpisodeResult, Observation

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

__all__ = [
    "Clock",
    "DaemonModelServer",
    "EpisodeClosed",
    "EpisodeContext",
    "EpisodeDaemons",
    "Mailbox",
    "MailboxOverflow",
    "ObservationRequest",
    "SimClock",
    "WallClock",
    "episode_daemon",
]

logger = logging.getLogger(__name__)

SETTLE_ROUNDS = 8  # checkpoints a woken sync task gets, after each wake, to reach its next sleep or ep.call

T = TypeVar("T")
Mode = Literal["sync", "live"]
Backlog = Literal["latest"] | int
Run = Callable[["EpisodeContext"], Awaitable[None]]
Reset = Callable[[Rows], "dict[int, Exception] | None"]
Finish = Callable[[Rows, "list[EpisodeResult]"], None]


class MailboxOverflow(RuntimeError):
    """A FIFO mailbox's backlog was exceeded; fails the episode."""


@dataclass(frozen=True, eq=False)
class ObservationRequest:
    """One admitted observation; ``reply`` / ``error`` answer it (once, while its episode runs)."""

    obs: Observation  # this row's observation, as predict() takes it
    rows: Rows  # exactly one row, bound to this observation
    _admitted_at: float | None  # None: the episode's clock is unknown
    seq: int | None  # the OBSERVATION's wire seq (None for a vector slot)
    _server: ModelServer = field(repr=False)

    @property
    def row(self) -> Row:
        return self.rows[0]

    @property
    def admitted_at(self) -> float:
        """``ep.clock.now()`` at admission (a sync observation ``k``: ``k * step_dt``)."""
        if self._admitted_at is None:
            _unknown_step_dt()
        return self._admitted_at

    async def reply(self, action: Action) -> None:
        await self._server.send_action(self.rows, columns.stack([action]))

    async def error(self, error: Exception) -> None:
        """Fail this observation alone; the episode goes on."""
        await self._server.send_error(self.rows, error)


class Mailbox:
    """One episode's admitted observations: ``"latest"`` keeps the newest, an int is a bounded FIFO.  One owner
    task receives; ``put`` never blocks."""

    def __init__(self, backlog: Backlog) -> None:
        if backlog != "latest" and not (isinstance(backlog, int) and not isinstance(backlog, bool) and backlog >= 1):
            raise ValueError(f'backlog must be "latest" or a positive int, got {backlog!r}')
        self.backlog = backlog
        self.dropped = 0
        self._items: deque[ObservationRequest] = deque()
        self._wake = anyio.Event()
        self._closed = False
        self._owner: int | None = None

    def put(self, request: ObservationRequest) -> None:
        if self._closed:
            return
        if self.backlog == "latest":
            if self._items:
                self._items.clear()
                self.dropped += 1
        elif len(self._items) >= self.backlog:
            raise MailboxOverflow(f"more than {self.backlog} observations waiting")
        self._items.append(request)
        self._wake.set()

    async def receive(self) -> ObservationRequest:
        self._own()
        while True:
            if self._closed:
                raise EpisodeClosed("the episode is closing")
            if self._items:
                return self._items.popleft()
            await self._wake.wait()
            self._wake = anyio.Event()

    def receive_nowait(self) -> ObservationRequest | None:
        self._own()
        return self._items.popleft() if self._items and not self._closed else None

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        self._wake.set()

    def _own(self) -> None:
        task = anyio.get_current_task().id
        if self._owner is None:
            self._owner = task
        elif self._owner != task:
            raise RuntimeError("an episode's observations have one receiving task")


class _UnknownClock:
    """A sync episode's clock when no ``step_dt`` is known: any use fails (the episode, from its tasks)."""

    def now(self) -> NoReturn:
        _unknown_step_dt()

    async def sleep(self, seconds: float, *, priority: int = 0) -> NoReturn:
        _unknown_step_dt()

    async def wait_until(self, deadline: float, *, priority: int = 0) -> NoReturn:
        _unknown_step_dt()

    def every(self, period: float, *, first: Literal["now", "after"] = "after", priority: int = 0) -> NoReturn:
        _unknown_step_dt()


def _unknown_step_dt() -> NoReturn:
    raise RuntimeError(
        "step_dt unknown: the benchmark declares no step_dt and the server sets none, so a sync episode has no clock"
    )


class EpisodeContext:
    """One episode's scope, given to ``run(ep)`` and its tasks; invalid after ``on_episode_end``.  Event-loop only.
    ``start_soon`` makes it a task group for ``SupervisorHost.start(ep)``."""

    def __init__(
        self,
        server: ModelServer,
        rows: Rows,
        clock: Clock,
        step_dt: float | None,
        mailbox: Mailbox,
        link: _Link | None,
    ) -> None:
        row = rows[0]
        self.episode = row.episode
        self.row_id = row.id
        self.mode: Mode = row.episode.mode
        self.clock = clock
        self.step_dt = step_dt
        self.logger = logging.getLogger(f"vla_eval.episode.{row.episode.episode_id[:8]}")
        self.stop_event = anyio.Event()
        self.can_publish = self.mode == "live" and link is not None and link.publish is not None
        self._server = server
        self._rows = rows  # the start Rows: this incarnation
        self._mailbox = mailbox
        self._link = link
        self._failed: Exception | None = None
        self._control: BaseException | None = None  # KeyboardInterrupt / SystemExit, raised after the drain
        self._tasks = 0
        self._owned = 0
        self._calls_done = self._calls_settled = 0  # finished ep.call s; those a settle has seen
        self._call_done = anyio.Event()  # set (and replaced) as each ep.call finishes
        self._delivering = anyio.Lock()  # a sync reply can arrive mid-advance: the next observation waits its turn
        self._drained = anyio.Event()
        self._drained.set()
        self._exited = anyio.Event()
        self._tg: TaskGroup | None = None
        self._scope = anyio.CancelScope()

    @property
    def closing(self) -> bool:
        return self.stop_event.is_set()

    async def observations(self) -> AsyncIterator[ObservationRequest]:
        """The admitted observations, in order; ends when the episode closes."""
        while True:
            try:
                request = await self._mailbox.receive()
            except EpisodeClosed:
                return
            yield request

    def receive_nowait(self) -> ObservationRequest | None:
        return self._mailbox.receive_nowait()

    def start_soon(self, fn: Callable[..., Awaitable[Any]], *args: Any) -> None:
        """Run ``fn(*args)`` as a task of this episode (its exceptions fail the episode)."""
        if self.closing:
            raise EpisodeClosed("the episode is closing")
        if self._tg is None:
            raise RuntimeError("the episode's scope has not started")
        self._tg.start_soon(self._task, fn, args)
        self._tasks += 1  # before the task can run: start_soon never switches

    async def call(self, fn: Callable[..., T], *args: Any) -> T:
        """``fn(*args)`` on the model thread; the episode's release waits for it.  ``EpisodeClosed`` if the episode
        is closing, or closed before the call completed (its result is dropped)."""
        await checkpoint()  # a cancelled caller submits nothing
        if self.closing:
            raise EpisodeClosed("the episode is closing")
        if self._owned == 0:
            self._drained = anyio.Event()
        self._owned += 1
        try:
            result = await self._server._run_submitted(fn, *args)
        except Exception as exc:
            if self.closing:
                raise EpisodeClosed("the episode closed during the call") from exc
            raise
        finally:
            self._owned -= 1
            self._calls_done += 1
            self._call_done.set()
            self._call_done = anyio.Event()
            if self._owned == 0:
                self._drained.set()
        if self.closing:
            raise EpisodeClosed("the episode closed during the call")
        return result

    async def publish(self, action: Action, *, source: ObservationRequest | None = None) -> None:
        """Send ``action`` to a live client now (``PUBLISH``, latest-wins there).  Dropped once closing."""
        link = self._link
        if not self.can_publish or link is None or link.publish is None:
            raise RuntimeError("publish needs a live episode whose client advertised 'publish'")
        if source is not None and not self._owns(source):
            raise ValueError("the source observation belongs to another episode")
        async with link.lock:
            if self.closing:  # an admitted send may complete; none starts after closing
                self.logger.debug("publish dropped: the episode is closing")
                return
            index, link.index = link.index, link.index + 1
            await link.publish(action, index, None if source is None else source.seq)

    def fail(self, error: Exception) -> None:
        """End the episode as failed by the model (terminal; first failure wins)."""
        if self._failed is None:
            self._failed = error
            if not self.closing:
                self.logger.error("episode failed: %s: %s", type(error).__name__, error, exc_info=error)
                try:
                    self._server.fail_episode(self._rows, error)
                finally:
                    self._close()
                return
        self._close()

    async def _settle(self) -> bool:
        """chrono's settle hook: True (settle again) once an ``ep.call`` of this episode finished since the last
        settle, waiting for one if any is in flight; so each True is a finished call, and ``max_settles`` bounds them."""
        if self.closing:
            raise EpisodeClosed("the episode is closing")
        if self._owned and self._calls_done == self._calls_settled:
            await self._call_done.wait()
        seen, self._calls_settled = self._calls_settled, self._calls_done
        return seen != self._calls_done

    def _owns(self, request: ObservationRequest) -> bool:
        return (
            int(request.rows.ids[0]) == self.row_id
            and int(request.rows._gen[0]) == int(self._rows._gen[0])
            and request.row.episode is self.episode
        )

    def _close(self) -> None:
        self.stop_event.set()
        self._mailbox.close()
        self._scope.cancel()

    async def _task(self, fn: Callable[..., Awaitable[Any]], args: tuple[Any, ...]) -> None:
        try:
            if not self.closing:
                await _guarded(self, fn, *args)
        finally:
            self._tasks -= 1
            if self._tasks == 0 and not self.closing:
                self.fail(RuntimeError("no task is left to answer observations"))


def _classify(ep: EpisodeContext, exc: BaseException) -> None:
    """The one exception policy: ordinary errors fail the episode, closing ``EpisodeClosed`` and cancellation are
    teardown, process control is captured (raised grouped after the drain)."""
    cancelled = anyio.get_cancelled_exc_class()
    for leaf in _leaves(exc):
        if isinstance(leaf, EpisodeClosed) and ep.closing:
            continue
        if isinstance(leaf, Exception):
            ep.fail(leaf)
        elif not isinstance(leaf, cancelled):
            ep._control = ep._control or leaf
            ep._close()


async def _guarded(ep: EpisodeContext, fn: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> None:
    """The episode task boundary: ``fn``'s exceptions are classified for ``ep``; cancellation propagates."""
    try:
        await fn(*args, **kwargs)
    except BaseException as exc:
        _classify(ep, exc)
        if isinstance(exc, anyio.get_cancelled_exc_class()):
            raise


def episode_daemon(
    ep: EpisodeContext, body: Callable[..., Awaitable[None]]
) -> Callable[..., Coroutine[Any, Any, None]]:
    """A chrono daemon body whose errors fail ``ep`` instead of escaping: ``sup.spawn(episode_daemon(ep, body))``."""

    async def run(*args: Any, **kwargs: Any) -> None:
        await _guarded(ep, body, *args, **kwargs)

    return run


class EpisodeDaemons:
    """One scope per started row running ``run(ep)``; a ``ModelServer`` forwards its callbacks, ``on_episode_failed``
    too.  ``backlog``: ``"latest"`` or a FIFO bound; ``step_dt``: default seconds per env step; ``settle_rounds``: a
    sync clock's checkpoints per wake (``SETTLE_ROUNDS``); unmet ``requires`` fail starts."""

    def __init__(
        self,
        server: ModelServer,
        run: Run,
        *,
        backlog: Backlog = "latest",
        step_dt: float | None = None,
        settle_rounds: int = SETTLE_ROUNDS,
        reset: Reset | None = None,
        finish: Finish | None = None,
        requires: frozenset[str] = frozenset(),
    ) -> None:
        Mailbox(backlog)  # validates
        if step_dt is not None:
            _check_step_dt(step_dt)
        SimClock(settle_rounds=settle_rounds)  # validates
        if unknown := set(requires) - set(CAPABILITIES):
            raise ValueError(f"unknown capabilities {sorted(unknown)}; known: {list(CAPABILITIES)}")
        self.server, self.run, self.backlog = server, run, backlog
        self.step_dt, self.settle_rounds, self.reset, self.finish = step_dt, settle_rounds, reset, finish
        self.requires = frozenset(requires)
        self._eps: dict[int, EpisodeContext] = {}  # by row id
        self._tg: TaskGroup | None = None
        self._stopping = True  # until on_serve_start

    async def on_serve_start(self, tg: TaskGroup) -> None:
        self._tg, self._stopping = tg, False
        tg.start_soon(self._watch)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        failed: dict[int, Exception] = {}
        plans: dict[int, tuple[Clock, float | None]] = {}
        for i, row in enumerate(rows):  # validated before reset
            try:
                plans[i] = self._admit(row)
            except Exception as exc:
                failed[i] = exc
        accepted = [i for i in range(len(rows)) if i not in failed]
        if not accepted:
            return failed
        if self.reset is not None:
            try:
                out = await self.server.run_on_model_thread(self.reset, rows.take(accepted)) or {}
            except Exception as exc:
                out = dict.fromkeys(range(len(accepted)), exc)
            failed.update({accepted[j]: exc for j, exc in out.items()})
        ready = [i for i in accepted if i not in failed]
        if self._stopping or self._tg is None:  # shut down during reset: finish the reset rows once, fail them
            if self.finish is not None and ready:
                aborted: list[EpisodeResult] = [{"status": "aborted", "failure_reason": None}] * len(ready)
                with anyio.CancelScope(shield=True):
                    try:
                        await self.server.run_on_model_thread(self.finish, rows.take(ready), aborted)
                    except Exception:
                        logger.exception("finish failed while rolling back a start")
            failed.update(dict.fromkeys(ready, RuntimeError("model server stopped")))
            return failed
        for i in ready:  # no await: a context is registered once its host is
            one = rows.take([i])
            ep = EpisodeContext(self.server, one, *plans[i], Mailbox(self.backlog), self.server._links.get(one[0].id))
            self._tg.start_soon(self._host, ep)
            self._eps[ep.row_id] = ep
        return failed

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        seq = rows._pending.seq if rows._pending is not None else None
        n = len(rows)
        for i, one_obs in enumerate(columns.split(obs, n)):
            one = rows if n == 1 else rows.take([i])
            ep = self._eps.get(int(one.ids[0]))
            if ep is None or ep.closing or ep.episode is not one._episodes[0]:
                await self.server.send_error(one, EpisodeClosed("the episode is not running"))
                continue
            if isinstance(ep.clock, SimClock) and self._tg is not None:  # each episode's clock on its own
                self._tg.start_soon(self._deliver, ep, one_obs, one, seq)
            else:
                await self._deliver(ep, one_obs, one, seq)

    async def _deliver(self, ep: EpisodeContext, obs: Observation, rows: Rows, seq: int | None) -> None:
        """Admit observation ``k``; a sync clock first moves to ``k * step_dt``, where the observation goes before
        that instant's other wakes (priority -1).  A reply can precede them, so an episode's deliveries queue."""
        clock = ep.clock
        async with ep._delivering:
            try:
                if not isinstance(clock, SimClock):
                    if ep.closing:
                        await self.server.send_error(rows, EpisodeClosed("the episode is not running"))
                        return
                    now = None if isinstance(clock, _UnknownClock) else clock.now()
                    ep._mailbox.put(ObservationRequest(obs, rows, now, seq, self.server))
                    return
                assert ep.step_dt is not None
                admitted = False

                def admit() -> None:
                    nonlocal admitted
                    if not ep.closing:
                        ep._mailbox.put(ObservationRequest(obs, rows, clock.now(), seq, self.server))
                        admitted = True

                cancel = clock.call_at(rows[0].step * ep.step_dt, admit, priority=-1)
                try:  # due wakes one at a time, each settling its model calls
                    await clock.advance_to(rows[0].step * ep.step_dt, settle=ep._settle)
                except EpisodeClosed:
                    if not ep.closing:
                        raise
                finally:
                    cancel()
                if not admitted:
                    await self.server.send_error(rows, EpisodeClosed("the episode is not running"))
            except Exception as exc:
                ep.fail(exc)

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        eps = [ep for rid in rows.ids.tolist() if (ep := self._eps.pop(rid, None)) is not None]
        for ep in eps:  # the last instant's wakes run (deterministic) before the episode closes
            async with ep._delivering:
                pass
        for ep in eps:
            ep._close()
        for ep in eps:  # tasks unwound and model calls drained before finish
            await ep._exited.wait()
        if self.finish is not None:
            await self.server.run_on_model_thread(self.finish, rows, results)

    def on_episode_failed(self, row_id: int, episode: Episode, error: Exception) -> None:
        ep = self._eps.get(row_id)
        if ep is not None and ep.episode is episode:
            ep._failed = ep._failed or error
            ep._close()

    # ------------------------------------------------------------------

    def _admit(self, row: Row) -> tuple[Clock, float | None]:
        episode = row.episode
        missing = self.requires - episode.capabilities
        if missing:
            raise MissingCapabilities(f"missing capabilities: {', '.join(sorted(missing))}")
        if "publish" in self.requires and episode.mode != "live":
            raise RuntimeError("this server publishes actions: it needs a live episode")
        dt = episode.config.get("step_dt")
        dt = self.step_dt if dt is None else _check_step_dt(dt)
        if episode.mode == "live":
            return WallClock(), dt
        return (_UnknownClock() if dt is None else SimClock(0.0, settle_rounds=self.settle_rounds)), dt

    async def _host(self, ep: EpisodeContext) -> None:
        with anyio.CancelScope(shield=True):  # cancellation reaches the episode only through ep._close()
            try:
                with ep._scope:
                    async with anyio.create_task_group() as tg:
                        ep._tg = tg
                        if not ep.closing:  # closed before its host ran: no root
                            ep.start_soon(self.run, ep)
            finally:
                ep._close()
                await ep._drained.wait()
                ep._exited.set()
        if ep._control is not None:
            raise BaseExceptionGroup("episode control", [ep._control])

    async def _watch(self) -> None:
        try:
            await anyio.sleep_forever()
        finally:  # server shutdown: admission closes, then every episode closes
            self._stopping = True
            for ep in list(self._eps.values()):
                ep._close()


def _check_step_dt(dt: Any) -> float:
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"step_dt must be a positive finite number of seconds, got {dt!r}")
    return float(dt)


class DaemonModelServer(ModelServer):
    """A ``ModelServer`` whose episodes each run ``run(ep)`` (override it, or pass ``run=``) on ``EpisodeDaemons``.
    Continuous inference (``backlog="latest"``: each inference takes the newest observation)::

        async def run(self, ep):
            async for request in ep.observations():
                await request.reply(await ep.call(self.predict, request.obs, request.row))

    Streaming a chunk on a time grid (RTC): ``examples/live_servers/rtc.py``.
    """

    supports_vector_step = False
    requires: frozenset[str] = frozenset()

    def __init__(
        self,
        run: Run | None = None,
        *,
        backlog: Backlog = "latest",
        step_dt: float | None = None,
        settle_rounds: int = SETTLE_ROUNDS,
        max_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.max_rows = max_rows
        self._run_fn = run
        self.daemons = EpisodeDaemons(
            self,
            self.run,
            backlog=backlog,
            step_dt=step_dt,
            settle_rounds=settle_rounds,
            reset=self.reset_episodes,
            finish=self.finish_episodes,
            requires=self.requires,
        )

    async def run(self, ep: EpisodeContext) -> None:
        """One episode's loop: answer ``ep.observations()`` (``request.reply``), now or later."""
        if self._run_fn is None:
            raise NotImplementedError(f"{type(self).__name__} must override run() or pass run=")
        await self._run_fn(ep)

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        """Reset the state of ``rows`` (``episode.config``: the ``EPISODE_START`` payload; warm up here or in
        ``__init__``).  Blocking.  Return ``{position: exception}`` for failed starts, or raise to fail them all."""
        return None

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        """Episodes in ``rows`` ended (``results[i]``: the ``EPISODE_END`` payload).  Blocking."""

    async def on_serve_start(self, tg: TaskGroup) -> None:
        await super().on_serve_start(tg)
        await self.daemons.on_serve_start(tg)

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        return await self.daemons.on_episode_start(rows)

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        await self.daemons.on_observation(obs, rows)

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        await self.daemons.on_episode_end(rows, results)

    def on_episode_failed(self, row_id: int, episode: Episode, error: Exception) -> None:
        self.daemons.on_episode_failed(row_id, episode, error)
