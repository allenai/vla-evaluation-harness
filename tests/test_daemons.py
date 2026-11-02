"""Episode daemons: one scope per episode (mailbox, tasks, model calls, clock), the one exception policy and the
teardown order; chrono's SupervisorHost inside an episode."""

from __future__ import annotations

import threading
import time
from typing import Any, cast

import anyio
import numpy as np
import pytest
from chrono_daemon import SimClock, Supervisor, WallClock
from chrono_daemon.recipes.supervisor_host import SupervisorHost

from vla_eval.connection import Connection
from vla_eval.model_servers.base import EpisodeClosed, ModelServer, _leaves, _Pending
from vla_eval.model_servers.daemons import (
    DaemonModelServer,
    EpisodeContext,
    EpisodeDaemons,
    Mailbox,
    MailboxOverflow,
    ObservationRequest,
    episode_daemon,
)
from vla_eval.protocol.messages import MessageType
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import LIVE, Daemon, StubBenchmark, echo, observe, running, serving, start


def _req(v: float = 0.0) -> ObservationRequest:
    return ObservationRequest({"value": v}, cast(Any, None), 0.0, None, cast(Any, None))


def _value(pending: _Pending) -> float:
    return float(np.asarray(pending.result()["actions"]).reshape(-1)[0])


# ---------------------------------------------------------------------------
# Mailbox
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_latest_keeps_the_newest_and_a_fifo_keeps_every_admission_until_its_bound():
    latest = Mailbox("latest")
    for v in (1.0, 2.0, 3.0):
        latest.put(_req(v))
    assert (await latest.receive()).obs["value"] == 3.0 and latest.dropped == 2
    assert latest.receive_nowait() is None
    fifo = Mailbox(2)
    fifo.put(_req(1.0))
    fifo.put(_req(2.0))
    with pytest.raises(MailboxOverflow):
        fifo.put(_req(3.0))
    assert [(await fifo.receive()).obs["value"] for _ in range(2)] == [1.0, 2.0]
    fifo.put(_req(4.0))
    fifo.close()
    with pytest.raises(EpisodeClosed):  # waiting items are dropped
        await fifo.receive()


@pytest.mark.anyio
async def test_a_mailbox_has_one_receiving_task():
    box = Mailbox("latest")
    assert box.receive_nowait() is None
    errors: list[Exception] = []

    async def other() -> None:
        try:
            await box.receive()
        except RuntimeError as exc:
            errors.append(exc)

    async with anyio.create_task_group() as tg:
        tg.start_soon(other)
    assert errors and "one receiving task" in str(errors[0])


@pytest.mark.parametrize("backlog", [0, True, "newest"])
def test_a_bad_backlog_is_refused(backlog):
    with pytest.raises(ValueError):
        Mailbox(backlog)


# ---------------------------------------------------------------------------
# Lifecycle and teardown order
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_several_admissions_before_the_owner_wakes_reach_it_in_order():
    seen: list[float] = []

    async def run(ep: EpisodeContext) -> None:
        await ep.call(time.sleep, 0.2)  # observations pile up meanwhile
        async for req in ep.observations():
            seen.append(float(req.obs["value"]))
            await req.reply({"actions": np.zeros(2, np.float32)})

    server = Daemon(run, backlog=8)
    async with serving(server):
        (rid,) = await start(server, mode="live")
        for v in (1.0, 2.0, 3.0):
            await observe(server, rid, v, wait=False)
        with anyio.fail_after(3):
            while len(seen) < 3:
                await anyio.sleep(0.01)
    assert seen == [1.0, 2.0, 3.0]


@pytest.mark.anyio
async def test_end_during_a_model_call_finishes_after_it_on_the_model_thread_and_frees_the_row():
    order: list[str] = []
    server: Daemon

    def slow() -> str:
        server.threads.add(threading.current_thread().name)
        time.sleep(0.2)
        order.append("call done")
        return "late"

    async def run(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            try:
                await ep.call(slow)
            except EpisodeClosed:
                order.append("discarded")
                raise
            await req.reply({"actions": np.zeros(2, np.float32)})

    server = Daemon(run)
    async with serving(server):
        (rid,) = await start(server)
        pending = await observe(server, rid, 1.0, wait=False)
        await anyio.sleep(0.05)
        await server._end_episodes([rid], [{"status": "aborted"}])
        order.append("ended")
        assert order == ["call done", "discarded", "ended"] and server.log[-1] == ("finish", [rid])
        assert not pending.done.is_set() and server._running() == 0
        assert await start(server) == [rid]  # reusable
    assert server.threads == {"model_0"}  # reset, the call and finish


@pytest.mark.anyio
async def test_closure_before_the_host_runs_spawns_no_root():
    ran: list[int] = []

    async def run(ep: EpisodeContext) -> None:
        ran.append(ep.row_id)

    server = Daemon(run)
    async with serving(server):
        (rid,) = await start(server)
        await server._end_episodes([rid], [{}])  # no checkpoint between: the host never entered
        await anyio.sleep(0.05)
    assert ran == [] and server.log == [("reset", [rid]), ("finish", [rid])]


@pytest.mark.anyio
async def test_a_child_tasks_exception_fails_only_its_episode():
    async def boom() -> None:
        raise ValueError("child exploded")

    async def run(ep: EpisodeContext) -> None:
        if ep.episode.config.get("boom"):
            ep.start_soon(boom)
        await echo(ep)

    server = Daemon(run)
    async with serving(server):
        bad, good = (await start(server, config={"boom": True}))[0], (await start(server))[0]
        await anyio.sleep(0.05)
        assert isinstance(server._failed.get(bad), ValueError) and good not in server._failed
        assert _value(await observe(server, good, 2.0)) == 2.0
        assert "child exploded" in str((await observe(server, bad, 1.0)).errors[0])


@pytest.mark.anyio
async def test_server_shutdown_is_teardown_not_failure():
    calls: list[str] = []

    def slow() -> None:
        time.sleep(0.2)
        calls.append("done")

    async def run(ep: EpisodeContext) -> None:
        async for _ in ep.observations():
            await ep.call(slow)

    server = Daemon(run)
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 1.0, wait=False)
        await anyio.sleep(0.05)
    assert calls == ["done"] and rid not in server._failed  # the call in flight finished; no EPISODE_ERROR
    await server._end_episodes([rid], [{}])  # the connection's cleanup still finishes it
    assert server.log[-1] == ("finish", [rid])


@pytest.mark.anyio
async def test_keyboard_interrupt_drains_every_episode_then_leaves_grouped():
    done: list[str] = []

    def slow() -> None:
        time.sleep(0.3)
        done.append("sibling call")

    async def run(ep: EpisodeContext) -> None:
        if ep.episode.config.get("interrupt"):
            await anyio.sleep(0.05)
            raise KeyboardInterrupt
        await ep.call(slow)  # the sibling's real outstanding future

    server = Daemon(run)
    with pytest.raises(BaseExceptionGroup) as info:
        async with serving(server):
            await start(server)
            await start(server, config={"interrupt": True})
            await anyio.sleep(2)
    assert any(isinstance(e, KeyboardInterrupt) for e in _leaves(info.value))
    assert done == ["sibling call"]


@pytest.mark.anyio
async def test_mixed_leaves_record_the_error_and_raise_control_after_the_drain():
    async def run(ep: EpisodeContext) -> None:
        cancelled = anyio.get_cancelled_exc_class()
        nested = ExceptionGroup("inner", [ValueError("ordinary")])
        raise BaseExceptionGroup("mixed", [nested, cancelled(), SystemExit(3)])

    server = Daemon(run)
    with pytest.raises(BaseExceptionGroup) as info:
        async with serving(server):
            (rid,) = await start(server)
            await anyio.sleep(1)
    assert isinstance(server._failed[rid], ValueError)
    assert [type(e) for e in _leaves(info.value)] == [SystemExit]


@pytest.mark.anyio
async def test_closed_episode_closed_is_teardown_but_an_ordinary_error_beside_it_is_kept():
    async def run(ep: EpisodeContext) -> None:
        try:
            await ep.call(time.sleep, 0.2)
        except EpisodeClosed as exc:
            raise ExceptionGroup("teardown", [exc, KeyError("cleanup bug")]) from None

    server = Daemon(run)
    async with serving(server):
        (rid,) = await start(server)
        await anyio.sleep(0.05)
        await server._end_episodes([rid], [{}])
    ep_failed = server.daemons._eps.get(rid)
    assert ep_failed is None and server._running() == 0  # ended cleanly; the KeyError was recorded, not raised


@pytest.mark.anyio
async def test_call_refuses_after_closing_and_drops_a_late_result():
    submitted: list[str] = []
    results: list[Any] = []

    async def run(ep: EpisodeContext) -> None:
        with anyio.CancelScope() as scope:
            scope.cancel()
            await ep.call(submitted.append, "cancelled before submission")
        results.append(await ep.call(lambda: "first"))
        results.append(await ep.call(lambda: "second"))  # owned 0 -> 1 again
        ep.fail(ValueError("stop"))
        with anyio.CancelScope(shield=True):  # past the cancellation closing delivers
            with pytest.raises(EpisodeClosed):
                await ep.call(submitted.append, "after closing")
            with pytest.raises(EpisodeClosed):
                ep.start_soon(anyio.sleep, 0)
            assert ep._tasks == 1
            results.append("checked")

    server = Daemon(run)
    async with serving(server):
        await start(server)
        await anyio.sleep(0.1)
    assert submitted == [] and results == ["first", "second", "checked"]


# ---------------------------------------------------------------------------
# Startup: validate before reset, roll back after it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hz", ["fast", True, float("inf")])
@pytest.mark.anyio
async def test_a_start_that_fails_validation_never_resets(hz):
    server = Daemon(echo)
    async with serving(server):
        (out,) = await server._start_episodes([({"control_hz": hz}, "s")])
    assert isinstance(out, ValueError) and "control_hz" in str(out)
    assert server.log == [] and server._running() == 0 and not server.daemons._eps


@pytest.mark.anyio
async def test_a_missing_capability_never_resets():
    class Needs(Daemon):
        requires = frozenset({"publish"})

    server = Needs(echo)
    async with serving(server):
        (refused,) = await server._start_episodes([({}, "s")], mode="live")
        (sync,) = await server._start_episodes([({}, "s")], capabilities=frozenset({"publish"}))
    assert "missing capabilities: publish" in str(refused) and "live episode" in str(sync)
    assert server.log == [] and server._running() == 0


@pytest.mark.anyio
async def test_reset_runs_for_accepted_positions_only_and_failures_keep_their_positions():
    server = Daemon(echo)
    server.reset_fail = {1}
    async with serving(server):
        out = await server._start_episodes([({}, "s"), ({"control_hz": 0}, "s"), ({}, "s"), ({}, "s")])
    assert isinstance(out[1], ValueError) and "control_hz" in str(out[1])
    assert isinstance(out[2], ValueError) and "cannot reset" in str(out[2])  # reset's position 1 is batch 2
    assert server.log == [("reset", [0, 2, 3])] and isinstance(out[0], int) and isinstance(out[3], int)

    def broken(rows: Any) -> None:
        raise OSError("no device")

    server.daemons.reset = broken  # raising fails every start of the call
    async with serving(server):
        assert [type(o) for o in await server._start_episodes([({}, "s"), ({}, "s")])] == [OSError, OSError]


@pytest.mark.anyio
async def test_a_reset_resuming_after_shutdown_starts_nothing_and_finishes_its_rows_once():
    gate = threading.Event()

    class Slow(Daemon):
        def reset_episodes(self, rows):
            gate.wait(2)
            return super().reset_episodes(rows)

    server = Slow(echo)
    out: list[Any] = []

    async def begin() -> None:
        out.extend(await server._start_episodes([({}, "s")]))

    async with serving(server) as tg:
        tg.start_soon(begin)
        await anyio.sleep(0.05)
        tg.cancel_scope.cancel()  # server shutdown while reset runs on the model thread
        with anyio.CancelScope(shield=True):
            await anyio.sleep(0.05)  # the shutdown sweep first
        gate.set()
    assert isinstance(out[0], RuntimeError) and "stopped" in str(out[0])
    assert server.log == [("reset", [0]), ("finish", [0])] and server._running() == 0


# ---------------------------------------------------------------------------
# Composition without DaemonModelServer, vector slots and transport-stamped mode
# ---------------------------------------------------------------------------


class Composed(ModelServer):
    """EpisodeDaemons inside a direct ModelServer, with plain functions and no reset hook; vector capable."""

    def __init__(self, run: Any, **kw: Any) -> None:
        super().__init__()
        self.daemons = EpisodeDaemons(self, run, **kw)

    async def on_serve_start(self, tg):
        await super().on_serve_start(tg)
        await self.daemons.on_serve_start(tg)

    async def on_episode_start(self, rows):
        return await self.daemons.on_episode_start(rows)

    async def on_observation(self, obs, rows):
        await self.daemons.on_observation(obs, rows)

    async def on_episode_end(self, rows, results):
        await self.daemons.on_episode_end(rows, results)

    def on_episode_failed(self, row_id, episode, error):
        self.daemons.on_episode_failed(row_id, episode, error)


@pytest.mark.anyio
async def test_a_vector_slot_saying_live_is_still_sync_on_a_sim_clock(free_port):
    seen: list[tuple[str, type, float | None]] = []

    async def run(ep: EpisodeContext) -> None:
        seen.append((ep.mode, type(ep.clock), ep.sim_dt))
        await echo(ep)

    async with running(Composed(run), free_port) as url, Connection(url) as conn:
        starts = [[0, {"mode": "live", "control_hz": 20}], [1, {"control_hz": 5.0}]]
        reply = await conn.exchange(
            {"ends": [], "starts": starts, "slots": [0, 1], "observations": {"value": np.asarray([1.0, 2.0])}}
        )
    assert reply["errors"] == [] and np.asarray(reply["actions"]["actions"])[:, 0].tolist() == [1.0, 2.0]
    assert seen == [("sync", SimClock, 0.05), ("sync", SimClock, 0.2)]  # per-slot rates


@pytest.mark.anyio
async def test_each_episode_gets_its_own_clock_by_transport_mode():
    clocks: dict[int, Any] = {}

    async def run(ep: EpisodeContext) -> None:
        clocks[ep.row_id] = ep.clock
        if isinstance(ep.clock, SimClock) and ep.row_id == 0:
            await ep.clock.advance(1.5)
        await echo(ep)

    server = Composed(run, control_hz=4.0)
    async with serving(server):
        await start(server, 2)
        await start(server, mode="live")
        await anyio.sleep(0.05)
    assert [type(clocks[i]) for i in range(3)] == [SimClock, SimClock, WallClock]
    assert clocks[0].now() == 1.5 and clocks[1].now() == 0.0  # independent, fresh at 0


# ---------------------------------------------------------------------------
# Sim time: the loop advances; rates from EPISODE_START or the server
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_advancing_by_sim_dt_fires_a_daemon_on_its_boundaries_and_stamps_admissions():
    fires: list[float] = []
    admitted: list[float] = []

    async def ticker(ep: EpisodeContext) -> None:
        async for t in ep.clock.every(0.5, first="now"):
            fires.append(t)

    async def run(ep: EpisodeContext) -> None:
        ep.start_soon(ticker, ep)
        async for req in ep.observations():
            admitted.append(req.admitted_at)
            assert ep.sim_dt is not None and isinstance(ep.clock, SimClock)
            await ep.clock.advance(ep.sim_dt)
            await req.reply({"actions": np.zeros(2, np.float32)})

    server = Daemon(run, control_hz=4.0)  # sim_dt 0.25
    async with serving(server):
        (rid,) = await start(server)
        for v in range(5):
            await observe(server, rid, v)
    assert admitted == [0.0, 0.25, 0.5, 0.75, 1.0] and fires == [0.0, 0.5, 1.0]


@pytest.mark.anyio
async def test_control_hz_reaches_sim_dt_from_each_runner_or_the_server_config(free_port):
    rates: list[float | None] = []

    async def run(ep: EpisodeContext) -> None:
        rates.append(ep.sim_dt)
        await echo(ep)

    class RatedBenchmark(StubBenchmark):
        control_hz = 20.0

    async with running(Daemon(run, control_hz=2.0), free_port) as url, Connection(url) as conn:
        await SyncEpisodeRunner().run_episode(RatedBenchmark(), {"name": "t"}, conn, max_steps=2)
        await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
        await LiveEpisodeRunner(hz=50.0).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
        await conn.start_episode({"control_hz": None})  # null: the server's rate
        await conn.act({"value": 1.0})
    assert rates == [0.05, 0.5, 0.02, 0.5]


# ---------------------------------------------------------------------------
# chrono's SupervisorHost inside an episode, through the episode_daemon adapter
# ---------------------------------------------------------------------------


class Hosted(Daemon):
    """Runs a chrono supervisor per episode: a daemon replying from the mailbox, optionally failing."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.events: list[str] = []

    async def body(self, ctx: Any, ep: EpisodeContext) -> None:
        async for req in ep.observations():
            if float(req.obs["value"]) < 0:
                raise ValueError("daemon exploded")
            out = await ep.call(self._slow, float(req.obs["value"]))
            await req.reply({"actions": np.full(2, out, np.float32)})

    def _slow(self, v: float) -> float:
        time.sleep(0.1 if v == 7 else 0)
        self.events.append(f"call {v}")
        return v

    async def run(self, ep: EpisodeContext) -> None:
        sup = Supervisor(clock=ep.clock)
        sup.spawn(episode_daemon(ep, self.body), ep, name="replier")
        host = SupervisorHost(sup, on_stopping=lambda: self.events.append("stopping"), default_grace=0.0)
        await host.start(cast(Any, ep))  # the host only calls start_soon
        try:
            await ep.stop_event.wait()
        finally:
            with anyio.CancelScope(shield=True):
                await host.stop()
                self.events.append("host stopped")

    def finish_episodes(self, rows, results):
        self.events.append("finish")
        super().finish_episodes(rows, results)


@pytest.mark.anyio
async def test_a_hosted_daemons_error_fails_its_episode_and_the_sibling_goes_on():
    server = Hosted()
    async with serving(server):
        bad, good = await start(server, 2)
        pending = await observe(server, bad, -1.0, wait=False)
        await anyio.sleep(0.1)
        assert isinstance(server._failed[bad], ValueError) and "daemon exploded" in str(pending.errors[0])
        assert _value(await observe(server, good, 3.0)) == 3.0


@pytest.mark.anyio
async def test_end_stops_the_host_after_the_daemons_call_and_before_finish():
    server = Hosted()
    async with serving(server):
        (rid,) = await start(server)
        await observe(server, rid, 7.0, wait=False)  # a 0.1 s call
        await anyio.sleep(0.03)
        await server._end_episodes([rid], [{}])
    assert server.events == ["stopping", "call 7.0", "host stopped", "finish"]  # the call drains before finish


# ---------------------------------------------------------------------------
# Over a real socket
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_plain_function_run_serves_sync_and_live(free_port):
    async with running(DaemonModelServer(run=echo), free_port) as url, Connection(url) as conn:
        await conn.start_episode({"task": {}})
        assert (await conn.act({"value": 2.0}))["actions"].tolist() == [2.0, 2.0]
        await conn.start_episode(LIVE)
        await conn.send(MessageType.OBSERVATION, {"value": 3.0})
        msg = await conn.recv(timeout=2)
        assert msg.type == MessageType.ACTION and msg.payload["actions"].tolist() == [3.0, 3.0]
        await conn.end_episode({})


@pytest.mark.anyio
async def test_an_overflowing_fifo_fails_the_episode_and_a_queued_end_still_ends():
    gate = threading.Event()

    async def run(ep: EpisodeContext) -> None:
        await ep.call(gate.wait, 2)
        await echo(ep)

    server = Daemon(run, backlog=2)
    async with serving(server):
        (rid,) = await start(server, mode="live")
        for v in range(3):
            await observe(server, rid, v, wait=False)
        assert isinstance(server._failed[rid], MailboxOverflow)
        gate.set()
        await server._end_episodes([rid], [{}])
    assert server.log[-1] == ("finish", [rid]) and server._running() == 0
