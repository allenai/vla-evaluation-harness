"""Server-initiated messages: terminal failure (fail_episode / EPISODE_ERROR), PUBLISH and its listener rules,
capability negotiation and old-client refusal."""

from __future__ import annotations

import time
from typing import Any, cast

import anyio
import numpy as np
import pytest
import websockets
from websockets.asyncio.server import serve as ws_serve

from vla_eval import connection as connection_module
from vla_eval.connection import Connection, ServerEpisodeError
from vla_eval.model_servers.base import Episode, Rows, _Link, _Pending
from vla_eval.model_servers.daemons import EpisodeContext, ObservationRequest
from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.base import ModelStartError, failure_reason
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import (
    LIVE,
    BatchLogServer,
    Daemon,
    LogServer,
    StubBenchmark,
    echo,
    observe,
    running,
    serving,
    start,
)


@pytest.fixture
def legacy(monkeypatch):
    """Clients before this protocol addition: their HELLO advertises ``episode_ready`` only."""
    monkeypatch.setattr(connection_module, "CAPABILITIES", {"episode_ready": True})


async def _until(cond: Any, timeout: float = 3.0) -> None:
    with anyio.fail_after(timeout):
        while not cond():
            await anyio.sleep(0.01)


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_capabilities_round_trip_and_only_exactly_true_counts(free_port):
    seen: list[frozenset[str]] = []

    async def run(ep: EpisodeContext) -> None:
        seen.append(ep.episode.capabilities)
        await echo(ep)

    async with running(Daemon(run), free_port) as url:
        async with Connection(url) as conn:
            assert {k: conn.server_info["capabilities"][k] for k in ("publish", "episode_error")} == {
                "publish": True,
                "episode_error": True,
            }
            await conn.start_episode(LIVE)
            await conn.end_episode({})
        async with websockets.connect(url) as ws:
            hello = {"harness_version": "x", "capabilities": {"publish": "yes", "episode_error": 1}}
            await ws.send(pack_message(Message(MessageType.HELLO, hello, 1)))
            await ws.recv()
            await ws.send(pack_message(Message(MessageType.EPISODE_START, LIVE, 2)))
            await _until(lambda: len(seen) == 2)
    assert seen == [frozenset({"publish", "episode_error"}), frozenset()]


class Streams(Daemon):
    requires = frozenset({"publish", "episode_error"})


@pytest.mark.anyio
async def test_a_server_requiring_capabilities_fails_an_old_clients_start_and_closes_its_socket(free_port, legacy):
    async with running(Streams(echo), free_port) as url, Connection(url) as conn:
        with anyio.fail_after(3), pytest.raises(ModelStartError, match="missing capabilities") as info:
            await LiveEpisodeRunner(hz=50.0).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=200)
        assert failure_reason(info.value) == "model_start"
        with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
            await conn.recv(timeout=2)
        assert closed.value.rcvd is not None and closed.value.rcvd.code == 1008


@pytest.mark.anyio
async def test_an_observation_without_a_start_from_a_client_without_capabilities_is_refused(free_port):
    async with running(Streams(echo), free_port) as url, websockets.connect(url) as ws:
        await ws.send(pack_message(Message(MessageType.HELLO, {"capabilities": {}}, 1)))
        await ws.recv()
        await ws.send(pack_message(Message(MessageType.OBSERVATION, {"value": 1.0}, 2)))
        msg = unpack_message(cast(bytes, await ws.recv()))
    assert msg.type == MessageType.ERROR and "missing capabilities" in msg.payload["error"]


@pytest.mark.anyio
async def test_episode_ready_follows_reset_with_the_run_installed(free_port):
    class Slow(Daemon):
        def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
            time.sleep(0.3)
            return super().reset_episodes(rows)

    server = Slow(echo)
    async with running(server, free_port) as url, Connection(url) as conn:
        t0 = time.monotonic()
        await conn.start_episode(LIVE)
        assert time.monotonic() - t0 >= 0.3 and server.log == [("reset", [0])] and 0 in server.daemons._eps
        await conn.send_observation({"value": 2.0})
        assert (await conn.recv(timeout=2)).payload["actions"].tolist() == [2.0, 2.0]
        await conn.end_episode({})


# ---------------------------------------------------------------------------
# fail_episode on the base
# ---------------------------------------------------------------------------


class FailingAtStart(BatchLogServer):
    """Fails the episodes whose task says so as soon as they start, before any observation."""

    async def on_episode_start(self, rows: Rows) -> dict[int, Exception] | None:
        failed = await super().on_episode_start(rows)
        doomed = [i for i, e in enumerate(rows.episodes) if e.config.get("task", {}).get("doomed")]
        if doomed:
            self.fail_episode(rows.take(doomed), ValueError("model gave up"))
        return failed


@pytest.mark.anyio
async def test_a_sync_episode_failed_before_its_first_observation_errors_it(free_port):
    server = FailingAtStart()
    async with running(server, free_port) as url, Connection(url) as conn:
        with pytest.raises(Exception) as info:
            await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t", "doomed": True}, conn)
        assert failure_reason(info.value) == "model_act" and "model gave up" in str(info.value)
        await _until(lambda: server.results)
    assert server.predicted == [] and server.results == [{"status": "error", "failure_reason": "model_act"}]


@pytest.mark.anyio
async def test_a_vector_slot_failed_at_start_fails_alone(free_port):
    server = FailingAtStart()
    async with running(server, free_port) as url, Connection(url) as conn:
        starts = [[0, {"task": {"doomed": True}}], [1, {"task": {}}]]
        reply = await conn.exchange(
            {"ends": [], "starts": starts, "slots": [0, 1], "observations": {"value": np.asarray([1.0, 2.0])}}
        )
    assert [s for s, _ in reply["errors"]] == [0] and "model gave up" in reply["errors"][0][1]
    assert np.asarray(reply["actions"]["actions"])[:, 0].tolist() == [2.0] and server.batches == [[1]]


@pytest.mark.anyio
async def test_fail_episode_completes_pending_positions_once_and_ignores_stale_rows():
    hooks: list[tuple[int, Episode]] = []
    sent: list[str] = []

    class Hooked(LogServer):
        def on_episode_failed(self, row_id: int, episode: Episode, error: Exception) -> None:
            hooks.append((row_id, episode))
            raise RuntimeError("hook exploded")  # logged: EPISODE_ERROR still goes out

    async def sink(error: Exception) -> None:
        sent.append(str(error))
        raise RuntimeError("sink exploded")

    server = Hooked()
    async with serving(server):
        a, b = await start(server, 2)
        server._links[a] = _Link(sink, None)
        stale = server._rows([b])
        await server._end_episodes([b], [{}])
        (b2,) = await start(server)  # the same id, another incarnation
        pending = _Pending(3)
        pending.ids = np.asarray([a, a, b2])
        pending.gens = server._gen[pending.ids].copy()
        server._open.add(pending)
        episode_a = server._table[a]
        server.fail_episode(server._rows([a, a]), ValueError("first"))  # a row listed twice: once
        server.fail_episode(server._rows([a]), ValueError("second"))  # first failure wins
        server.fail_episode(stale, ValueError("stale"))  # an ended incarnation: ignored
        await anyio.sleep(0.05)
    assert hooks == [(a, episode_a)] and sent == ["first"]
    assert {i: str(e) for i, e in pending.errors.items()} == {0: "first", 1: "first"} and pending.left == 1
    assert server._failed == {a: server._failed[a]} and str(server._failed[a]) == "first"
    await server._end_episodes([a], [{}])
    assert a not in server._failed and a not in server._links


@pytest.mark.anyio
async def test_a_cancelled_observation_leaves_nothing_tracked():
    class Hang(LogServer):
        async def on_observation(self, obs, rows):
            await anyio.sleep_forever()

    server = Hang()
    (rid,) = await start(server)
    pending = _Pending(1)
    with anyio.move_on_after(0.05):
        with server._track(pending):
            await server._observe({"value": np.asarray([1.0])}, [rid], pending)
    assert not server._open


# ---------------------------------------------------------------------------
# EPISODE_ERROR through a real server, listener and runner
# ---------------------------------------------------------------------------


class Breaks(Daemon):
    """A live daemon that fails its episode after ``after`` observations (an S2 context overflow, say)."""

    def __init__(self, after: int = 2) -> None:
        super().__init__()
        self.after = after
        self.results: list[dict[str, Any]] = []

    async def run(self, ep: EpisodeContext) -> None:
        n = 0
        async for req in ep.observations():
            n += 1
            if n > self.after:
                raise OverflowError("context overflow")
            await req.reply({"actions": np.zeros(1, np.float32)})

    def finish_episodes(self, rows, results):
        self.results += [{k: r.get(k) for k in ("status", "failure_reason")} for r in results]


@pytest.mark.anyio
async def test_a_failed_live_episode_ends_model_act_through_the_runner(free_port):
    server = Breaks(after=2)
    async with running(server, free_port) as url, Connection(url) as conn:
        with anyio.fail_after(5), pytest.raises(ServerEpisodeError) as info:
            await LiveEpisodeRunner(hz=50.0).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=10_000)
        assert failure_reason(info.value) == "model_act" and "context overflow" in str(info.value)
        await _until(lambda: server.results)
        # the connection goes on: a sync episode next
        server.after = 100
        await SyncEpisodeRunner().run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=2)
    assert server.results[0] == {"status": "error", "failure_reason": "model_act"}


def _raw_server(on_message: Any) -> Any:
    async def handler(ws: Any) -> None:
        async for raw in ws:
            msg = unpack_message(raw)
            if msg.type == MessageType.HELLO:
                hello = Message(MessageType.HELLO, {"harness_version": "x"}, msg.seq)
                await ws.send(pack_message(hello))
            else:
                await on_message(ws, msg)

    return handler


@pytest.mark.anyio
async def test_the_listener_takes_publish_and_episode_error_for_the_running_episode_only(free_port):
    """A raw server: on each EPISODE_START it sends a wrong-seq PUBLISH with a high index first, then its script."""
    starts: list[int] = []

    async def on_message(ws: Any, msg: Message) -> None:
        if msg.type == MessageType.EPISODE_START:
            starts.append(msg.seq)
            s = msg.seq
            send = [
                (MessageType.PUBLISH, {"action": {"a": -1}, "index": 99}, s + 1000),  # no such episode
                (MessageType.EPISODE_ERROR, {"error": "x"}, s - 1),  # an earlier episode's
                (MessageType.PUBLISH, {"action": {"a": 0}, "index": 0}, s),
                (MessageType.PUBLISH, {"action": {"a": -1}, "index": 0}, s),  # stale index
                (MessageType.ACTION, {"a": 1}, s + 1),
                (MessageType.ERROR, {"error": "logged only"}, s + 1),
                (MessageType.PUBLISH, {"action": {"a": 2}, "index": 3}, s),
            ]
            for t, p, q in send:
                await ws.send(pack_message(Message(t, p, q)))
        elif msg.type == MessageType.OBSERVATION and msg.payload.get("fail"):
            await ws.send(pack_message(Message(MessageType.EPISODE_ERROR, {"error": "boom"}, starts[-1])))

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            got: list[Any] = []
            conn.on_action(lambda a: got.append(("action", a["a"])))
            conn.on_publish(lambda a: got.append(("publish", a["a"])))
            async with conn.listening():
                await conn.start_episode({})
                await _until(lambda: len(got) == 3)
                await conn.start_episode({})  # index 0 counts again
                await _until(lambda: len(got) == 6)
            assert got == [("publish", 0), ("action", 1), ("publish", 2)] * 2

            with pytest.raises(ServerEpisodeError, match="boom"):
                async with conn.listening():
                    await conn.send_observation({"fail": True})
                    await anyio.sleep(2)
            await conn.end_episode({})  # raw END: the episode's EPISODE_ERROR no longer counts
            async with conn.listening():
                await conn.send_observation({"fail": True})
                await anyio.sleep(0.2)


@pytest.mark.anyio
async def test_a_failed_start_send_and_close_clear_the_running_episode(free_port):
    async def on_message(ws: Any, msg: Message) -> None:
        pass

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        conn = Connection(f"ws://127.0.0.1:{free_port}")
        await conn.connect()
        await conn.start_episode({})
        assert conn._episode_active

        class Broken:
            state = conn._ws.state

            async def send(self, data: bytes) -> None:
                raise OSError("send failed")

        real, conn._ws = conn._ws, Broken()
        setattr(conn, "_ensure_connected", _noop)
        with pytest.raises(OSError):
            await conn.start_episode({})
        assert not conn._episode_active
        conn._ws = real
        await conn.start_episode({})
        await conn.close()
        assert not conn._episode_active


async def _noop() -> None:
    return None


@pytest.mark.anyio
async def test_latest_wins_between_replies_and_publishes_in_the_action_buffer(free_port):
    async def on_message(ws: Any, msg: Message) -> None:
        if msg.type == MessageType.OBSERVATION:
            await ws.send(pack_message(Message(MessageType.PUBLISH, {"action": {"v": "p"}, "index": 0}, msg.seq - 1)))
            await ws.send(pack_message(Message(MessageType.ACTION, {"v": "a"}, msg.seq)))
            await ws.send(pack_message(Message(MessageType.PUBLISH, {"action": {"v": "p2"}, "index": 1}, msg.seq - 1)))

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            buffer = ActionBuffer(hold_fn=lambda last: {"v": "hold"})
            conn.on_action(buffer.update)
            conn.on_publish(buffer.update)
            async with conn.listening():
                await conn.start_episode({})
                await conn.send_observation({})
                await _until(lambda: buffer.update_count == 3)
            assert buffer.get() == {"v": "p2"} and buffer.get() == {"v": "hold"}


# ---------------------------------------------------------------------------
# ep.publish
# ---------------------------------------------------------------------------


class Publisher(Daemon):
    """Publishes ``n`` actions per observation, the last from a separate task too (concurrent publishers)."""

    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.n = n

    async def run(self, ep: EpisodeContext) -> None:
        async for req in ep.observations():
            for k in range(self.n):
                ep.start_soon(ep.publish, {"actions": np.full(1, k, np.float32)})
            await ep.publish({"actions": np.full(1, 100.0, np.float32)}, source=req)


@pytest.mark.anyio
async def test_publish_sends_indexed_actions_bound_to_the_episode(free_port):
    async with running(Publisher(n=4), free_port) as url, websockets.connect(url) as ws:
        caps = {"capabilities": {"publish": True, "episode_error": True}}
        await ws.send(pack_message(Message(MessageType.HELLO, caps, 1)))
        await ws.recv()
        await ws.send(pack_message(Message(MessageType.EPISODE_START, LIVE, 2)))
        await ws.send(pack_message(Message(MessageType.OBSERVATION, {"value": 1.0}, 3)))
        msgs = [unpack_message(cast(bytes, await ws.recv())) for _ in range(5)]
    assert {m.type for m in msgs} == {MessageType.PUBLISH} and {m.seq for m in msgs} == {2}
    assert [m.payload["index"] for m in msgs] == [0, 1, 2, 3, 4]  # unique, increasing in send order
    sources = {float(m.payload["action"]["actions"][0]): m.payload["source_seq"] for m in msgs}
    assert sources == {0.0: None, 1.0: None, 2.0: None, 3.0: None, 100.0: 3}


@pytest.mark.anyio
async def test_publish_needs_a_live_capable_episode_and_its_own_source():
    errors: list[tuple[str, str]] = []
    other: list[ObservationRequest] = []

    async def run(ep: EpisodeContext) -> None:
        async for req in ep.observations():
            if ep.can_publish:
                try:
                    await ep.publish({"a": 1}, source=other[0])
                except ValueError as exc:
                    errors.append(("foreign", str(exc)))
            try:
                await ep.publish({"a": 1}, source=req)
            except RuntimeError as exc:
                errors.append((ep.mode, str(exc)))
            other.append(req)
            await req.reply({"actions": np.zeros(1, np.float32)})

    sent: list[Any] = []

    async def publish(action: Any, index: int, source_seq: int | None) -> None:
        sent.append(index)

    server = Daemon(run)
    async with serving(server):
        (sync,) = await start(server, capabilities=frozenset({"publish"}))
        (live_old,) = await start(server, mode="live")  # no publish capability
        (live,) = await start(server, mode="live", capabilities=frozenset({"publish"}), link=_Link(None, publish))
        for rid in (sync, live_old, live):
            await observe(server, rid, 1.0, wait=False)
            await anyio.sleep(0.05)
    assert [e[0] for e in errors] == ["sync", "live", "foreign"]
    assert sent == [0]


@pytest.mark.anyio
async def test_no_publish_starts_after_closing_even_while_one_is_blocked_in_send():
    entered, release = anyio.Event(), anyio.Event()
    sent: list[int] = []

    async def publish(action: Any, index: int, source_seq: int | None) -> None:
        if index == 0:
            entered.set()
            await release.wait()
        sent.append(index)

    async def admitted() -> None:
        with anyio.CancelScope(shield=True):  # this send completes despite closing
            await ep_ref[0].publish({"a": 0})

    ep_ref: list[EpisodeContext] = []

    async def run(ep: EpisodeContext) -> None:
        ep_ref.append(ep)
        ep.start_soon(admitted)  # blocks in send
        await entered.wait()
        ep.start_soon(ep.publish, {"a": 1})  # waits for the lock
        await anyio.sleep_forever()

    server = Daemon(run)
    async with serving(server):
        (rid,) = await start(server, mode="live", capabilities=frozenset({"publish"}), link=_Link(None, publish))
        await entered.wait()
        await anyio.sleep(0.02)
        async with anyio.create_task_group() as tg:
            tg.start_soon(server._end_episodes, [rid], [{}])
            await anyio.sleep(0.02)
            release.set()
    assert sent == [0]  # the admitted send completed; the queued one never started


@pytest.mark.anyio
async def test_a_live_episode_left_without_tasks_is_failed_before_any_observation(free_port):
    async def run(ep: EpisodeContext) -> None:
        return

    async with running(Daemon(run), free_port) as url, Connection(url) as conn:
        with anyio.fail_after(3), pytest.raises(ServerEpisodeError, match="no task is left"):
            await conn.start_episode(LIVE)  # EPISODE_READY precedes the run's EPISODE_ERROR
            async with conn.listening():
                await anyio.sleep_forever()


@pytest.mark.anyio
async def test_a_stopped_dispatcher_fails_the_queued_observation_and_every_later_one():
    server = LogServer(max_batch_size=8, max_wait_time=5.0)
    queued: list[_Pending] = []
    async with anyio.create_task_group() as outer:
        async with serving(server):
            (rid,) = await start(server)

            async def first() -> None:
                queued.append(await observe(server, rid, 1.0))

            outer.start_soon(first)  # waits up to 5 s for a batch when the server stops
            await anyio.sleep(0.05)
    later = await observe(server, rid, 2.0)
    assert ["stopped" in str(p.errors[0]) for p in (*queued, later)] == [True, True] and server.predicted == []
