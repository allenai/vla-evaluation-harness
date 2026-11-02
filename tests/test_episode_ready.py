"""EPISODE_READY: a runner begins an episode (clock, first observation) only once the server started it."""

from __future__ import annotations

import argparse
import time
from typing import Any

import anyio
import pytest
import websockets
from websockets.asyncio.server import serve as ws_serve

from vla_eval.cli import main as cli
from vla_eval.config import EvalConfig
from vla_eval.connection import Connection
from vla_eval.model_servers.base import Rows
from vla_eval.protocol.messages import Message, MessageType, make_hello_payload, pack_message, unpack_message
from vla_eval.runners.base import ModelStartError, failure_reason
from vla_eval.runners.clock import Clock
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import LiveLogServer, LogServer, StubBenchmark, _Log, start_server, stop_server


class _SlowStart(_Log):
    """``reset_episodes`` takes 0.3 s; ``ready_at``: when each returned."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ready_at: list[float] = []

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        time.sleep(0.3)
        out = super().reset_episodes(rows)
        self.ready_at.append(time.monotonic())
        return out


class SlowLogServer(_SlowStart, LogServer):
    pass


class SlowLiveLogServer(_SlowStart, LiveLogServer):
    pass


class LogClock(Clock):
    def __init__(self) -> None:
        super().__init__()
        self.resets: list[float] = []

    def reset(self) -> None:
        self.resets.append(time.monotonic())
        super().reset()


class LogConnection(Connection):
    """``sent``: each message's type and when its send began."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sent: list[tuple[MessageType, float]] = []

    async def send(self, msg_type: MessageType, payload: dict[str, Any]) -> int:
        self.sent.append((msg_type, time.monotonic()))
        return await super().send(msg_type, payload)


def _runner(mode: str, clock: Clock | None = None) -> SyncEpisodeRunner | LiveEpisodeRunner:
    return SyncEpisodeRunner() if mode == "sync" else LiveEpisodeRunner(hz=100.0, clock=clock)


async def _ends_of(server: Any, n: int) -> list[dict[str, Any]]:
    with anyio.fail_after(3):
        while len(server.results) < n:
            await anyio.sleep(0.02)
    return server.results


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["sync", "live"])
@pytest.mark.anyio
async def test_a_slow_start_delays_the_first_observation_and_the_clock(free_port, mode):
    server = SlowLogServer() if mode == "sync" else SlowLiveLogServer()
    clock = LogClock()
    task = await start_server(server, free_port)
    try:
        conn = LogConnection(f"ws://127.0.0.1:{free_port}")
        async with conn:
            result = await _runner(mode, clock).run_episode(
                StubBenchmark(done_at_step=3), {"name": "t"}, conn, max_steps=50
            )
    finally:
        await stop_server(task)
    assert result["metrics"]["success"]
    first_obs = next(t for m, t in conn.sent if m == MessageType.OBSERVATION)
    assert first_obs >= server.ready_at[0]
    if mode == "live":
        assert clock.resets[0] >= server.ready_at[0]


@pytest.mark.parametrize("mode", ["sync", "live"])
@pytest.mark.anyio
async def test_a_failed_start_raises_before_any_observation(free_port, mode):
    server = LogServer() if mode == "sync" else LiveLogServer()
    task = await start_server(server, free_port)
    runner = _runner(mode)
    try:
        conn = LogConnection(f"ws://127.0.0.1:{free_port}")
        async with conn:
            with pytest.raises(ModelStartError, match="cannot load this task") as exc:
                await runner.run_episode(StubBenchmark(), {"name": "t", "broken": True}, conn, max_steps=5)
            assert failure_reason(exc.value) == "model_start"
            assert [m for m, _ in conn.sent] == [MessageType.HELLO, MessageType.EPISODE_START]  # no EPISODE_END

            result = await runner.run_episode(StubBenchmark(done_at_step=2), {"name": "t"}, conn, max_steps=5)
            assert result["metrics"]["success"]
            assert await _ends_of(server, 1) == [{"status": "success", "failure_reason": None}]
    finally:
        await stop_server(task)


def _raw_server(log: list[Message], *, episode_ready: bool) -> Any:
    """Answers HELLO (advertising ``episode_ready`` or not) and logs every other message, answering none."""

    async def handler(ws: Any) -> None:
        async for raw in ws:
            log.append(msg := unpack_message(raw))
            if msg.type == MessageType.HELLO:
                payload = make_hello_payload(capabilities={"episode_ready": True} if episode_ready else {})
                await ws.send(pack_message(Message(type=MessageType.HELLO, payload=payload, seq=msg.seq)))

    return handler


@pytest.mark.parametrize("mode", ["sync", "live"])
@pytest.mark.anyio
async def test_a_start_the_server_never_answers_times_out_and_is_ended(free_port, mode):
    log: list[Message] = []
    async with ws_serve(_raw_server(log, episode_ready=True), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}", timeout=0.3) as conn:
            with pytest.raises(TimeoutError, match="no EPISODE_READY within 0.3s") as exc:
                await _runner(mode).run_episode(StubBenchmark(), {"name": "t"}, conn, max_steps=5)
        assert failure_reason(exc.value) == "timeout"
        with anyio.fail_after(2):
            while len(log) < 3:
                await anyio.sleep(0.01)
    assert [m.type for m in log] == [MessageType.HELLO, MessageType.EPISODE_START, MessageType.EPISODE_END]
    assert log[2].payload == {"status": "error", "failure_reason": "timeout"}  # it may still have started


# ---------------------------------------------------------------------------
# Capability negotiation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_new_client_does_not_wait_for_an_old_server(free_port):
    log: list[Message] = []
    async with ws_serve(_raw_server(log, episode_ready=False), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}", timeout=30.0) as conn:
            with anyio.fail_after(1):
                await conn.start_episode({})
    assert log[0].payload["capabilities"] == {"episode_ready": True}


@pytest.mark.parametrize("advertise", [False, True])
@pytest.mark.anyio
async def test_the_server_sends_episode_ready_only_to_a_client_that_advertises_it(free_port, advertise):
    task = await start_server(LogServer(), free_port)

    async def send(ws: Any, msg_type: MessageType, payload: dict[str, Any], seq: int) -> None:
        await ws.send(pack_message(Message(type=msg_type, payload=payload, seq=seq)))

    try:
        async with websockets.connect(f"ws://127.0.0.1:{free_port}") as ws:
            hello = make_hello_payload(**({"capabilities": {"episode_ready": True}} if advertise else {}))
            await send(ws, MessageType.HELLO, hello, 1)
            with anyio.fail_after(3):
                assert unpack_message(await ws.recv(decode=False)).payload["capabilities"]["episode_ready"] is True
                await send(ws, MessageType.EPISODE_START, {}, 2)
                await send(ws, MessageType.OBSERVATION, {"value": 1.0}, 3)
                replies = [unpack_message(await ws.recv(decode=False)) for _ in range(1 + advertise)]
    finally:
        await stop_server(task)
    expected = [(MessageType.EPISODE_READY, 2)] if advertise else []
    assert [(r.type, r.seq) for r in replies] == [*expected, (MessageType.ACTION, 3)]


@pytest.mark.anyio
async def test_start_episode_drops_replies_of_an_earlier_episode(free_port):
    async def handler(ws: Any) -> None:
        async for raw in ws:
            msg = unpack_message(raw)
            if msg.type == MessageType.HELLO:
                payload = make_hello_payload(capabilities={"episode_ready": True})
                await ws.send(pack_message(Message(type=MessageType.HELLO, payload=payload, seq=msg.seq)))
            elif msg.type == MessageType.EPISODE_START:
                stale = Message(type=MessageType.ACTION, payload={}, seq=msg.seq - 1)
                for m in (stale, Message(type=MessageType.EPISODE_READY, payload={}, seq=msg.seq)):
                    await ws.send(pack_message(m))

    async with ws_serve(handler, "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with anyio.fail_after(2):
                await conn.start_episode({})


# ---------------------------------------------------------------------------
# wait_first_action is gone
# ---------------------------------------------------------------------------


def test_a_config_setting_wait_first_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="'wait_first_action' was removed: the runner now waits for"):
        EvalConfig.from_dict({"benchmark": "mod:Cls", "mode": "live", "wait_first_action": False})


def test_cmd_run_rejects_wait_first_action_before_starting_anything(monkeypatch, capsys) -> None:
    config = {"benchmarks": [{"benchmark": "pkg.mod:Bench", "wait_first_action": True}]}
    monkeypatch.setattr(cli, "_load_config", lambda _path: config)
    args = argparse.Namespace(config="unused.yaml", server_url=None, output_dir=None, param=None, benchmark_field=None)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_run(args)
    assert exc.value.code == 1
    assert "'wait_first_action' was removed" in capsys.readouterr().err
