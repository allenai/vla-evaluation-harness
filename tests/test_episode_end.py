"""The harness side of the callback contract: every started server episode ends with its status, and a
connection never hands an earlier episode's reply to the current one."""

from __future__ import annotations

from typing import Any

import anyio
import pytest
from websockets.asyncio.server import serve as ws_serve

from vla_eval.connection import Connection
from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import LogServer, StubBenchmark, start_server, stop_server


# ---------------------------------------------------------------------------
# The harness ends every server episode with its status
# ---------------------------------------------------------------------------


class NegativeBenchmark(StubBenchmark):
    def make_obs(self, raw_obs, task):
        return {"value": -1.0}


class BrokenStepBenchmark(StubBenchmark):
    def step(self, action):
        raise RuntimeError("physics exploded")


async def _ends_of(server: LogServer, n: int) -> list[dict[str, Any]]:
    with anyio.fail_after(3):
        while len(server.results) < n:
            await anyio.sleep(0.02)
    return server.results


@pytest.mark.anyio
async def test_sync_runner_ends_every_episode_with_its_status(free_port):
    server = LogServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            runner = SyncEpisodeRunner()
            await runner.run_episode(StubBenchmark(done_at_step=2), {"name": "t"}, conn, max_steps=5)
            await runner.run_episode(StubBenchmark(done_at_step=9), {"name": "t"}, conn, max_steps=2)
            with pytest.raises(Exception):
                await runner.run_episode(NegativeBenchmark(), {"name": "t"}, conn, max_steps=5)
            with pytest.raises(Exception):
                await runner.run_episode(BrokenStepBenchmark(), {"name": "t"}, conn, max_steps=5)
            assert await _ends_of(server, 4) == [
                {"status": "success", "failure_reason": None},
                {"status": "fail", "failure_reason": None},
                {"status": "error", "failure_reason": "model_act"},
                {"status": "error", "failure_reason": "env_step"},
            ]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_live_runner_ends_a_failed_episode_with_its_status(free_port):
    server = LogServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with pytest.raises(Exception):
                await LiveEpisodeRunner(hz=100.0).run_episode(BrokenStepBenchmark(), {"name": "t"}, conn, max_steps=5)
            assert await _ends_of(server, 1) == [{"status": "error", "failure_reason": "env_step"}]
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# Connection: replies of an earlier episode never reach the current one
# ---------------------------------------------------------------------------


def _raw_server(on_message: Any) -> Any:
    """A websocket handler that answers HELLO and hands every other message to ``on_message(ws, msg)``."""

    async def handler(ws: Any) -> None:
        async for raw in ws:
            msg = unpack_message(raw)
            if msg.type == MessageType.HELLO:
                hello = Message(type=MessageType.HELLO, payload={"harness_version": "x"}, seq=msg.seq)
                await ws.send(pack_message(hello))
            else:
                await on_message(ws, msg)

    return handler


def _action(msg: Message) -> Message:
    return Message(type=MessageType.ACTION, payload={"actions": [msg.payload["value"]]}, seq=msg.seq)


@pytest.mark.anyio
async def test_connection_drops_replies_from_before_episode_start(free_port):
    """A raw server that answers each observation, and replays the last action after every EPISODE_START."""
    last: list[Message] = []

    async def on_message(ws: Any, msg: Message) -> None:
        if msg.type == MessageType.EPISODE_START and last:
            await ws.send(pack_message(last[-1]))
        elif msg.type == MessageType.OBSERVATION:
            last.append(_action(msg))
            await ws.send(pack_message(last[-1]))

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({})
            assert (await conn.act({"value": 1.0}))["actions"] == [1.0]
            await conn.start_episode({})  # the server replays action 1.0
            assert (await conn.act({"value": 2.0}))["actions"] == [2.0]

            got: list[Any] = []
            conn.on_action(lambda a: got.append(a["actions"]))
            async with conn.listening():
                await conn.start_episode({})  # replays 2.0: dropped
                await conn.send_observation({"value": 3.0})
                with anyio.fail_after(2):
                    while not got:
                        await anyio.sleep(0.01)
            assert got == [[3.0]]


@pytest.mark.anyio
async def test_the_live_listener_drops_an_action_older_than_one_applied(free_port):
    """A raw server that answers two observations newest first."""
    held: list[Message] = []

    async def on_message(ws: Any, msg: Message) -> None:
        if msg.type == MessageType.OBSERVATION:
            held.append(_action(msg))
            if len(held) == 2:
                for m in reversed(held):
                    await ws.send(pack_message(m))

    async with ws_serve(_raw_server(on_message), "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            got: list[Any] = []
            conn.on_action(lambda a: got.append(a["actions"]))
            await conn.start_episode({})
            async with conn.listening():
                await conn.send_observation({"value": 1.0})
                await conn.send_observation({"value": 2.0})
                with anyio.fail_after(2):
                    while not got:
                        await anyio.sleep(0.01)
                await anyio.sleep(0.1)
            assert got == [[2.0]]
