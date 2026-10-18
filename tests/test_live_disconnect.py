"""Live-mode disconnect isolation across the client and production server handler."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any
from unittest.mock import patch

import anyio
import numpy as np
import pytest
import websockets
from websockets.asyncio.server import serve as ws_serve

from vla_eval.benchmarks.base import Benchmark, repeat_last_hold
from vla_eval.connection import Connection
from vla_eval.model_servers.base import ModelServer, SessionContext
from vla_eval.model_servers.serve import serve_async
from vla_eval.orchestrator import Orchestrator
from vla_eval.protocol.messages import Message, MessageType, make_hello_payload, pack_message, unpack_message

from tests.conftest import stop_server, wait_for_server


class LifecycleModelServer(ModelServer):
    """No-weight server that records the production handler's episode lifecycle."""

    def __init__(self, *, hold_first_action: bool) -> None:
        self.hold_first_action = hold_first_action
        self.release_first_action = anyio.Event()
        self.first_observation_seen = anyio.Event()
        self.first_action_sent = anyio.Event()
        self.second_action_sent = anyio.Event()
        self.starts: list[tuple[str, str]] = []
        self.observations: list[tuple[str, float]] = []
        self.ends: list[str] = []
        self._episode_ordinals: dict[str, int] = {}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        ordinal = len(self.starts) + 1
        self.starts.append((ctx.session_id, ctx.episode_id))
        self._episode_ordinals[ctx.episode_id] = ordinal

    async def on_observation(self, obs: dict[str, Any], ctx: SessionContext) -> None:
        self.observations.append((ctx.episode_id, float(obs["value"])))
        ordinal = self._episode_ordinals.get(ctx.episode_id)
        if ordinal is None:
            raise RuntimeError("observation arrived without EPISODE_START")
        if ordinal == 1:
            self.first_observation_seen.set()
            if self.hold_first_action:
                await self.release_first_action.wait()
        await ctx.send_action({"actions": np.asarray([ordinal], dtype=np.float32)})
        if ordinal == 1:
            self.first_action_sent.set()
        elif ordinal == 2:
            self.second_action_sent.set()

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self.ends.append(ctx.episode_id)


class LifecycleBenchmark(Benchmark):
    """Small live benchmark that exposes its first environment step as an event."""

    first_apply_entered: anyio.Event
    release_first_apply: anyio.Event
    second_action_sent: anyio.Event
    actions_by_episode: list[list[float]]
    block_before_first_apply: bool

    @classmethod
    def reset_test_state(cls) -> None:
        cls.first_apply_entered = anyio.Event()
        cls.release_first_apply = anyio.Event()
        cls.second_action_sent = anyio.Event()
        cls.actions_by_episode = []
        cls.block_before_first_apply = False

    def __init__(self) -> None:
        self.episode_number = 0
        self.step = 0
        self.actions: list[float] = []

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": "disconnect"}]

    async def start_episode(self, task: dict[str, Any], recorder: Any = None) -> None:
        self.episode_number += 1
        self.step = 0
        self.actions = []
        type(self).actions_by_episode.append(self.actions)

    async def apply_action(self, action: dict[str, Any]) -> None:
        if self.episode_number == 1 and self.step == 0 and type(self).block_before_first_apply:
            type(self).first_apply_entered.set()
            await type(self).release_first_apply.wait()
        value = float(np.asarray(action["actions"]).reshape(-1)[0])
        self.actions.append(value)
        self.step += 1
        if self.episode_number == 1 and self.step == 1 and not type(self).block_before_first_apply:
            type(self).first_apply_entered.set()
            await type(self).release_first_apply.wait()
        elif self.episode_number == 2 and self.step == 1:
            await type(self).second_action_sent.wait()

    async def get_observation(self) -> dict[str, Any]:
        return {"value": float(self.step + 1)}

    async def is_done(self) -> bool:
        return self.step >= 3

    async def get_time(self) -> float:
        return self.step / 10.0

    async def get_result(self) -> dict[str, Any]:
        non_hold_actions = [value for value in self.actions if value != 0.0]
        return {
            "success": bool(non_hold_actions)
            and non_hold_actions[0] == float(self.episode_number)
            and all(value == float(self.episode_number) for value in non_hold_actions)
        }

    def get_hold_action(self, last_action: dict[str, Any] | None) -> dict[str, Any]:
        return repeat_last_hold(last_action, 1)


@pytest.mark.anyio
@pytest.mark.parametrize("disconnect_after_action", [False, True], ids=["before-first-action", "after-first-action"])
async def test_live_disconnect_fails_current_episode_and_isolates_next(
    free_port: int, disconnect_after_action: bool
) -> None:
    """A broken live session fails once; the next episode gets a fresh server lifecycle and actions."""
    LifecycleBenchmark.reset_test_state()
    LifecycleBenchmark.block_before_first_apply = not disconnect_after_action
    model_server = LifecycleModelServer(hold_first_action=not disconnect_after_action)
    LifecycleBenchmark.second_action_sent = model_server.second_action_sent
    server_task = asyncio.create_task(serve_async(model_server, host="127.0.0.1", port=free_port))

    created = anyio.Event()
    connections: list[Connection] = []

    class CapturedConnection(Connection):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            connections.append(self)
            created.set()

    config = {
        "server": {"url": f"ws://127.0.0.1:{free_port}", "timeout": 1.0},
        "benchmarks": [
            {
                "benchmark": "tests.test_live_disconnect:LifecycleBenchmark",
                "name": "live_disconnect",
                "mode": "live",
                "episodes_per_task": 2,
                "max_steps": 3,
                "hz": 100.0,
                "paced": False,
                "wait_first_action": disconnect_after_action,
            }
        ],
    }

    run_task: asyncio.Task[Any] | None = None
    try:
        with anyio.fail_after(8.0):
            await wait_for_server(free_port)
            with patch("vla_eval.orchestrator.Connection", CapturedConnection):
                run_task = asyncio.create_task(Orchestrator(config, no_save=True).run())
                await created.wait()
                await model_server.first_observation_seen.wait()
                await LifecycleBenchmark.first_apply_entered.wait()
                if disconnect_after_action:
                    await model_server.first_action_sent.wait()

                await connections[0]._ws.close(code=1012, reason="test disconnect")
                model_server.release_first_action.set()
                LifecycleBenchmark.release_first_apply.set()
                results = await run_task
    finally:
        model_server.release_first_action.set()
        LifecycleBenchmark.release_first_apply.set()
        LifecycleBenchmark.second_action_sent.set()
        with anyio.CancelScope(shield=True):
            if run_task is not None and not run_task.done():
                run_task.cancel()
                with anyio.move_on_after(2.0):
                    with contextlib.suppress(asyncio.CancelledError):
                        await run_task
            with anyio.move_on_after(2.0):
                await stop_server(server_task)

    episodes = results[0]["tasks"][0]["episodes"]
    assert episodes[0]["failure_reason"] == "connection_closed"
    assert episodes[1].get("failure_reason") is None
    assert episodes[1]["metrics"]["success"] is True, LifecycleBenchmark.actions_by_episode
    assert results[0]["num_errors"] == 1

    if disconnect_after_action:
        assert LifecycleBenchmark.actions_by_episode[0] == [1.0]
    else:
        assert LifecycleBenchmark.actions_by_episode[0] == []

    second_non_hold = [value for value in LifecycleBenchmark.actions_by_episode[1] if value != 0.0]
    assert second_non_hold and second_non_hold[0] == 2.0
    assert all(value == 2.0 for value in second_non_hold)

    assert len(model_server.starts) == 2
    assert model_server.starts[0][0] != model_server.starts[1][0]
    started_episode_ids = {episode_id for _, episode_id in model_server.starts}
    assert set(model_server.ends) == started_episode_ids
    assert all(episode_id for episode_id, _ in model_server.observations)
    assert {episode_id for episode_id, _ in model_server.observations}.issubset(started_episode_ids)


class ReconnectTracePeer:
    def __init__(self) -> None:
        self.connection_count = 0
        self.first_connection_closed = anyio.Event()
        self.trace: list[tuple[int, MessageType, int]] = []

    async def handler(self, ws: Any) -> None:
        self.connection_count += 1
        connection = self.connection_count
        try:
            async for raw in ws:
                msg = unpack_message(raw)
                self.trace.append((connection, msg.type, msg.seq))
                if msg.type == MessageType.HELLO:
                    reply = Message(
                        type=MessageType.HELLO,
                        payload=make_hello_payload(model_server="ReconnectTracePeer", capabilities={}),
                        seq=msg.seq,
                    )
                    await ws.send(pack_message(reply))
                elif msg.type == MessageType.EPISODE_START and connection == 1:
                    await ws.close(code=1012, reason="test reconnect")
                    self.first_connection_closed.set()
        except websockets.exceptions.ConnectionClosed:
            pass


@pytest.mark.anyio
async def test_implicit_reconnect_sends_hello_before_monotonic_business_seq() -> None:
    """HELLO must precede and have a lower seq than the send that triggered reconnect."""
    peer = ReconnectTracePeer()
    async with ws_serve(peer.handler, "127.0.0.1", 0, compression=None, max_size=None, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        conn = Connection(f"ws://127.0.0.1:{port}", timeout=1.0)
        try:
            with anyio.fail_after(5.0):
                await conn.connect()
                await conn.start_episode({"task": "first"})
                await peer.first_connection_closed.wait()
                await conn.start_episode({"task": "second"})
                await asyncio.gather(
                    conn.send_observation({"value": 1.0}),
                    conn.send_observation({"value": 2.0}),
                )
        finally:
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(2.0):
                    await conn.close()

    second_connection = [(msg_type, seq) for connection, msg_type, seq in peer.trace if connection == 2]
    assert [msg_type for msg_type, _ in second_connection] == [
        MessageType.HELLO,
        MessageType.EPISODE_START,
        MessageType.OBSERVATION,
        MessageType.OBSERVATION,
    ]
    seqs = [seq for _, seq in second_connection]
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))


class ListenerClosePeer:
    def __init__(self) -> None:
        self.close_now = anyio.Event()

    async def handler(self, ws: Any) -> None:
        raw = await ws.recv()
        hello = unpack_message(raw)
        reply = Message(
            type=MessageType.HELLO,
            payload=make_hello_payload(model_server="ListenerClosePeer", capabilities={}),
            seq=hello.seq,
        )
        await ws.send(pack_message(reply))
        await self.close_now.wait()
        await ws.close(code=1012, reason="test listener")


@pytest.mark.anyio
async def test_listening_propagates_bare_connection_closed() -> None:
    """A listener failure interrupts the body and retains the exception type used by the orchestrator."""
    peer = ListenerClosePeer()
    async with ws_serve(peer.handler, "127.0.0.1", 0, compression=None, max_size=None, ping_interval=None) as server:
        port = server.sockets[0].getsockname()[1]
        conn = Connection(f"ws://127.0.0.1:{port}", timeout=1.0)
        body_blocked = anyio.Event()
        try:
            with anyio.fail_after(5.0):
                await conn.connect()
                with pytest.raises(websockets.exceptions.ConnectionClosed):
                    async with conn.listening():
                        peer.close_now.set()
                        await body_blocked.wait()
        finally:
            peer.close_now.set()
            with anyio.CancelScope(shield=True):
                with anyio.move_on_after(2.0):
                    await conn.close()


@pytest.mark.anyio
async def test_listening_normal_exit_is_not_an_error(echo_server: str) -> None:
    """Cancelling the listener because its context completed normally stays silent."""
    conn = Connection(echo_server)
    try:
        with anyio.fail_after(5.0):
            await conn.connect()
            async with conn.listening():
                pass
    finally:
        with anyio.CancelScope(shield=True):
            with anyio.move_on_after(2.0):
                await conn.close()


@pytest.mark.anyio
async def test_sync_disconnect_stays_visible_and_reconnect_starts_clean(free_port: int) -> None:
    """Sync mode still surfaces a real disconnect and works after an explicit episode-boundary reconnect."""
    model_server = LifecycleModelServer(hold_first_action=True)
    server_task = asyncio.create_task(serve_async(model_server, host="127.0.0.1", port=free_port))
    conn = Connection(f"ws://127.0.0.1:{free_port}", timeout=1.0)
    act_task: asyncio.Task[Any] | None = None

    try:
        with anyio.fail_after(8.0):
            await wait_for_server(free_port)
            await conn.connect()
            await conn.start_episode({"task": "first", "mode": "sync"})
            act_task = asyncio.create_task(conn.act({"value": 1.0}))
            await model_server.first_observation_seen.wait()
            await conn._ws.close(code=1012, reason="test sync disconnect")
            model_server.release_first_action.set()
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await act_task

            await conn.reconnect()
            await conn.start_episode({"task": "second", "mode": "sync"})
            action = await conn.act({"value": 2.0})
            await conn.end_episode({"success": True})
    finally:
        model_server.release_first_action.set()
        with anyio.CancelScope(shield=True):
            if act_task is not None and not act_task.done():
                act_task.cancel()
                with anyio.move_on_after(2.0):
                    with contextlib.suppress(asyncio.CancelledError):
                        await act_task
            with anyio.move_on_after(2.0):
                await conn.close()
            with anyio.move_on_after(2.0):
                await stop_server(server_task)

    assert float(np.asarray(action["actions"]).reshape(-1)[0]) == 2.0
    assert len(model_server.starts) == 2
    assert model_server.starts[0][0] != model_server.starts[1][0]
    assert set(model_server.ends) == {episode_id for _, episode_id in model_server.starts}
    assert all(episode_id for episode_id, _ in model_server.observations)
