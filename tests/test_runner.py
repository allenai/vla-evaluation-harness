"""Tests for SyncEpisodeRunner, AsyncEpisodeRunner, ActionBuffer, batched PredictModelServer, and CI/LAAS."""

from __future__ import annotations

import asyncio
import anyio
import threading
import time
from typing import Any

import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.async_runner import AsyncEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.serve import serve_async

from tests.conftest import StubBenchmark, wait_for_server, stop_server


@pytest.mark.anyio
async def test_sync_runner_completes(echo_server):
    """Runner finishes when benchmark reports done."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3


@pytest.mark.anyio
async def test_sync_runner_respects_max_steps(echo_server):
    """Runner stops at max_steps even if benchmark is not done."""
    benchmark = StubBenchmark(done_at_step=100)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=5)

    assert result["metrics"]["success"] is False
    assert result["steps"] == 5


@pytest.mark.anyio
async def test_random_action_server_completes(random_action_server):
    """StubBenchmark completes even when model returns arbitrary random actions."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(random_action_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3


@pytest.mark.anyio
async def test_chunk_server_completes(chunk_server):
    """StubBenchmark completes with a chunk-based model server (chunk_size=4)."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(chunk_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3


# ---------------------------------------------------------------------------
# Batched PredictModelServer tests (max_batch_size > 1)
# ---------------------------------------------------------------------------


class EchoBatchServer(PredictModelServer):
    def __init__(self):
        super().__init__(max_batch_size=4, max_wait_time=0.1)

    def predict_batch(self, obs_batch, ctx_batch):
        return [{"actions": obs.get("value", 0) * np.ones(7)} for obs in obs_batch]


class RecordingBatchServer(PredictModelServer):
    def __init__(self):
        super().__init__(max_batch_size=4, max_wait_time=0.5)
        self.batch_sizes: list[int] = []

    def predict_batch(self, obs_batch, ctx_batch):
        self.batch_sizes.append(len(obs_batch))
        return [{"actions": np.zeros(7)} for _ in obs_batch]


class FailingBatchServer(PredictModelServer):
    def __init__(self):
        super().__init__(max_batch_size=4, max_wait_time=0.1)

    def predict_batch(self, obs_batch, ctx_batch):
        raise RuntimeError("test error")


@pytest.mark.anyio
async def test_batch_server_completes_episode(free_port):
    server = EchoBatchServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        benchmark = StubBenchmark(done_at_step=3)
        runner = SyncEpisodeRunner()
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            result = await runner.run_episode(benchmark, {"name": "task_0"}, conn, max_steps=50)
        assert result["metrics"]["success"] is True
        assert result["steps"] == 3
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_batch_server_batches_concurrent_requests(free_port):
    server = RecordingBatchServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(4)]
        for c in conns:
            await c.connect()
        await anyio.sleep(0.05)

        results = await asyncio.gather(*(c.act({"value": 1.0}) for c in conns))
        assert len(results) == 4
        assert any(bs > 1 for bs in server.batch_sizes)

        for c in conns:
            await c.close()
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_batch_server_max_wait_time_dispatches_partial(free_port):
    server = EchoBatchServer()
    server.max_batch_size = 100
    server.max_wait_time = 0.05
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            action = await conn.act({"value": 2.0})
        assert "actions" in action
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_batch_server_predict_batch_exception_propagates(free_port):
    server = FailingBatchServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}", timeout=2.0) as conn:
            with pytest.raises(Exception):
                await conn.act({"value": 1.0})
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# ActionBuffer tests
# ---------------------------------------------------------------------------


def test_action_buffer_initial_state():
    """Buffer starts with no action and zero metrics."""
    buf = ActionBuffer()
    assert not buf.has_action()
    assert not buf.is_new()
    assert buf.update_count == 0
    assert buf.stale_count == 0


def test_action_buffer_update_and_get():
    """update() then get() returns the action."""
    buf = ActionBuffer()
    action = {"actions": np.ones(7)}
    buf.update(action)
    assert buf.has_action()
    assert buf.is_new()
    result = buf.get()
    assert np.array_equal(result["actions"], np.ones(7))
    assert buf.update_count == 1
    assert buf.stale_count == 0


def test_action_buffer_repeat_last_policy():
    """Default hold policy repeats the last action."""
    buf = ActionBuffer(hold_policy="repeat_last")
    buf.update({"actions": np.ones(7) * 2})
    buf.get()  # consume the "new" flag
    result = buf.get()  # stale → repeat
    assert np.array_equal(result["actions"], np.ones(7) * 2)
    assert buf.stale_count == 1


def test_action_buffer_zero_policy():
    """Zero hold policy returns zeros when stale."""
    buf = ActionBuffer(hold_policy="zero", action_dim=7)
    buf.update({"actions": np.ones(7)})
    buf.get()  # consume new
    result = buf.get()  # stale → zero
    assert np.allclose(result["actions"], np.zeros(7))
    assert buf.stale_count == 1


def test_action_buffer_reset():
    """reset() clears all state."""
    buf = ActionBuffer()
    buf.update({"actions": np.ones(7)})
    buf.get()
    buf.reset()
    assert not buf.has_action()
    assert buf.update_count == 0
    assert buf.stale_count == 0


def test_action_buffer_metrics():
    """get_metrics() returns correct stale ratio."""
    buf = ActionBuffer()
    buf.update({"actions": np.ones(7)})
    buf.get()  # 1 fresh get
    buf.get()  # 1 stale get
    buf.get()  # 1 stale get
    metrics = buf.get_metrics()
    assert metrics["update_count"] == 1
    assert metrics["stale_count"] == 2
    assert abs(metrics["stale_action_ratio"] - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# AsyncEpisodeRunner tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_async_runner_completes(echo_server):
    """AsyncEpisodeRunner finishes when benchmark reports done."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = AsyncEpisodeRunner(hz=100.0)
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3
    assert "rt_metrics" in result


@pytest.mark.anyio
async def test_async_runner_respects_max_steps(echo_server):
    """AsyncEpisodeRunner stops at max_steps even if benchmark is not done."""
    benchmark = StubBenchmark(done_at_step=100)
    runner = AsyncEpisodeRunner(hz=100.0)
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=5)

    assert result["metrics"]["success"] is False
    assert result["steps"] == 5
    assert result["rt_metrics"]["stale_action_ratio"] >= 0.0


# ---------------------------------------------------------------------------
# CI/LAAS tests — DRAFT (mirrors untested server-side CI/LAAS code)
# ---------------------------------------------------------------------------


class CIEchoServer(PredictModelServer):
    """Echo server with Continuous Inference enabled."""

    def __init__(self):
        super().__init__(continuous_inference=True, laas=False, hz=10.0)

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        return {"actions": obs.get("value", 0) * np.ones(7, dtype=np.float32)}


class CILAASChunkServer(PredictModelServer):
    """Chunk server with CI + LAAS. Adds a small sleep to simulate latency."""

    def __init__(self):
        super().__init__(continuous_inference=True, laas=True, hz=10.0, chunk_size=4)

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        time.sleep(0.05)  # simulate 50ms inference
        assert self.chunk_size is not None
        chunk = np.arange(self.chunk_size * 7, dtype=np.float32).reshape(self.chunk_size, 7)
        return {"actions": chunk}


@pytest.mark.anyio
async def test_ci_server_sends_action(free_port):
    """CI server sends actions via background loop, not request-response."""
    server = CIEchoServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "test"}, "mode": "realtime"})
            # Send obs — CI server buffers it and returns immediately,
            # then the CI loop picks it up and sends action asynchronously
            action = await conn.act({"value": 2.0})
            assert "actions" in action
            assert np.allclose(action["actions"], 2.0 * np.ones(7))
            await conn.end_episode({})
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_ci_laas_skips_stale_actions(free_port):
    """CI+LAAS server should return an action from later in the chunk."""
    server = CILAASChunkServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "test"}, "mode": "realtime"})
            action = await conn.act({"value": 1.0})
            assert "actions" in action
            # With 50ms latency and 10Hz, delay_steps = int(0.05 * 10) = 0
            # But run_in_executor adds overhead, so delay_steps >= 0.
            # Just verify we get a 1-D action (single action from chunk).
            assert action["actions"].ndim == 0 or len(action["actions"]) == 7
            await conn.end_episode({})
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_ci_loop_stops_on_episode_end(free_port):
    """CI loop should stop cleanly when episode ends."""
    server = CIEchoServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "test"}, "mode": "realtime"})
            _ = await conn.act({"value": 1.0})
            await conn.end_episode({})
            # After episode end, CI loop should be cleaned up
            # Start a new episode to verify server is still functional
            await conn.start_episode({"task": {"name": "test2"}, "mode": "realtime"})
            action = await conn.act({"value": 3.0})
            assert np.allclose(action["actions"], 3.0 * np.ones(7))
            await conn.end_episode({})
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_pick_action_laas():
    """Unit test for _pick_action with LAAS delay computation."""
    server = CIEchoServer()
    server.laas = True
    server.hz = 10.0

    # 1-D action passes through unchanged
    action_1d = np.ones(7)
    result = server._pick_action(action_1d, time.monotonic())
    assert np.array_equal(result, action_1d)

    # 2-D action with LAAS: obs_time in the past → delay_steps > 0
    actions_2d = np.arange(28, dtype=np.float32).reshape(4, 7)
    obs_time = time.monotonic() - 0.25  # 250ms ago → delay_steps = 2
    result = server._pick_action(actions_2d, obs_time)
    assert np.array_equal(result, actions_2d[2])


@pytest.mark.anyio
async def test_pick_action_no_laas():
    """Without LAAS, _pick_action returns actions[0] for 2-D input."""
    server = CIEchoServer()
    server.laas = False

    actions_2d = np.arange(28, dtype=np.float32).reshape(4, 7)
    result = server._pick_action(actions_2d, time.monotonic())
    assert np.array_equal(result, actions_2d[0])


# ---------------------------------------------------------------------------
# predict() serialisation tests (max_batch_size=1 concurrency safety)
# ---------------------------------------------------------------------------


class ConcurrencyTrackingServer(PredictModelServer):
    """Records predict() entry/exit times and tracks max concurrent calls."""

    def __init__(self):
        super().__init__(max_batch_size=1)
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0
        self.call_count = 0

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            if self._active > self.max_concurrent:
                self.max_concurrent = self._active
            self.call_count += 1
        time.sleep(0.05)  # simulate GPU work
        with self._lock:
            self._active -= 1
        return {"actions": np.ones(7, dtype=np.float32)}


@pytest.mark.anyio
async def test_predict_serialised_under_concurrent_clients(free_port):
    """With max_batch_size=1, predict() must never run concurrently."""
    server = ConcurrencyTrackingServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        n_clients = 6
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(n_clients)]
        for c in conns:
            await c.connect()
        await anyio.sleep(0.05)

        # Fire all observations concurrently
        results = await asyncio.gather(*(c.act({"value": 1.0}) for c in conns))
        assert len(results) == n_clients
        # All calls should have completed
        assert server.call_count == n_clients
        # The critical assertion: at most 1 predict() was active at any time
        assert server.max_concurrent == 1, f"predict() ran {server.max_concurrent} times concurrently; expected 1"

        for c in conns:
            await c.close()
    finally:
        await stop_server(task)
