"""Tests for SyncEpisodeRunner, LiveEpisodeRunner, ActionBuffer, and model-thread safety of the servers."""

from __future__ import annotations

import asyncio
import anyio
import time
from functools import partial

import numpy as np
import pytest

from vla_eval.benchmarks.base import repeat_last_hold
from vla_eval.connection import Connection
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import BatchLogServer, LiveLogServer, LogServer, RandomActionModelServer, StubBenchmark
from tests.conftest import start_server, stop_server, wait_for_server


_SERVERS = {
    "echo": LogServer,
    "random": RandomActionModelServer,  # arbitrary actions
    "chunk": lambda: LogServer(chunk_size=4),
    "batch": lambda: BatchLogServer(max_batch_size=4, max_wait_time=0.1),
    "live": lambda: LiveLogServer(delay=0.02),  # actions pushed by a LiveModelServer
}


@pytest.fixture
async def server_url(request, free_port: int):
    task = await start_server(_SERVERS[request.param](), free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)


@pytest.mark.parametrize(
    ("runner", "server_url", "done_at", "max_steps"),
    [
        ("sync", "echo", 3, 50),
        ("sync", "echo", 100, 5),  # stops at max_steps
        ("sync", "random", 3, 50),
        ("sync", "chunk", 3, 50),
        ("sync", "batch", 3, 50),
        ("live", "echo", 3, 50),
        ("live", "echo", 100, 5),
        ("live", "live", 20, 50),
    ],
    indirect=["server_url"],
)
@pytest.mark.anyio
async def test_runner_ends_when_done_or_at_max_steps(request, server_url, runner, done_at, max_steps):
    episode_runner = SyncEpisodeRunner() if runner == "sync" else LiveEpisodeRunner(hz=100.0)
    async with Connection(server_url) as conn:
        result = await episode_runner.run_episode(
            StubBenchmark(done_at_step=done_at), {"name": "task_0"}, conn, max_steps=max_steps
        )
    assert result["metrics"]["success"] is (done_at <= max_steps)
    assert result["steps"] == min(done_at, max_steps)
    if runner == "live":
        assert result["rt_metrics"]["stale_action_ratio"] >= 0.0
        if request.node.callspec.params["server_url"] == "live":
            assert result["rt_metrics"]["fresh_count"] > 0  # pushed actions reached the env


@pytest.mark.anyio
async def test_batch_server_batches_concurrent_requests(free_port):
    server = BatchLogServer(max_batch_size=4, max_wait_time=0.5)
    task = await start_server(server, free_port)
    try:
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(4)]
        for c in conns:
            await c.connect()
        await anyio.sleep(0.05)

        results = await asyncio.gather(*(c.act({"value": 1.0}) for c in conns))
        assert len(results) == 4
        assert any(len(b) > 1 for b in server.batches)

        for c in conns:
            await c.close()
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# ActionBuffer tests
# ---------------------------------------------------------------------------


# Reuse the shared hold helper (absolute-control repeat-last, action_dim=7).
_repeat_hold = partial(repeat_last_hold, action_dim=7)


def test_action_buffer_initial_state():
    """Buffer starts with no action and zero metrics."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    assert not buf.has_action()
    assert not buf.is_new()
    assert buf.update_count == 0
    assert buf.stale_count == 0


def test_action_buffer_update_and_get():
    """update() then get() returns the action."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    action = {"actions": np.ones(7)}
    buf.update(action)
    assert buf.has_action()
    assert buf.is_new()
    result = buf.get()
    assert np.array_equal(result["actions"], np.ones(7))
    assert buf.update_count == 1
    assert buf.stale_count == 0


def test_action_buffer_hold_repeats_last():
    """On a stale tick, hold_fn receives the last fresh action (repeat semantics)."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7) * 2})
    buf.get()  # consume the "new" flag
    result = buf.get()  # stale → hold_fn(last_fresh) → repeat
    assert np.array_equal(result["actions"], np.ones(7) * 2)
    assert buf.stale_count == 1


def test_action_buffer_hold_null_action():
    """hold_fn can return a fixed null action (delta/velocity control)."""
    null = {"actions": np.zeros(7, dtype=np.float32)}
    buf = ActionBuffer(hold_fn=lambda last: null)
    buf.update({"actions": np.ones(7)})
    buf.get()  # consume new
    result = buf.get()  # stale → null
    assert np.allclose(result["actions"], np.zeros(7))
    assert buf.stale_count == 1


def test_action_buffer_reset():
    """reset() clears all state."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7)})
    buf.get()
    buf.reset()
    assert not buf.has_action()
    assert buf.update_count == 0
    assert buf.stale_count == 0


def test_action_buffer_metrics():
    """get_metrics() returns correct stale ratio."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7)})
    buf.get()  # 1 fresh get
    buf.get()  # 1 stale get
    buf.get()  # 1 stale get
    metrics = buf.get_metrics()
    assert metrics["update_count"] == 1
    assert metrics["fresh_count"] == 1
    assert metrics["stale_count"] == 2
    assert abs(metrics["stale_action_ratio"] - 2 / 3) < 1e-6


def test_action_buffer_metrics_count_ticks_not_arrivals():
    """Actions overwritten before any tick used them count as received, not as fresh ticks."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7)})
    buf.update({"actions": np.ones(7) * 2})  # overwrites the first before a tick reads it
    buf.get()  # fresh
    buf.get()  # stale
    metrics = buf.get_metrics()
    assert metrics["update_count"] == 2
    assert metrics["fresh_count"] == 1
    assert metrics["stale_action_ratio"] == 0.5  # 1 of 2 ticks held, not 1 of 3


# ---------------------------------------------------------------------------
# The model thread: one call at a time, always the same thread
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_predict_serialised_under_concurrent_clients(free_port):
    """With max_batch_size=1, predict() must never run concurrently."""
    server = LogServer(delay=0.05)
    task = await start_server(server, free_port)
    try:
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(6)]
        for c in conns:
            await c.connect()
        await anyio.sleep(0.05)
        results = await asyncio.gather(*(c.act({"value": 1.0}) for c in conns))
        assert len(results) == len(server.calls) == 6
        assert server.max_active == 1, f"predict() ran {server.max_active} times concurrently; expected 1"
        for c in conns:
            await c.close()
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_model_calls_stay_on_one_thread(free_port):
    """A timed-out first call, reconnects and concurrent clients never move the model to another thread."""
    server = LogServer(chunk_size=2, delay=1.5)  # the first call outlasts the first client's timeout
    task = await start_server(server, free_port)
    try:
        url = f"ws://127.0.0.1:{free_port}"
        async with Connection(url, timeout=0.5) as conn:
            await conn.start_episode({"task": {"name": "t"}})
            with pytest.raises(TimeoutError):
                await conn.act({"value": 1.0})
        server.delay = 0.0

        async def client():
            async with Connection(url) as c:
                for _ in range(2):
                    await c.start_episode({"task": {"name": "t"}})
                    for _ in range(3):
                        await c.act({"value": 1.0})
                    await c.end_episode({})

        await asyncio.gather(client(), client())
        assert len(server.threads) == 1
    finally:
        await stop_server(task)


def test_a_second_ctrl_c_exits_without_waiting(tmp_path, free_port):
    """The first SIGINT waits for the predict() in flight; a second one exits at once."""
    import os
    import signal
    import subprocess
    import sys

    script = tmp_path / "srv.py"
    script.write_text(
        "import signal, time, numpy as np\n"
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        "from vla_eval.model_servers.predict import PredictModelServer\n"
        "from vla_eval.model_servers.serve import serve\n"
        "class S(PredictModelServer):\n"
        "    def predict(self, obs, row):\n"
        "        time.sleep(30)\n"
        "        return {'actions': np.zeros(7, np.float32)}\n"
        f"serve(S(), host='127.0.0.1', port={free_port})\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], env={**os.environ, "PYTHONPATH": os.getcwd()})
    try:
        url = f"ws://127.0.0.1:{free_port}"

        async def send() -> None:
            await wait_for_server(free_port)
            async with Connection(url, timeout=0.5) as conn:
                await conn.start_episode({"task": {"name": "t"}})
                with pytest.raises(TimeoutError):
                    await conn.act({"value": 1.0})

        anyio.run(send)
        proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        assert proc.poll() is None  # still finishing the call in flight
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=3) == 130
    finally:
        proc.kill()
