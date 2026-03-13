"""Tests for bench_supply and bench_demand tuning benchmark scripts.

bench_supply is testable end-to-end: we spin up a real instant server and
let measure_supply() flood it with saturating clients.

bench_demand's core is Docker-based (can't unit-test), but we test the
helper pieces: InstantServer, _patch_config, print_demand_table.
"""

from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request

import pytest

from tests.conftest import start_server, stop_server

...
# ── bench_supply tests ──────────────────────────────────────────────────
from experiments.bench_supply import measure_supply, set_server_config  # noqa: E402


def _http_get(url: str) -> dict:
    """Blocking HTTP GET — must be called via asyncio.to_thread in async tests."""
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


@pytest.fixture
def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def instant_server_url(free_port: int):
    """Spin up an InstantServer for supply tests."""
    from experiments.bench_demand import InstantServer

    server = InstantServer()
    task = await start_server(server, free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)


@pytest.mark.anyio
async def test_measure_supply_returns_positive_throughput(instant_server_url: str):
    """measure_supply against a real instant server should report μ > 0."""
    result = await measure_supply(instant_server_url, num_clients=2, requests_per_client=5)
    assert result["mu_rps"] > 0
    assert result["total_requests"] == 10
    assert result["elapsed"] > 0
    assert result["mean_latency_ms"] > 0
    assert result["median_latency_ms"] > 0


@pytest.mark.anyio
async def test_measure_supply_with_image(instant_server_url: str):
    """measure_supply should work with image observations."""
    result = await measure_supply(instant_server_url, num_clients=2, requests_per_client=3, image_size=64)
    assert result["mu_rps"] > 0
    assert result["total_requests"] == 6


@pytest.mark.anyio
async def test_measure_supply_more_clients_more_requests(instant_server_url: str):
    """Increasing clients should still complete all requests."""
    result = await measure_supply(instant_server_url, num_clients=4, requests_per_client=5)
    assert result["total_requests"] == 20


# ── /config HTTP control plane tests ─────────────────────────────────────


@pytest.fixture
async def echo_server_url(free_port: int):
    """Spin up an EchoModelServer for /config tests."""
    from tests.conftest import EchoModelServer

    server = EchoModelServer()
    task = await start_server(server, free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)


@pytest.mark.anyio
async def test_config_read(echo_server_url: str):
    """GET /config should return current config as JSON."""
    http_url = echo_server_url.replace("ws://", "http://")
    data = await asyncio.to_thread(_http_get, f"{http_url}/config")
    assert "config" in data
    assert data["config"]["max_batch_size"] == 1  # PredictModelServer default


@pytest.mark.anyio
async def test_config_update(echo_server_url: str):
    """GET /config?max_batch_size=8 should update the attribute."""
    http_url = echo_server_url.replace("ws://", "http://")
    data = await asyncio.to_thread(_http_get, f"{http_url}/config?max_batch_size=8")
    assert data["applied"]["max_batch_size"] == 8
    assert data["config"]["max_batch_size"] == 8

    # Verify the change persists on a second read
    data = await asyncio.to_thread(_http_get, f"{http_url}/config")
    assert data["config"]["max_batch_size"] == 8


@pytest.mark.anyio
async def test_config_unknown_key_returns_422(echo_server_url: str):
    """GET /config?bad_key=1 should return 422."""
    http_url = echo_server_url.replace("ws://", "http://")

    def _expect_422():
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{http_url}/config?bad_key=1", timeout=5)
        assert exc_info.value.code == 422

    await asyncio.to_thread(_expect_422)


@pytest.mark.anyio
async def test_config_bad_type_returns_422(echo_server_url: str):
    """GET /config?max_batch_size=abc should return 422."""
    http_url = echo_server_url.replace("ws://", "http://")

    def _expect_422():
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{http_url}/config?max_batch_size=abc", timeout=5)
        assert exc_info.value.code == 422

    await asyncio.to_thread(_expect_422)


@pytest.mark.anyio
async def test_config_partial_error_still_applies_valid(echo_server_url: str):
    """Valid keys should be applied even if some keys are invalid."""
    http_url = echo_server_url.replace("ws://", "http://")
    data = await asyncio.to_thread(_http_get, f"{http_url}/config?max_batch_size=4&bad_key=1")
    assert data["applied"]["max_batch_size"] == 4
    assert "errors" in data
    assert data["config"]["max_batch_size"] == 4


@pytest.mark.anyio
async def test_set_server_config_helper(echo_server_url: str):
    """set_server_config() from bench_supply should work end-to-end."""
    result = await asyncio.to_thread(set_server_config, echo_server_url, max_batch_size=16)
    assert result["applied"]["max_batch_size"] == 16
    assert result["config"]["max_batch_size"] == 16


@pytest.mark.anyio
async def test_websocket_still_works_after_config_change(echo_server_url: str):
    """WebSocket inference should still work after /config changes."""
    import numpy as np

    from vla_eval.connection import Connection

    # Change max_wait_time (not max_batch_size — EchoModelServer has no predict_batch)
    await asyncio.to_thread(set_server_config, echo_server_url, max_wait_time=0.5)

    # WebSocket should still work
    async with Connection(echo_server_url) as conn:
        await conn.start_episode({"task": "test"})
        result = await conn.act({"value": 3.0})
        expected = 3.0 * np.ones(7, dtype=np.float32)
        np.testing.assert_array_almost_equal(result["actions"], expected)
        await conn.end_episode({"success": True})


# ── bench_demand unit tests (no Docker) ─────────────────────────────────

from experiments.bench_demand import InstantServer, _patch_config, print_demand_table  # noqa: E402


@pytest.mark.anyio
async def test_instant_server_counts_observations():
    """InstantServer.on_observation() should increment call_count for every observation."""
    from vla_eval.model_servers.base import SessionContext

    server = InstantServer()
    ctx = SessionContext(session_id="s", episode_id="e", mode="sync")

    async def _noop_send(action: dict) -> None:
        pass

    ctx._send_action_fn = _noop_send

    for i in range(4):
        await server.on_observation({"value": float(i)}, ctx)
    assert server.call_count == 4


def test_instant_server_scalar_action():
    """InstantServer should return 1D action."""
    from vla_eval.model_servers.base import SessionContext

    server = InstantServer()
    ctx = SessionContext(session_id="s", episode_id="e", mode="sync")
    result = server.predict({}, ctx)
    assert result["actions"].shape == (7,)


def test_patch_config_replaces_server_url():
    """_patch_config should set server.url, output_dir, and episode counts."""
    config = {
        "server": {"url": "ws://old:1234"},
        "output_dir": "/old",
        "foo": "bar",
        "benchmarks": [{"episodes_per_task": 50}],
    }
    patched = _patch_config(config, "ws://new:5678", "/tmp/new", num_shards=4, episodes_per_shard=10)
    assert patched["server"]["url"] == "ws://new:5678"
    assert patched["output_dir"] == "/tmp/new"
    assert patched["foo"] == "bar"
    assert patched["benchmarks"][0]["max_tasks"] == 1
    assert patched["benchmarks"][0]["episodes_per_task"] == 40  # 10 * 4
    assert patched["benchmarks"][0]["throughput_mode"] is True
    # Original unchanged
    assert config["server"]["url"] == "ws://old:1234"
    assert config["benchmarks"][0]["episodes_per_task"] == 50


def test_patch_config_adds_server_key():
    """_patch_config should work even if config has no server key."""
    config = {"docker": {"image": "test"}, "benchmarks": [{"episodes_per_task": 1}]}
    patched = _patch_config(config, "ws://localhost:9999", "/tmp/out", num_shards=8, episodes_per_shard=5)
    assert patched["server"]["url"] == "ws://localhost:9999"
    assert patched["benchmarks"][0]["max_tasks"] == 1
    assert patched["benchmarks"][0]["episodes_per_task"] == 40  # 5 * 8
    assert patched["benchmarks"][0]["throughput_mode"] is True


def test_print_demand_table(capsys):
    """print_demand_table should produce formatted output."""
    results = [
        {
            "num_shards": 1,
            "total_requests": 19,
            "elapsed": 10.0,
            "wall_elapsed": 12.0,
            "init_overhead": 2.0,
            "lambda_rps": 1.9,
        },
        {
            "num_shards": 4,
            "total_requests": 76,
            "elapsed": 10.0,
            "wall_elapsed": 11.5,
            "init_overhead": 1.5,
            "lambda_rps": 7.6,
        },
    ]
    print_demand_table(results)
    out = capsys.readouterr().out
    assert "1.9" in out
    assert "7.6" in out
    assert "19" in out
    assert "76" in out
