"""Tests for Connection client: roundtrip and reconnection."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.connection import Connection

from tests.conftest import start_echo_server, stop_server


@pytest.mark.anyio
async def test_server_client_roundtrip(echo_server):
    async with Connection(echo_server) as conn:
        await conn.start_episode({"task": "test"})
        result = await conn.act({"value": 3.0})
        expected = 3.0 * np.ones(7, dtype=np.float32)
        np.testing.assert_array_almost_equal(result["actions"], expected)
        await conn.end_episode({"success": True})


@pytest.mark.anyio
async def test_is_connected_property(echo_server):
    conn = Connection(echo_server)
    assert not conn.is_connected
    await conn.connect()
    assert conn.is_connected
    await conn.close()
    assert not conn.is_connected


@pytest.mark.anyio
async def test_reconnect_fails_after_max_retries():
    conn = Connection(
        "ws://127.0.0.1:19999",  # nothing listening
        max_retries=2,
        backoff_base=0.1,
    )
    with pytest.raises(ConnectionError, match="unreachable"):
        await conn.reconnect()


@pytest.mark.anyio
async def test_reconnect_after_server_restart(free_port):
    url = f"ws://127.0.0.1:{free_port}"
    task1 = await start_echo_server(free_port)

    conn = Connection(url, max_retries=5, backoff_base=0.3)
    await conn.connect()
    await conn.start_episode({"task": "test"})
    result = await conn.act({"value": 1.0})
    np.testing.assert_array_almost_equal(result["actions"], np.ones(7, dtype=np.float32))

    # Kill and restart server
    await stop_server(task1)
    task2 = await start_echo_server(free_port)

    try:
        await conn.reconnect()
        assert conn.is_connected
        await conn.start_episode({"task": "test2"})
        result = await conn.act({"value": 5.0})
        np.testing.assert_array_almost_equal(result["actions"], 5.0 * np.ones(7, dtype=np.float32))
    finally:
        await conn.close()
        await stop_server(task2)


@pytest.mark.anyio
async def test_ensure_connected_auto_reconnects(free_port):
    url = f"ws://127.0.0.1:{free_port}"
    task1 = await start_echo_server(free_port)

    conn = Connection(url, max_retries=5, backoff_base=0.3)
    await conn.connect()
    await conn.start_episode({"task": "test"})
    result = await conn.act({"value": 2.0})
    np.testing.assert_array_almost_equal(result["actions"], 2.0 * np.ones(7, dtype=np.float32))

    # Kill and restart — start_episode should auto-reconnect
    await stop_server(task1)
    task2 = await start_echo_server(free_port)

    try:
        await conn.start_episode({"task": "test2"})
        result = await conn.act({"value": 7.0})
        np.testing.assert_array_almost_equal(result["actions"], 7.0 * np.ones(7, dtype=np.float32))
    finally:
        await conn.close()
        await stop_server(task2)


@pytest.mark.anyio
async def test_close_is_idempotent():
    conn = Connection("ws://127.0.0.1:19999")
    await conn.close()
    await conn.close()
    assert not conn.is_connected
