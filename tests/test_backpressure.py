"""Tests for backpressure warnings in serve.py and predict.py."""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import patch

import anyio
import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers import serve as serve_module

from tests.conftest import start_server, stop_server


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class SlowModelServer(PredictModelServer):
    """Model server that sleeps during predict to simulate slow inference."""

    def __init__(self, delay: float = 0.5) -> None:
        super().__init__()
        self._delay = delay

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        time.sleep(self._delay)
        return {"actions": np.zeros(7, dtype=np.float32)}


class SlowBatchModelServer(PredictModelServer):
    """Batch model server that sleeps during predict_batch."""

    def __init__(self, delay: float = 0.5, max_batch_size: int = 2) -> None:
        super().__init__(max_batch_size=max_batch_size, max_wait_time=0.05)
        self._delay = delay

    def predict_batch(
        self,
        obs_batch: list[dict[str, Any]],
        ctx_batch: list[SessionContext],
    ) -> list[dict[str, Any]]:
        time.sleep(self._delay)
        return [{"actions": np.zeros(7, dtype=np.float32)} for _ in obs_batch]


# ---------------------------------------------------------------------------
# serve.py — in-flight backpressure monitor
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_backpressure_warning_logged(free_port, caplog):
    """Concurrent slow observations should trigger backpressure warning."""
    server = SlowModelServer(delay=0.8)

    # Use a low threshold and fast check interval for testing
    with patch.object(serve_module, "_BACKPRESSURE_CHECK_INTERVAL", 0.1), patch.object(
        serve_module, "_BACKPRESSURE_COOLDOWN", 0.0
    ):
        task = await start_server(server, free_port)
        try:
            url = f"ws://127.0.0.1:{free_port}"
            # Connect multiple clients and send observations concurrently
            async with anyio.create_task_group() as tg:

                async def send_obs(client_id: int) -> None:
                    async with Connection(url) as conn:
                        await conn.start_episode({"task": f"test_{client_id}"})
                        await conn.act({"value": 1.0})

                for i in range(5):
                    tg.start_soon(send_obs, i)

            # Check that the backpressure warning was logged
            warning_messages = [
                r.message
                for r in caplog.records
                if r.levelno == logging.WARNING and "Backpressure detected" in r.message
            ]
            assert len(warning_messages) >= 1, f"Expected at least one backpressure warning, got: {warning_messages}"
        finally:
            await stop_server(task)


@pytest.mark.anyio
async def test_inflight_counter_returns_to_zero(free_port):
    """In-flight counter should return to 0 after all observations complete."""
    server = SlowModelServer(delay=0.1)
    # Reset module-level counter
    serve_module._inflight = 0

    task = await start_server(server, free_port)
    try:
        url = f"ws://127.0.0.1:{free_port}"
        async with Connection(url) as conn:
            await conn.start_episode({"task": "test"})
            await conn.act({"value": 1.0})

        # Give server a moment to finish
        await anyio.sleep(0.1)
        assert serve_module._inflight == 0
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# predict.py — batch queue depth warning
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_batch_queue_depth_warning(free_port, caplog):
    """Queue depth exceeding 2x max_batch_size should trigger warning."""
    # Slow batch server: max_batch_size=2, slow inference
    server = SlowBatchModelServer(delay=0.5, max_batch_size=2)

    with patch(
        "vla_eval.model_servers.predict._QUEUE_DEPTH_COOLDOWN",
        0.0,
    ):
        task = await start_server(server, free_port)
        try:
            url = f"ws://127.0.0.1:{free_port}"
            # Send many concurrent observations to overflow the batch queue
            async with anyio.create_task_group() as tg:

                async def send_obs(client_id: int) -> None:
                    async with Connection(url) as conn:
                        await conn.start_episode({"task": f"test_{client_id}"})
                        await conn.act({"value": 1.0})

                for i in range(8):
                    tg.start_soon(send_obs, i)

            warning_messages = [
                r.message for r in caplog.records if r.levelno == logging.WARNING and "Batch queue depth" in r.message
            ]
            assert len(warning_messages) >= 1, f"Expected at least one queue depth warning, got: {warning_messages}"
        finally:
            await stop_server(task)
