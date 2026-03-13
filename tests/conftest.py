"""Shared fixtures and helpers for the vla-evaluation-harness test suite."""

from __future__ import annotations

import asyncio
import contextlib
import time
import socket
from typing import Any

import anyio
import numpy as np
import pytest
import websockets

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async


# ---------------------------------------------------------------------------
# Reusable test doubles
# ---------------------------------------------------------------------------


class EchoModelServer(PredictModelServer):
    """Echo model server: returns obs['value'] * ones(7)."""

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        return {"actions": obs.get("value", 0) * np.ones(7, dtype=np.float32)}


class RandomActionModelServer(PredictModelServer):
    """Returns random actions. Validates that any benchmark can handle arbitrary actions."""

    def __init__(self, action_dim: int = 7, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._action_dim = action_dim

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        return {"actions": self._rng.standard_normal(self._action_dim).astype(np.float32)}


class ChunkModelServer(PredictModelServer):
    """Returns action chunks. Validates chunk buffering with StubBenchmark."""

    def __init__(self, action_dim: int = 7) -> None:
        super().__init__(chunk_size=4, action_ensemble="newest")
        self._action_dim = action_dim

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        assert self.chunk_size is not None
        chunk = np.ones((self.chunk_size, self._action_dim), dtype=np.float32)
        return {"actions": chunk}


class StubBenchmark(StepBenchmark):
    """Minimal benchmark that succeeds after *done_at_step* steps."""

    def __init__(self, done_at_step: int = 3, num_tasks: int = 2, **kwargs: Any) -> None:
        super().__init__()
        self.done_at_step = done_at_step
        self.num_tasks = num_tasks
        self._step_count = 0

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": f"task_{i}"} for i in range(self.num_tasks)]

    def reset(self, task: dict[str, Any]) -> Any:
        self._step_count = 0
        return {"value": 1.0}

    def step(self, action: dict[str, Any]) -> StepResult:
        self._step_count += 1
        done = self._step_count >= self.done_at_step
        return StepResult(obs={"value": 1.0}, reward=1.0 if done else 0.0, done=done, info={})

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        return {"value": 1.0, "task_description": task.get("name", "")}

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        return {"success": step_result.done}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 50}


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


async def wait_for_server(port: int, timeout: float = 5.0) -> None:
    """Poll until the WebSocket server is ready to accept connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}"):
                return
        except (OSError, websockets.exceptions.InvalidHandshake):
            await anyio.sleep(0.05)
    raise TimeoutError(f"Server on port {port} did not start within {timeout}s")


async def start_server(model_server: Any, port: int) -> asyncio.Task:
    """Start any ModelServer and wait for it to accept connections."""
    task = asyncio.create_task(serve_async(model_server, port=port))
    await wait_for_server(port)
    return task


async def start_echo_server(port: int) -> asyncio.Task:
    """Start an EchoModelServer and wait for it to accept connections."""
    return await start_server(EchoModelServer(), port)


async def stop_server(task: asyncio.Task) -> None:
    """Cancel a server task and suppress CancelledError."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, anyio.get_cancelled_exc_class()):
        await task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def echo_server(free_port: int):
    """Start an echo server and yield its WebSocket URL. Cleans up on exit."""
    task = await start_echo_server(free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)


@pytest.fixture
async def random_action_server(free_port: int):
    """Start a RandomActionModelServer and yield its WebSocket URL."""
    task = await start_server(RandomActionModelServer(), free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)


@pytest.fixture
async def chunk_server(free_port: int):
    """Start a ChunkModelServer and yield its WebSocket URL."""
    task = await start_server(ChunkModelServer(), free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)
