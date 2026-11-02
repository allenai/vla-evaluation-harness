"""Shared fixtures and helpers for the vla-evaluation-harness test suite."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import socket
from typing import Any, NamedTuple

import anyio
import numpy as np
import pytest
import websockets

from vla_eval.benchmarks.base import StepBenchmark, StepResult, repeat_last_hold
from vla_eval.model_servers.base import Row, Rows
from vla_eval.model_servers.live import LiveModelServer
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.tracking import Tracker


# ---------------------------------------------------------------------------
# Reusable test doubles
# ---------------------------------------------------------------------------


class RecordingTracker(Tracker):
    """Tracker that records every hook call — drives lifecycle assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def on_eval_begin(self, eval_id, config):
        self.calls.append(("on_eval_begin", (eval_id, config)))

    def on_benchmark_begin(self, bench_name, bench_config):
        self.calls.append(("on_benchmark_begin", (bench_name, bench_config)))

    def on_episode_end(self, bench_name, task_name, ep_dict, status):
        self.calls.append(("on_episode_end", (bench_name, task_name, ep_dict, status)))

    def on_benchmark_end(self, bench_name, result):
        self.calls.append(("on_benchmark_end", (bench_name, result)))

    def on_eval_end(self, all_results):
        self.calls.append(("on_eval_end", (all_results,)))

    def close(self):
        self.calls.append(("close", ()))


class BrokenTracker(Tracker):
    """Tracker that raises on every hook — proves call_each isolates failures."""

    def on_eval_begin(self, *a, **kw):
        raise RuntimeError("backend exploded")

    def on_benchmark_begin(self, *a, **kw):
        raise RuntimeError("backend exploded")

    def on_episode_end(self, *a, **kw):
        raise RuntimeError("backend exploded")

    def on_benchmark_end(self, *a, **kw):
        raise RuntimeError("backend exploded")

    def on_eval_end(self, *a, **kw):
        raise RuntimeError("backend exploded")


class EchoModelServer(PredictModelServer):
    """Echo model server: returns obs['value'] * ones(7)."""

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        return {"actions": obs.get("value", 0) * np.ones(7, dtype=np.float32)}


class RandomActionModelServer(PredictModelServer):
    """Returns random actions. Validates that any benchmark can handle arbitrary actions."""

    def __init__(self, action_dim: int = 7, seed: int = 0) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._action_dim = action_dim

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        return {"actions": self._rng.standard_normal(self._action_dim).astype(np.float32)}


EMPTY = -2.0  # an observation value _Log answers with an empty chunk


class Call(NamedTuple):
    value: float
    step: int
    session: str
    t: float  # monotonic, at the call's start


class _Log:
    """Echoes ``value`` (``out``: "echo" a ``[chunk_size, 7]`` chunk or ``[7]``, "ramp" ``value + arange(42)``
    as ``[6, 7]``, "noactions" ``{"foo": value}``) after ``delay``; a negative value raises.  Logs calls, model
    threads, starts and ends; a ``broken`` task fails to start, ``suite_chunk`` sets the episode's chunk size."""

    def __init__(self, *, delay: float = 0.0, out: str = "echo", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay, self.out = delay, out
        self.calls: list[Call] = []
        self.threads: set[str] = set()
        self.started: list[list[int]] = []
        self.ended: list[list[int]] = []
        self.results: list[dict[str, Any]] = []  # each end's status and failure_reason
        self.ended_busy: list[bool] = []  # an inference was running when the episode finished
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    @property
    def predicted(self) -> list[tuple[float, int]]:
        return [(c.value, c.step) for c in self.calls]

    def reset_episodes(self, rows: Rows) -> dict[int, Exception] | None:
        self.threads.add(threading.current_thread().name)
        self.started.append(rows.ids.tolist())
        tasks = [e.config.get("task", {}) for e in rows.episodes]
        for e, task in zip(rows.episodes, tasks):
            if "suite_chunk" in task:
                e.chunk_size = task["suite_chunk"]
        return {i: ValueError("cannot load this task") for i, task in enumerate(tasks) if task.get("broken")}

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        v = float(obs.get("value", 0.0))
        with self._lock:
            self.threads.add(threading.current_thread().name)
            self.calls.append(Call(v, row.step, row.episode.session_id, time.monotonic()))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        if v == EMPTY:
            return {"actions": np.zeros((0, 7), np.float32)}
        if v < 0:
            raise ValueError("bad observation")
        if self.out == "ramp":
            return {"actions": v + np.arange(42, dtype=np.float64).reshape(6, 7)}
        if self.out == "noactions":
            return {"foo": v}
        n = getattr(self, "chunk_size", None)
        return {"actions": v * np.ones((n, 7) if n else 7, np.float32)}

    def finish_episodes(self, rows: Rows, results: list[dict[str, Any]]) -> None:
        self.threads.add(threading.current_thread().name)
        self.ended.append(rows.ids.tolist())
        self.ended_busy.append(self.active > 0)
        self.results += [{k: r.get(k) for k in ("status", "failure_reason")} for r in results]


class LogServer(_Log, PredictModelServer):
    pass


class BatchLogServer(LogServer):
    """``predict_batch``: logs each batch's row ids and steps; a negative value fails the whole batch."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.batches: list[list[int]] = []
        self.steps: list[list[int]] = []

    def predict_batch(self, obs: dict[str, Any], rows: Rows) -> dict[str, Any]:
        values = np.asarray(obs["value"], np.float32)
        self.batches.append(rows.ids.tolist())
        self.steps.append(rows.step.tolist())
        if (values < 0).any():
            raise ValueError("negative value")
        if self.chunk_size:
            return {"actions": values[:, None, None] * np.ones((len(rows), self.chunk_size, 7), np.float32)}
        return {"actions": values[:, None] * np.ones((len(rows), 7), np.float32)}


class LiveLogServer(_Log, LiveModelServer):
    pass


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

    def get_hold_action(self, last_action: Any) -> dict[str, Any]:
        return repeat_last_hold(last_action, 1)


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
