#!/usr/bin/env python3
"""Benchmark: sharding + batch inference speedup measurement.

Simulates realistic VLA evaluation workload using configurable delays:
- sim_delay: time per simulation step (simulates MuJoCo/robosuite)
- inference_delay: time per model inference call (simulates GPU forward pass)

Usage:
    uv run python experiments/bench_parallel.py
    uv run python experiments/bench_parallel.py --shards 1,2,4,8,16,32,64 --inference-delay 0.1
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Any

import anyio
import numpy as np
import websockets

from vla_eval.connection import Connection
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.types import Observation

# ---------------------------------------------------------------------------
# Model server with simulated delay
# ---------------------------------------------------------------------------


class SlowBatchEchoServer(PredictModelServer):
    """PredictModelServer with simulated batch inference delay."""

    def __init__(self, inference_delay: float = 0.050, max_batch_size: int = 8) -> None:
        super().__init__(max_batch_size=max_batch_size, max_wait_time=0.02)
        self.inference_delay = inference_delay

    def predict_batch(
        self,
        obs_batch: list[Observation],
        ctx_batch: list[SessionContext],
    ) -> list[dict[str, Any]]:
        time.sleep(self.inference_delay)  # same delay for entire batch
        return [{"actions": np.ones(7, dtype=np.float32)} for _ in obs_batch]


# ---------------------------------------------------------------------------
# Episode runner (minimal, with sim delay)
# ---------------------------------------------------------------------------


async def run_shard(
    url: str,
    n_episodes: int,
    shard_id: int,
    steps_per_episode: int,
    sim_delay: float,
) -> float:
    """Run n_episodes sequentially on one connection. Returns elapsed seconds."""
    async with Connection(url) as conn:
        t0 = time.monotonic()
        for ep in range(n_episodes):
            await conn.start_episode({"task": f"shard{shard_id}_ep{ep}"})
            for step in range(steps_per_episode):
                _action = await conn.act({"images": {}, "task_description": "bench"})
                await anyio.sleep(sim_delay)  # simulate sim step
            await conn.end_episode({"success": True})
        return time.monotonic() - t0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _wait_for_server(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}"):
                return
        except (OSError, websockets.exceptions.InvalidHandshake):
            await anyio.sleep(0.05)
    raise TimeoutError(f"Server on port {port} not ready")


async def run_scenario(
    label: str,
    server: Any,
    n_shards: int,
    episodes_per_shard: int,
    steps_per_episode: int,
    sim_delay: float,
) -> float:
    """Start server, run n_shards concurrent shard tasks, return wall-clock time."""
    port = _free_port()
    url = f"ws://127.0.0.1:{port}"

    async with anyio.create_task_group() as srv_tg:
        srv_tg.start_soon(serve_async, server, "0.0.0.0", port)
        await _wait_for_server(port)

        t0 = time.monotonic()
        async with anyio.create_task_group() as shard_tg:
            for i in range(n_shards):
                shard_tg.start_soon(run_shard, url, episodes_per_shard, i, steps_per_episode, sim_delay)
        wall = time.monotonic() - t0

        srv_tg.cancel_scope.cancel()

    total_eps = n_shards * episodes_per_shard
    total_steps = total_eps * steps_per_episode
    print(f"  {label:45s} | {wall:7.2f}s | {total_eps:4d} eps | {total_steps:5d} steps")
    return wall


async def main(args: argparse.Namespace) -> None:
    shards = [int(x) for x in args.shards.split(",")]

    print(f"\n{'=' * 80}")
    print("Parallel Evaluation Benchmark")
    print(f"  sim_delay={args.sim_delay * 1000:.0f}ms  inference_delay={args.inference_delay * 1000:.0f}ms")
    print(
        f"  steps/episode={args.steps_per_episode}  episodes/shard={args.episodes_per_shard}  max_batch_size={args.max_batch_size}"
    )
    print(f"  shards={shards}")
    print(f"{'=' * 80}")
    print(f"  {'Scenario':45s} | {'Wall':>7s} | {'Eps':>8s} | {'Steps':>9s}")
    print(f"  {'-' * 45}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 9}")

    for n in shards:
        label = f"{n} shard{'s' if n != 1 else ''}, batch={args.max_batch_size}"
        await run_scenario(
            label,
            SlowBatchEchoServer(inference_delay=args.inference_delay, max_batch_size=args.max_batch_size),
            n,
            args.episodes_per_shard,
            args.steps_per_episode,
            args.sim_delay,
        )

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark sharding + batch inference speedup",
    )
    parser.add_argument("--sim-delay", type=float, default=0.010, help="Seconds per sim step (default: 0.010)")
    parser.add_argument(
        "--inference-delay", type=float, default=0.050, help="Seconds per inference call (default: 0.050)"
    )
    parser.add_argument("--steps-per-episode", type=int, default=20, help="Steps per episode (default: 20)")
    parser.add_argument("--episodes-per-shard", type=int, default=5, help="Episodes per shard (default: 5)")
    parser.add_argument(
        "--max-batch-size", type=int, default=8, help="Max batch size for batched scenarios (default: 8)"
    )
    parser.add_argument(
        "--shards", default="1,2,4,8", help="Comma-separated shard counts to benchmark (default: 1,2,4,8)"
    )
    parsed = parser.parse_args()
    anyio.run(main, parsed)
