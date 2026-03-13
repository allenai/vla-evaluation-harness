#!/usr/bin/env python3
"""Measure actual LIBERO sim_delay by timing env.step() calls.

Run inside the LIBERO Docker container with an instant model server on the host.
This script bypasses the harness and directly times the environment.

Usage (from host, with instant server on port 18926):
    docker run --rm --gpus all --network host \
      -v $(pwd):/workspace \
      libero run --no-docker --config /workspace/experiments/_sim_delay_config.yaml

Or run this script directly inside the container:
    python experiments/measure_sim_delay.py
"""

from __future__ import annotations

import time
from typing import Any

import anyio

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.types import Action, Observation


class InstantServer(PredictModelServer):
    """Returns immediately, counts calls, records per-call timestamps."""

    def __init__(self, chunk_size: int = 12) -> None:
        super().__init__(chunk_size=chunk_size)
        self.call_count = 0
        self.timestamps: list[float] = []
        self.step_timestamps: list[float] = []
        self._episode_start_time: float | None = None

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self.timestamps.append(time.monotonic())
        self.call_count += 1
        print(f"  predict() call #{self.call_count} at step {ctx.step}")
        if self.chunk_size and self.chunk_size > 1:
            return {"actions": np.zeros((self.chunk_size, 7), dtype=np.float32)}
        return {"actions": np.zeros(7, dtype=np.float32)}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        self._episode_start_time = time.monotonic()
        self.call_count = 0
        self.timestamps = []
        self.step_timestamps = []
        print(f"\n=== Episode started: {config.get('task', {}).get('name', '?')} ===")

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        self.step_timestamps.append(time.monotonic())
        await super().on_observation(obs, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_end(result, ctx)
        if not self.step_timestamps:
            return
        total_steps = len(self.step_timestamps)
        elapsed = self.step_timestamps[-1] - self.step_timestamps[0]
        step_intervals = [
            self.step_timestamps[i + 1] - self.step_timestamps[i] for i in range(len(self.step_timestamps) - 1)
        ]
        predict_intervals = (
            [self.timestamps[i + 1] - self.timestamps[i] for i in range(len(self.timestamps) - 1)]
            if len(self.timestamps) > 1
            else []
        )

        import statistics

        print(f"\n{'=' * 60}")
        print(f"Episode done: {total_steps} total steps, {self.call_count} predict() calls")
        print(f"Total elapsed: {elapsed:.3f}s")
        if step_intervals:
            print(f"\nPer-step interval (all {len(step_intervals)} intervals):")
            print(f"  mean:   {statistics.mean(step_intervals) * 1000:.1f} ms")
            print(f"  median: {statistics.median(step_intervals) * 1000:.1f} ms")
        if predict_intervals:
            print(
                f"\nPer-predict() interval ({len(predict_intervals)} intervals, "
                f"should ≈ {self.chunk_size} × per_step):"
            )
            print(f"  mean:   {statistics.mean(predict_intervals) * 1000:.1f} ms")
            print(f"  median: {statistics.median(predict_intervals) * 1000:.1f} ms")
            sim_delay_est = statistics.median(step_intervals) if step_intervals else 0
            print(f"\n  sim_delay ≈ {sim_delay_est * 1000:.1f} ms = {sim_delay_est:.4f} s")
        print(f"{'=' * 60}\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18926)
    parser.add_argument("--chunk-size", type=int, default=12)
    args = parser.parse_args()

    server = InstantServer(chunk_size=args.chunk_size)

    # Wait for server
    url = f"ws://127.0.0.1:{args.port}"
    import websockets

    async with anyio.create_task_group() as tg:
        tg.start_soon(serve_async, server, "0.0.0.0", args.port)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                async with websockets.connect(url):
                    break
            except (OSError, Exception):
                await anyio.sleep(0.05)

        print(f"Instant server ready on port {args.port}, chunk_size={args.chunk_size}")
        print("Waiting for Docker benchmark to connect...")
        print(f"Server URL for config: ws://127.0.0.1:{args.port}")

        # Keep running until killed (serve_async runs forever via anyio.sleep_forever)


if __name__ == "__main__":
    anyio.run(main)
