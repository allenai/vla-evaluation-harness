#!/usr/bin/env python3
"""Measure real inference latency by sending requests to a running model server.

Usage:
    uv run python experiments/measure_inference_delay.py --url ws://localhost:8000 --suite libero_spatial --num-requests 20
    uv run python experiments/measure_inference_delay.py --url ws://localhost:8000 --suite calvin --image-size 200 --num-requests 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from functools import partial

import anyio

import numpy as np

from vla_eval.connection import Connection


async def measure(url: str, suite: str, num_requests: int, image_size: int = 256) -> None:
    """Send num_requests inference requests and measure round-trip time."""
    fake_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    async with Connection(url) as conn:
        await conn.start_episode({"task": {"suite": suite, "name": f"measure_delay_{suite}"}})

        latencies = []
        for i in range(num_requests):
            obs = {
                "images": {"agentview": fake_image},
                "task_description": "pick up the red cup and place it on the plate",
            }
            t0 = time.monotonic()
            action = await conn.act(obs)  # noqa: F841
            t1 = time.monotonic()
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)
            print(f"  step {i + 1:3d}/{num_requests}: {latency_ms:8.1f} ms")

        await conn.end_episode({"success": True})

    # Separate inference calls from buffer hits using a threshold
    # With chunk_size>1, buffer hits are ~1-5ms, inference calls are >>10ms
    threshold_ms = 20.0  # anything above this is a real inference call
    inference_latencies = [lat for lat in latencies if lat > threshold_ms]
    buffer_latencies = [lat for lat in latencies if lat <= threshold_ms]

    print(f"\n{'=' * 50}")
    print(f"Total steps: {len(latencies)}")
    print(f"  Inference calls (>{threshold_ms:.0f}ms): {len(inference_latencies)}")
    print(f"  Buffer hits    (≤{threshold_ms:.0f}ms): {len(buffer_latencies)}")

    if inference_latencies:
        # Skip first inference (cold start)
        warm_inf = inference_latencies[1:] if len(inference_latencies) > 1 else inference_latencies
        print(f"\nInference latency ({len(warm_inf)} warm calls, excl. first cold start):")
        print(f"  mean:   {statistics.mean(warm_inf):.1f} ms")
        print(f"  median: {statistics.median(warm_inf):.1f} ms")
        if len(warm_inf) > 1:
            print(f"  stdev:  {statistics.stdev(warm_inf):.1f} ms")
        print(f"  min:    {min(warm_inf):.1f} ms")
        print(f"  max:    {max(warm_inf):.1f} ms")
        print(f"  cold:   {inference_latencies[0]:.1f} ms")
        print(f"\n  inference_delay ≈ {statistics.median(warm_inf) / 1000:.4f} s")

    if buffer_latencies:
        print(f"\nBuffer hit latency ({len(buffer_latencies)} calls):")
        print(f"  mean:   {statistics.mean(buffer_latencies):.1f} ms")
        print(f"  median: {statistics.median(buffer_latencies):.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure CogACT inference latency")
    parser.add_argument("--url", required=True, help="WebSocket URL of running model server")
    parser.add_argument("--suite", default="libero_spatial", help="Suite name (default: libero_spatial)")
    parser.add_argument("--num-requests", type=int, default=30, help="Number of requests to send (default: 30)")
    parser.add_argument(
        "--image-size", type=int, default=256, help="NxN RGB image size (default: 256, CALVIN uses 200)"
    )
    args = parser.parse_args()

    print(f"Measuring inference delay: {args.url}, suite={args.suite}, image={args.image_size}x{args.image_size}")
    anyio.run(partial(measure, args.url, args.suite, args.num_requests, args.image_size))
