#!/usr/bin/env python3
"""Supply curve benchmark: measure server throughput μ(B) by flooding a real model server.

Connects saturating clients to an already-running model server and measures
how many requests/sec it actually processes.  No mock — this measures the
real GPU inference pipeline end-to-end.

The server must already be running (e.g. via `vla-eval serve`).

To sweep batch sizes at runtime (no server restart), use ``--sweep-batch-sizes``::

    uv run python experiments/bench_supply.py --url ws://GPU-NODE:8000 \\
        --sweep-batch-sizes 1,2,4,8,16 --num-clients 32

This sends ``GET /config?max_batch_size=B`` to the server's HTTP control
plane before each measurement round.

Usage:
    # Server already running at ws://GPU-NODE:8000
    uv run python experiments/bench_supply.py --url ws://GPU-NODE:8000

    # More clients, more requests
    uv run python experiments/bench_supply.py --url ws://GPU-NODE:8000 \\
        --num-clients 16 --requests-per-client 50

    # Custom observation (e.g. 256x256 image like LIBERO)
    uv run python experiments/bench_supply.py --url ws://GPU-NODE:8000 --image-size 256
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from functools import partial
from typing import Any

import anyio
import anyio.to_process

import numpy as np

from vla_eval.connection import Connection


async def _run_saturating_client(
    url: str,
    requests_per_client: int,
    image_size: int | None,
    suite: str = "libero_spatial",
) -> list[float]:
    """Send requests as fast as possible. Returns per-request latencies (seconds)."""
    if image_size:
        fake_image = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

        def obs_factory() -> dict:
            return {"images": {"agentview": fake_image}}
    else:

        def obs_factory() -> dict:
            return {"value": 1.0}

    latencies: list[float] = []
    async with Connection(url) as conn:
        await conn.start_episode({"task": {"name": "supply_bench", "suite": suite}})
        for _ in range(requests_per_client):
            obs = obs_factory()
            t0 = time.monotonic()
            await conn.act(obs)
            latencies.append(time.monotonic() - t0)
        await conn.end_episode({"success": True})
    return latencies


def _client_worker(url: str, requests_per_client: int, image_size: int | None, suite: str) -> list[float]:
    """Run one saturating client in its own process with its own event loop.

    Each process gets independent serialization, avoiding single-event-loop
    contention that would otherwise bottleneck on CPU-bound msgpack encoding.
    This matches real evaluation where each shard is a separate process.
    """
    return anyio.run(_run_saturating_client, url, requests_per_client, image_size, suite)


async def measure_supply(
    url: str,
    *,
    num_clients: int = 8,
    requests_per_client: int = 200,
    image_size: int | None = None,
    suite: str = "libero_spatial",
    timeout: float | None = None,
) -> dict[str, Any]:
    """Measure μ: throughput of a real model server under saturation.

    Each client runs in a **separate process** so that msgpack
    serialization parallelises across CPU cores — matching the real
    evaluation topology where every shard is a Docker container.

    Returns dict with keys: url, num_clients, total_requests, elapsed, mu_rps,
    mean_latency_ms, median_latency_ms, timed_out.
    """
    all_latencies_nested: list[list[float]] = []
    limiter = anyio.CapacityLimiter(num_clients)
    worker = partial(_client_worker, url, requests_per_client, image_size, suite)

    async def _spawn() -> None:
        result = await anyio.to_process.run_sync(worker, limiter=limiter)
        all_latencies_nested.append(result)

    t0 = time.monotonic()
    timed_out = False
    with anyio.move_on_after(timeout) as cancel_scope:
        async with anyio.create_task_group() as tg:
            for _ in range(num_clients):
                tg.start_soon(_spawn)

    if cancel_scope.cancelled_caught:
        timed_out = True
        print(f"  Timeout after {timeout}s — cancelled remaining clients")

    elapsed = time.monotonic() - t0
    all_latencies = [lat for lats in all_latencies_nested for lat in lats]
    total = len(all_latencies)

    result: dict[str, Any] = {
        "url": url,
        "num_clients": num_clients,
        "total_requests": total,
        "elapsed": round(elapsed, 3),
        "mu_rps": round(total / elapsed, 1) if elapsed > 0 else 0,
        "timed_out": timed_out,
    }
    if total > 0:
        result["mean_latency_ms"] = round(np.mean(all_latencies) * 1000, 1)
        result["median_latency_ms"] = round(np.median(all_latencies) * 1000, 1)
    else:
        result["mean_latency_ms"] = 0.0
        result["median_latency_ms"] = 0.0
    return result


def set_server_config(ws_url: str, **params: Any) -> dict[str, Any]:
    """Update server config via HTTP control plane (GET /config?key=value).

    *ws_url* is the WebSocket URL (``ws://host:port``).  The function
    converts it to an HTTP URL and sends a GET request with query params.
    """
    http_url = ws_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{http_url}/config?{qs}" if qs else f"{http_url}/config"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supply curve: measure real model server throughput μ",
    )
    parser.add_argument("--url", required=True, help="WebSocket URL of running model server")
    parser.add_argument("--num-clients", type=int, default=8, help="Concurrent saturating clients (default: 8)")
    parser.add_argument(
        "--requests-per-client", type=int, default=200, help="Requests each client sends (default: 200)"
    )
    parser.add_argument(
        "--image-size", type=int, default=None, help="Send NxN RGB images as observations (default: tiny payload)"
    )
    parser.add_argument(
        "--suite", default="libero_spatial", help="Suite name for episode_start (default: libero_spatial)"
    )
    parser.add_argument(
        "--sweep-batch-sizes",
        type=str,
        default=None,
        help="Comma-separated batch sizes to sweep via /config (e.g. 1,2,4,8,16)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-measurement timeout in seconds. If exceeded, remaining clients "
        "are cancelled and partial results are recorded.",
    )
    args = parser.parse_args()

    if args.sweep_batch_sizes:
        batch_sizes = [int(b) for b in args.sweep_batch_sizes.split(",")]
        print(f"Supply sweep: {args.url}")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  {args.num_clients} saturating clients × {args.requests_per_client} requests each")
        if args.image_size:
            print(f"  Observation: {args.image_size}×{args.image_size} RGB image")

        print(f"\n{'B':>4}  {'μ (obs/s)':>10}  {'Median lat':>10}  {'Mean lat':>10}")
        print("-" * 40)
        for B in batch_sizes:
            resp = set_server_config(args.url, max_batch_size=B)
            applied_b = resp.get("config", {}).get("max_batch_size", B)
            result = anyio.run(
                partial(
                    measure_supply,
                    args.url,
                    num_clients=args.num_clients,
                    requests_per_client=args.requests_per_client,
                    image_size=args.image_size,
                    suite=args.suite,
                    timeout=args.timeout,
                )
            )
            tag = " *" if result.get("timed_out") else ""
            print(
                f"{applied_b:>4}  {result['mu_rps']:>10.1f}  "
                f"{result['median_latency_ms']:>8.1f}ms  {result['mean_latency_ms']:>8.1f}ms{tag}"
            )
        return

    print(f"Supply benchmark: {args.url}")
    print(f"  {args.num_clients} clients × {args.requests_per_client} requests each")
    if args.image_size:
        print(f"  Observation: {args.image_size}×{args.image_size} RGB image")

    result = anyio.run(
        partial(
            measure_supply,
            args.url,
            num_clients=args.num_clients,
            requests_per_client=args.requests_per_client,
            image_size=args.image_size,
            suite=args.suite,
            timeout=args.timeout,
        )
    )

    tag = " (TIMEOUT)" if result.get("timed_out") else ""
    print(f"\nResults:{tag}")
    print(f"  Total requests:    {result['total_requests']}")
    print(f"  Elapsed:           {result['elapsed']:.3f}s")
    print(f"  Throughput μ:      {result['mu_rps']:.1f} req/s")
    print(f"  Latency (mean):    {result['mean_latency_ms']:.1f} ms")
    print(f"  Latency (median):  {result['median_latency_ms']:.1f} ms")


if __name__ == "__main__":
    main()
