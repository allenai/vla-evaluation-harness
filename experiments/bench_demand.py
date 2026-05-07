#!/usr/bin/env python3
"""Demand curve benchmark: measure observation arrival rate λ(N) from real environments.

Starts an instant-response model server, then launches N real Docker
benchmark shards against it.  Counts on_observation() calls to measure
the true observation rate including all real-world overhead (physics sim,
rendering, observation serialization, network).

Timing is measured from the **first observation arrival**, not from
subprocess launch, so container init / image pull overhead is excluded.

Prerequisites:
    - Docker image for your benchmark (e.g. libero)
    - A benchmark config YAML (will be patched to point at the instant server)

Usage:
    uv run python experiments/bench_demand.py \
        --config configs/benchmarks/libero/spatial.yaml \
        --shards 1,8,16,24,32,48 \
        --episodes-per-shard 10
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import yaml
from tqdm import tqdm

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.types import Action, Observation


# ---------------------------------------------------------------------------
# Resource monitor — samples CPU, GPU, RAM at 1-second intervals
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Background thread that samples system resources at 1-second intervals."""

    # Previous (idle, total) tick counters for delta CPU-percent math.
    # Class-scoped (not instance-scoped) so staticmethod ``_cpu_percent``
    # can update it without binding to an instance.
    _cpu_prev: tuple[int, int] | None = None

    def __init__(self) -> None:
        self._samples: list[dict[str, Any]] = []
        self._stop = False
        self._thread: Any = None

    def start(self) -> None:
        import threading

        self._stop = False
        self._samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop = True
        if self._thread:
            self._thread.join(timeout=3)
        return self._summarize()

    def _loop(self) -> None:
        while not self._stop:
            self._samples.append(self._sample())
            time.sleep(1.0)

    def _sample(self) -> dict[str, Any]:
        return {
            "cpu_pct": self._cpu_percent(),
            "ram_used_gb": self._ram_used_gb(),
            **self._gpu_stats(),
        }

    def _summarize(self) -> dict[str, Any]:
        if not self._samples:
            return {}
        # Skip first 2 samples (container startup noise) if enough data
        steady = self._samples[2:] if len(self._samples) > 4 else self._samples
        keys = steady[0].keys()
        result: dict[str, Any] = {}
        for k in keys:
            vals = sorted(s.get(k, 0) for s in steady)
            result[f"median_{k}"] = vals[len(vals) // 2]
            result[f"peak_{k}"] = vals[-1]
        return result

    @staticmethod
    def _cpu_percent() -> float:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            vals = [int(x) for x in line.split()[1:]]
            # idle is index 3; total = sum of all
            idle, total = vals[3], sum(vals)
            prev = ResourceMonitor._cpu_prev
            ResourceMonitor._cpu_prev = (idle, total)
            if prev is None:
                return 0.0
            prev_idle, prev_total = prev
            d_idle = idle - prev_idle
            d_total = total - prev_total
            return round(100.0 * (1.0 - d_idle / max(d_total, 1)), 1)
        except Exception:
            return 0.0

    @staticmethod
    def _ram_used_gb() -> float:
        try:
            with open("/proc/meminfo") as f:
                lines = {line.split(":")[0]: line for line in f}
            total = int(lines["MemTotal"].split()[1])
            avail = int(lines["MemAvailable"].split()[1])
            return round((total - avail) / 1048576, 1)  # kB → GB
        except Exception:
            return 0.0

    @staticmethod
    def _gpu_stats() -> dict[str, float]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            max_util = 0.0
            total_mem_used = 0.0
            total_mem = 0.0
            for line in out.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                max_util = max(max_util, float(parts[0]))
                total_mem_used += float(parts[1])
                total_mem += float(parts[2])
            return {
                "gpu_util_pct": round(max_util, 1),
                "gpu_mem_used_gb": round(total_mem_used / 1024, 1),
                "gpu_mem_total_gb": round(total_mem / 1024, 1),
            }
        except Exception:
            return {"gpu_util_pct": 0.0, "gpu_mem_used_gb": 0.0, "gpu_mem_total_gb": 0.0}


# ---------------------------------------------------------------------------
# Instant server
# ---------------------------------------------------------------------------


class InstantServer(PredictModelServer):
    """Returns immediately with zero-latency actions. Counts all observations."""

    def __init__(self, action_dim: int = 7) -> None:
        super().__init__(chunk_size=1)
        self.action_dim = action_dim
        self.call_count = 0
        self._pbar = tqdm()
        self._first_obs_time: float | None = None

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return {"actions": np.zeros(self.action_dim, dtype=np.float32)}

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        if self._first_obs_time is None:
            self._first_obs_time = time.monotonic()
        self.call_count += 1
        self._pbar.update(1)
        await super().on_observation(obs, ctx)

    def reset(self) -> None:
        """Reset counters for the next measurement round."""
        self.call_count = 0
        self._first_obs_time = None
        self._pbar.reset()
        self._pbar.refresh()


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _extract_action_dim(config: dict[str, Any]) -> int:
    """Return the shared benchmark action dim required by a demand sweep config."""
    action_dims = {int(bench.get("action_dim", 7)) for bench in config.get("benchmarks", [])}
    if not action_dims:
        return 7
    if len(action_dims) != 1:
        raise ValueError(f"bench_demand requires one shared action_dim across benchmarks, got {sorted(action_dims)}")
    return next(iter(action_dims))


def _patch_config(
    config: dict,
    server_url: str,
    output_dir: str,
    num_shards: int,
    episodes_per_shard: int,
) -> dict:
    """Patch config to point at our instant server and control episode count.

    Sets ``max_tasks=1`` and ``episodes_per_task = episodes_per_shard * num_shards``
    so that each shard gets exactly ``episodes_per_shard`` episodes via round-robin,
    regardless of the original config's episode/task settings.
    """
    config = copy.deepcopy(config)
    config["server"] = {"url": server_url}
    config["output_dir"] = output_dir
    episodes_per_task = episodes_per_shard * num_shards
    for bench in config.get("benchmarks", []):
        bench["max_tasks"] = 1
        bench["episodes_per_task"] = episodes_per_task
        bench["throughput_mode"] = True
    return config


def _build_shard_commands(
    config_path: str,
    config: dict,
    num_shards: int,
    gpus: str = "all",
    cpus: str | None = None,
    dev: bool = False,
) -> list[list[str]]:
    """Build docker run commands for N shards (does not launch them)."""
    from vla_eval.config import DockerConfig
    from vla_eval.docker_resources import shard_docker_flags

    docker = shutil.which("docker")
    assert docker, "docker not found"
    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    assert docker_cfg.image, "docker.image not set in config"

    config_abs = str(Path(config_path).resolve())
    results_dir = str(Path(config.get("output_dir", "./results")).resolve())
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    src_dir = Path(__file__).resolve().parents[1] / "src"

    cmds = []
    for shard_id in range(num_shards):
        container_name = f"vla-eval-demand-{os.getpid()}-{shard_id}"
        cmd = [
            docker,
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "host",
            "-v",
            f"{results_dir}:/workspace/results",
            "-v",
            f"{config_abs}:/tmp/eval_config.yaml:ro",
        ]
        if dev:
            cmd.extend(["-v", f"{src_dir}:/workspace/src"])
        for vol in docker_cfg.volumes:
            cmd.extend(["-v", vol])
        for env_str in docker_cfg.env:
            cmd.extend(["-e", env_str])
        cmd.extend(shard_docker_flags(shard_id, num_shards, cpus=cpus, gpus=gpus))
        cmd.extend(
            [
                docker_cfg.image,
                "run",
                "--no-docker",
                "--config",
                "/tmp/eval_config.yaml",
                "--shard-id",
                str(shard_id),
                "--num-shards",
                str(num_shards),
            ]
        )
        cmds.append(cmd)
    return cmds


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


async def measure_demand(
    config_path: str,
    num_shards: int,
    port: int,
    gpus: str = "all",
    cpus: str | None = None,
    episodes_per_shard: int = 10,
    timeout: float | None = None,
    dev: bool = False,
) -> dict[str, Any]:
    """Measure λ(N): real observation rate (obs/s) with N Docker shards."""
    # Load and patch config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    action_dim = _extract_action_dim(config)
    server_url = f"ws://127.0.0.1:{port}"
    tmp_dir = tempfile.mkdtemp(prefix="bench-demand-")
    patched = _patch_config(config, server_url, tmp_dir, num_shards, episodes_per_shard)
    tmp_config = os.path.join(tmp_dir, "config.yaml")
    with open(tmp_config, "w") as f:
        yaml.dump(patched, f)

    # Start instant server
    server = InstantServer(action_dim=action_dim)
    monitor = ResourceMonitor()

    # Wait for server ready
    import websockets

    result: dict[str, Any] = {}

    async with anyio.create_task_group() as tg:
        tg.start_soon(serve_async, server, "0.0.0.0", port)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                async with websockets.connect(server_url):
                    break
            except (OSError, Exception):
                await anyio.sleep(0.05)

        # Build and launch Docker shards via anyio — automatic cleanup on cancel
        server.reset()
        monitor.start()
        wall_t0 = time.monotonic()
        cmds = _build_shard_commands(tmp_config, patched, num_shards, gpus=gpus, cpus=cpus, dev=dev)
        stderr_bufs: list[bytes] = [b""] * num_shards

        async def _run_shard(idx: int, cmd: list[str]) -> None:
            result = await anyio.run_process(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
            stderr_bufs[idx] = result.stderr

        timed_out = False
        with anyio.move_on_after(timeout) as cancel_scope:
            async with anyio.create_task_group() as shard_tg:
                for i, cmd in enumerate(cmds):
                    shard_tg.start_soon(_run_shard, i, cmd)

        if cancel_scope.cancelled_caught:
            timed_out = True
            print(f"  Timeout after {timeout}s — killing remaining containers")
            pid = os.getpid()
            for i in range(num_shards):
                subprocess.run(
                    ["docker", "kill", f"vla-eval-demand-{pid}-{i}"],
                    capture_output=True,
                )

        t_end = time.monotonic()
        wall_elapsed = t_end - wall_t0

        # Elapsed from first observation (excludes container init / image pull overhead)
        if server._first_obs_time is not None:
            obs_elapsed = t_end - server._first_obs_time
        else:
            obs_elapsed = wall_elapsed

        # Log stderr from shards for diagnostics
        for i, buf in enumerate(stderr_bufs):
            stderr = buf.decode("utf-8", errors="replace").strip()
            if stderr:
                print(f"  [shard {i}] stderr ({len(buf)}B):", flush=True)
                for line in stderr.splitlines()[-8:]:
                    print(f"    {line}", flush=True)

        peak = monitor.stop()

        result = {
            "num_shards": num_shards,
            "total_requests": server.call_count,
            "elapsed": round(obs_elapsed, 3),
            "wall_elapsed": round(wall_elapsed, 3),
            "init_overhead": round(wall_elapsed - obs_elapsed, 3),
            "lambda_rps": round(server.call_count / obs_elapsed, 1) if obs_elapsed > 0 else 0,
            "timed_out": timed_out,
            **peak,
        }

        # Cancel the server task group
        tg.cancel_scope.cancel()

    server._pbar.close()

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def sweep_demand(
    config_path: str,
    shard_counts: list[int],
    port: int,
    gpus: str = "all",
    cpus: str | None = None,
    episodes_per_shard: int = 10,
    timeout: float | None = None,
    dev: bool = False,
) -> list[dict[str, Any]]:
    """Run measure_demand for each N in shard_counts. Returns list of result dicts."""
    results = []
    for n in shard_counts:
        print(f"\n--- N={n} shards (episodes_per_shard={episodes_per_shard}) ---")
        try:
            result = anyio.run(
                partial(
                    measure_demand,
                    config_path,
                    n,
                    port,
                    gpus=gpus,
                    cpus=cpus,
                    episodes_per_shard=episodes_per_shard,
                    timeout=timeout,
                    dev=dev,
                )
            )
        except KeyboardInterrupt:
            print("\n  Interrupted.")
            break
        results.append(result)
        tag = " (TIMEOUT)" if result.get("timed_out") else ""
        print(f"  observations: {result['total_requests']}{tag}")
        print(
            f"  elapsed: {result['elapsed']:.1f}s (wall: {result['wall_elapsed']:.1f}s, init: {result['init_overhead']:.1f}s)"
        )
        print(f"  λ = {result['lambda_rps']:.1f} obs/s{tag}")
    return results


def print_demand_table(results: list[dict[str, Any]]) -> None:
    """Print formatted demand curve table with resource utilization."""
    has_resources = any("median_cpu_pct" in r for r in results)
    if has_resources:
        print(f"\n{'=' * 115}")
        print(
            f"{'N':>4}  {'observations':>12}  {'elapsed':>8}  {'λ (obs/s)':>10}  "
            f"{'CPU%':>6} {'(peak)':>6}  {'GPU%':>6} {'(peak)':>6}  {'GPU_MEM':>8}  {'SYS_RAM':>8}"
        )
        print(f"{'-' * 115}")
        for r in results:
            tag = " *" if r.get("timed_out") else ""
            gpu_mem = f"{r.get('peak_gpu_mem_used_gb', 0):.1f}GB"
            sys_ram = f"{r.get('peak_ram_used_gb', 0):.1f}GB"
            print(
                f"{r['num_shards']:4d}  {r['total_requests']:12d}  "
                f"{r['elapsed']:8.1f}  {r['lambda_rps']:10.1f}{tag:2s}  "
                f"{r.get('median_cpu_pct', 0):6.1f} {r.get('peak_cpu_pct', 0):6.1f}  "
                f"{r.get('median_gpu_util_pct', 0):6.1f} {r.get('peak_gpu_util_pct', 0):6.1f}  "
                f"{gpu_mem:>8}  {sys_ram:>8}"
            )
        print(f"{'=' * 115}")
    else:
        print(f"\n{'=' * 70}")
        print(f"{'N':>4}  {'observations':>12}  {'elapsed':>10}  {'init':>8}  {'λ (obs/s)':>10}")
        print(f"{'-' * 70}")
        for r in results:
            tag = " *" if r.get("timed_out") else ""
            print(
                f"{r['num_shards']:4d}  {r['total_requests']:12d}  "
                f"{r['elapsed']:10.1f}  {r['init_overhead']:8.1f}  {r['lambda_rps']:10.1f}{tag}"
            )
        print(f"{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demand curve: measure real request rate λ(N) from Docker environments",
    )
    parser.add_argument("--config", "-c", required=True, help="Path to benchmark YAML config (must have docker.image)")
    parser.add_argument("--shards", required=True, help="Comma-separated shard counts to sweep, e.g. 1,8,16,24,32,48")
    parser.add_argument("--port", type=int, default=18925, help="Port for instant server (default: 18925)")
    parser.add_argument("--gpus", default="all", help="GPU devices for benchmarks, e.g. '0,1' (default: all)")
    parser.add_argument("--cpus", default=None, help="CPU range for benchmarks, e.g. '0-31' (default: all)")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Bind-mount local ./src into benchmark containers so bench_demand uses local code without rebuilding the image.",
    )
    parser.add_argument(
        "--episodes-per-shard",
        type=int,
        default=10,
        help="Episodes each shard runs (default: 10). Config is patched to max_tasks=1 "
        "and episodes_per_task=episodes_per_shard*N so every shard gets this many episodes.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-shard-count timeout in seconds. If a shard count takes longer, "
        "remaining containers are killed and partial results are recorded.",
    )
    args = parser.parse_args()

    shard_counts = [int(x) for x in args.shards.split(",")]

    print(f"Demand benchmark: {args.config}")
    print(f"  port={args.port}, gpus={args.gpus}, cpus={args.cpus}")
    print(f"  episodes_per_shard={args.episodes_per_shard}, timeout={args.timeout}, dev={args.dev}")
    print(f"  shard counts: {shard_counts}")

    results = sweep_demand(
        args.config,
        shard_counts,
        args.port,
        gpus=args.gpus,
        cpus=args.cpus,
        episodes_per_shard=args.episodes_per_shard,
        timeout=args.timeout,
        dev=args.dev,
    )
    if results:
        print_demand_table(results)


if __name__ == "__main__":
    main()
