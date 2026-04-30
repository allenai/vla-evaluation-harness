"""Docker resource allocation for sharded benchmark execution.

Provides :func:`shard_docker_flags` to compute ``docker run`` flags
(``--gpus``, ``--cpuset-cpus``, ``OMP_NUM_THREADS``) that isolate GPU
and CPU resources across parallel shard containers.
"""

from __future__ import annotations

import os
import subprocess


def _format_cpuset(cpu_ids: list[int]) -> str:
    """Format CPU IDs into cpuset notation (e.g. ``"0-5,12-17"``)."""
    ids = sorted(cpu_ids)
    ranges: list[str] = []
    start = prev = ids[0]
    for c in ids[1:]:
        if c == prev + 1:
            prev = c
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def parse_cpus(spec: str | None) -> list[int]:
    """Parse CPU range spec into sorted list of CPU IDs.

    Examples: ``"0-5"`` → ``[0..5]``, ``"0-5,12-17"`` → ``[0..5,12..17]``.
    ``None`` returns all host CPUs via ``os.cpu_count()``.
    """
    if spec is None:
        return list(range(os.cpu_count() or 1))
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def _detect_gpu_ids() -> list[str]:
    """Query nvidia-smi for available GPU indices."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [idx.strip() for idx in out.strip().splitlines() if idx.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["0"]


def parse_gpus(spec: str | None) -> list[str]:
    """Parse GPU spec into list of device IDs.

    ``None`` or ``"all"`` enumerates GPUs via nvidia-smi.
    ``"0,1"`` returns ``["0", "1"]``.
    """
    if spec is None or spec.strip().lower() == "all":
        return _detect_gpu_ids()
    return [g.strip() for g in spec.split(",")]


def gpu_docker_flag(spec: str | None) -> list[str]:
    """Return ``--gpus`` flag pair for a single (non-sharded) container."""
    if spec is None or spec.strip().lower() == "all":
        return ["--gpus", "all"]
    return ["--gpus", f"device={spec}"]


def tty_docker_flags() -> list[str]:
    """``-i`` / ``-t`` flags so an in-container process can read the host's terminal.

    Both attached when stdin and stdout are TTYs; ``-i`` only when just stdin is; nothing otherwise.
    Lets ``ensure_license``-style stdin prompts reach the user without breaking CI / sharded runs.
    """
    import sys

    if sys.stdin.isatty() and sys.stdout.isatty():
        return ["-i", "-t"]
    if sys.stdin.isatty():
        return ["-i"]
    return []


def shard_docker_flags(
    shard_id: int,
    num_shards: int,
    *,
    cpus: str | None = None,
    gpus: str | None = None,
) -> list[str]:
    """Compute ``docker run`` resource flags for one shard.

    Parameters:
        shard_id: Zero-based shard index.
        num_shards: Total number of shards.
        cpus: CPU spec (e.g. ``"0-31"``).  ``None`` = all host CPUs.
        gpus: GPU spec (e.g. ``"0,1"`` or ``"all"``).  ``None`` = ``"all"``.

    Returns:
        Flag list to extend a ``docker run`` command, e.g.
        ``["--gpus", "device=0", "--cpuset-cpus", "0-5",
          "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1"]``.
    """
    flags: list[str] = []

    # GPU: round-robin across available devices
    gpu_list = parse_gpus(gpus)
    device = gpu_list[shard_id % len(gpu_list)]
    flags.extend(["--gpus", f"device={device}"])

    # CPU: partition available cores across shards
    cpu_ids = parse_cpus(cpus)
    if num_shards > 1 and len(cpu_ids) >= num_shards:
        per_shard = len(cpu_ids) // num_shards
        start_idx = shard_id * per_shard
        shard_cpus = cpu_ids[start_idx : start_idx + per_shard]
        flags.extend(["--cpuset-cpus", _format_cpuset(shard_cpus)])

    # OpenMP/MKL: force single-threaded to avoid cross-container contention.  Some benchmark images
    # (e.g. CALVIN) ship CPU-only PyTorch that runs per-step tensor ops (torchvision transforms, tensor
    # creation).  Without this cap each container spawns one OpenMP thread per visible core, causing
    # massive context-switch overhead when multiple shards share a host (e.g. 8 shards × 48 threads =
    # 384 threads on 48 cores → no scaling).  Single-image transforms see no benefit from
    # multi-threaded BLAS/OpenMP, so OMP_NUM_THREADS=1 is always safe here.
    flags.extend(["-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1"])

    return flags
