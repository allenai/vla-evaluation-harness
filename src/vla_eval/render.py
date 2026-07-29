"""Render backend selection: run a benchmark's simulator on the GPU or on the CPU.

Top-level config key ``render: gpu|cpu`` (CLI ``--render``) picks the backend for
the whole run.  It is run-level rather than per-benchmark-entry because the
renderer is bound at the first simulator import and cannot be re-bound in-process.

Benchmarks declare support via :attr:`Benchmark.render_backends` and implement
:meth:`Benchmark.configure_render`; the helpers here supply the per-family env so
an adapter's override is one line.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Final

from vla_eval.docker_resources import is_no_gpu_spec

logger = logging.getLogger(__name__)

RENDER_MODES: Final = ("gpu", "cpu")
DEFAULT_RENDER_MODE: Final = "gpu"

_LAVAPIPE_ICD_CANDIDATES: Final = (
    "/opt/lavapipe/lvp_icd.json",
    "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
)


def normalize_render_mode(value: object) -> str:
    """Validate a ``render:`` value, defaulting to ``"gpu"`` when unset."""
    if value is None:
        return DEFAULT_RENDER_MODE
    mode = str(value).strip().lower()
    if mode not in RENDER_MODES:
        raise ValueError(f"render: {value!r} is not supported (choose one of: {', '.join(RENDER_MODES)}).")
    return mode


def check_gpu_spec_conflict(mode: str, gpus: str | None) -> None:
    """Reject a ``docker.gpus`` spec that contradicts *mode*, rather than guessing which wins.

    Shared by ``vla-eval run`` and the smoke runner so the two cannot drift: a GPU renderer in
    a device-free container fails obscurely deep inside the simulator, and a CPU renderer with
    devices attached silently keeps the GPU the mode exists to release.
    """
    if mode == "gpu" and is_no_gpu_spec(gpus):
        raise ValueError(
            f"docker.gpus: {gpus!r} conflicts with render: gpu — the simulator needs a device. "
            "Use --render cpu, or give docker.gpus a device spec."
        )
    if mode == "cpu" and gpus is not None and not is_no_gpu_spec(gpus):
        raise ValueError(
            f"docker.gpus: {gpus!r} conflicts with render: cpu — cpu attaches no device. "
            "Drop docker.gpus, or set it to 'none'."
        )


def apply_env(env: Mapping[str, str]) -> dict[str, str]:
    """Assign *env* into ``os.environ`` and return what was applied.

    Plain assignment, not ``setdefault``: benchmark images bake ``MUJOCO_GL`` /
    ``PYOPENGL_PLATFORM`` as image ENV, so a default would silently never win.
    """
    os.environ.update(env)
    return dict(env)


# ---------------------------------------------------------------------------
# Per-family CPU env
# ---------------------------------------------------------------------------


def mujoco_cpu_env() -> dict[str, str]:
    """Software-rendering env for MuJoCo / OpenGL adapters (OSMesa).

    The base image already ships ``libosmesa6``, so no Dockerfile change is needed.
    """
    return {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "LIBGL_ALWAYS_SOFTWARE": "1"}


def configure_mujoco_render(mode: str) -> dict[str, str]:
    """``configure_render`` body for MuJoCo adapters: OSMesa on cpu, image default on gpu."""
    return apply_env(mujoco_cpu_env()) if mode == "cpu" else {}


def resolve_lavapipe_icd(override_env: str) -> str | None:
    """Find the lavapipe (Mesa software Vulkan) ICD path, or None if unavailable.

    Honors ``override_env`` if set — falling back silently when an explicit user
    setting points at a missing file would be surprising, so that case logs an
    error and returns None instead of trying the implicit defaults.
    """
    user_icd = os.environ.get(override_env)
    if user_icd:
        if os.path.isfile(user_icd):
            return user_icd
        logger.error(
            "%s=%s does not exist; refusing to silently fall back to a different ICD path",
            override_env,
            user_icd,
        )
        return None
    for candidate in _LAVAPIPE_ICD_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


def lavapipe_cpu_env(icd: str) -> dict[str, str]:
    """Software-Vulkan env for SAPIEN adapters, pointing Vulkan dispatch at *icd*.

    ``LP_NUM_THREADS=4`` / single-threaded BLAS is an empirical sweet spot for Mesa
    lavapipe at 256x256 (~30% over the unset default); an existing user setting wins.
    """
    env = {"VK_ICD_FILENAMES": icd}
    for key, default in (("LP_NUM_THREADS", "4"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
        env[key] = os.environ.get(key, default)
    return env


# ---------------------------------------------------------------------------
# Capability check + application
# ---------------------------------------------------------------------------


def supported_backends(benchmark_cls: type[Any]) -> frozenset[str]:
    """Render backends *benchmark_cls* declares, defaulting to gpu-only."""
    backends = getattr(benchmark_cls, "render_backends", None)
    return frozenset(backends) if backends is not None else frozenset({DEFAULT_RENDER_MODE})


def supports_render_mode(benchmark_cls: type[Any], mode: str) -> bool:
    return mode in supported_backends(benchmark_cls)


def unsupported_render_message(name: str, benchmark_cls: type[Any], mode: str) -> str:
    backends = ", ".join(sorted(supported_backends(benchmark_cls)))
    return f"{name} does not support render: {mode} (declares render_backends: {backends})"


def apply_render_mode(benchmark_cls: type[Any], mode: str, name: str) -> dict[str, str]:
    """Configure the process renderer for *mode*, before any simulator import.

    Returns the env the benchmark actually applied — which can differ from what
    *mode* implies (e.g. RoboMME's ``ROBOMME_USE_LAVAPIPE=auto`` probe), so callers
    record it as provenance rather than re-deriving it.
    """
    if not supports_render_mode(benchmark_cls, mode):
        raise ValueError(unsupported_render_message(name, benchmark_cls, mode))
    applied = benchmark_cls.configure_render(mode)
    if applied:
        logger.info("Render backend %s for %s: %s", mode, name, applied)
    return dict(applied)
