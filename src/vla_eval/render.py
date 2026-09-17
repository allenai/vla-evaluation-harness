"""Render backend selection: run a benchmark's simulator on the GPU or on the CPU.

Top-level config key ``render: gpu|cpu`` (CLI ``--render``) picks the backend for
the whole run.  It is run-level rather than per-benchmark-entry because the
renderer is bound at the first simulator import and cannot be re-bound in-process.

Benchmarks declare support via :attr:`Benchmark.render_backends` and implement
:meth:`Benchmark.configure_render`; the helpers here supply the per-family env so
an adapter's override is one line.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from typing import Any, Final

from vla_eval.config import EvalConfig
from vla_eval.docker_resources import NO_GPU_SPEC, is_no_gpu_spec

logger = logging.getLogger(__name__)

RENDER_MODES: Final = ("gpu", "cpu")
DEFAULT_RENDER_MODE: Final = "gpu"

# Both manifest spellings, in both locations. Mesa's distro packages dropped the arch
# suffix at 26 while conda-forge kept it, so neither name can be assumed. The image's own
# /opt/lavapipe stash is searched first: it is the one an adapter curated, and it is the
# only copy a config that bind-mounts the host's /usr/share/vulkan/icd.d cannot hide.
_LAVAPIPE_ICD_CANDIDATES: Final = (
    "/opt/lavapipe/lvp_icd.json",
    "/opt/lavapipe/lvp_icd.x86_64.json",
    "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
    "/usr/share/vulkan/icd.d/lvp_icd.json",
)

# Substring identifying a lavapipe driver in an ICD path or its library_path.
_LAVAPIPE_LIBRARY_MARK: Final = "lvp"


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


# Env applied by whichever call bound the process renderer: None until bound, {} on the
# native GPU path, the lavapipe env otherwise. The orchestrator calls configure_render once
# per benchmark *entry*, and the shipped SimplerEnv configs carry four in one process.
_sapien_render_env: dict[str, str] | None = None


def configure_sapien_render(mode: str, icd_override_env: str) -> dict[str, str]:
    """``configure_render`` body for SAPIEN adapters: lavapipe on cpu, image default on gpu.

    ``icd_override_env`` names the benchmark's escape hatch for an ICD in a path
    :func:`resolve_lavapipe_icd` does not know about.

    SAPIEN's own ``sapien_cpu`` device is not this: it cannot render at all. Software
    rasterization is an ordinary Vulkan device supplied by a different driver, so the
    whole switch is narrowing the loader to the lavapipe ICD -- which also makes the
    GPU unreachable, and is what :func:`assert_lavapipe_vulkan` later reads back.

    The renderer binds once per process, at the first sapien import. Entries after the
    first replay what that call applied, so every entry reports the renderer it actually
    ran on rather than an empty env.
    """
    global _sapien_render_env

    if _sapien_render_env is not None:
        # Re-binding is impossible either way, so a mode change is an error rather than a
        # replay: returning the other mode's env would report a renderer that is not running.
        # render: is run-level, so this only fires if a caller drives the hook directly.
        bound = "cpu" if _sapien_render_env else "gpu"
        if mode != bound:
            raise RuntimeError(
                f"render: {mode} requested after this process already bound SAPIEN to the "
                f"{bound} path; the Vulkan ICD binds at the first sapien import and cannot "
                "be re-bound."
            )
        return dict(_sapien_render_env)

    if mode == "gpu":
        _sapien_render_env = {}
        return {}
    if mode != "cpu":
        raise NotImplementedError(f"No SAPIEN render backend for {mode!r}")

    # Nothing has bound the renderer yet, so an already-imported sapien means the ICD is
    # loaded and this call's env would never reach it.
    already = sorted(m for m in ("sapien", "sapien.core", "sapien.render") if m in sys.modules)
    if already:
        raise RuntimeError(
            f"render: cpu requested after {', '.join(already)} was imported; the Vulkan ICD "
            "binds at that import and cannot be re-bound. Configure the render mode first."
        )

    icd = resolve_lavapipe_icd(icd_override_env)
    if icd is None:
        raise RuntimeError(
            "render: cpu needs the Mesa lavapipe Vulkan ICD, which is not installed here. "
            f"Install mesa-vulkan-drivers (Mesa >= 24.3), or set {icd_override_env} to its JSON path."
        )
    env = lavapipe_cpu_env(icd)
    # VK_ICD_FILENAMES is what Vulkan loaders before 1.3.207 read, VK_DRIVER_FILES what
    # later ones do; a config that sets only one would leave the other's loader broad.
    env["VK_DRIVER_FILES"] = env["VK_ICD_FILENAMES"]
    _sapien_render_env = apply_env(env)
    return dict(_sapien_render_env)


def _icd_names_lavapipe(path: str) -> bool:
    """True when *path* is a readable ICD manifest naming a lavapipe driver.

    The manifest is always opened, never trusted by file name: the override this guards
    against is exactly a GPU ICD bind-mounted over ``/opt/lavapipe/lvp_icd.json``, which a
    name check would wave through. An unreadable manifest is not lavapipe either -- the
    loader cannot use it, so whatever renders came from somewhere else.
    """
    try:
        with open(path) as fh:
            library = json.load(fh).get("ICD", {}).get("library_path", "")
    except (OSError, ValueError):
        return False
    return _LAVAPIPE_LIBRARY_MARK in os.path.basename(str(library))


def assert_lavapipe_vulkan(benchmark: str) -> str:
    """Raise unless the Vulkan loader is still pinned to lavapipe; return that ICD path.

    SAPIEN 2.x exposes no device-query API -- no ``sapien.Device``, no device accessor on
    ``SapienRenderer`` -- so "did cpu mode land on a GPU?" cannot be asked after the fact.
    What decides it is the loader's ICD list, which this reads back just before the
    simulator is built: while every entry is a lavapipe manifest, no GPU device is
    enumerable and the renderer cannot select one. A container ``env:`` entry or a
    bind-mount that re-broadened the list would otherwise restore the GPU silently, and
    the run would report cpu while rendering on the device it was meant to release.
    """
    # Which variable the loader honors depends on its version -- VK_DRIVER_FILES from
    # 1.3.207, VK_ICD_FILENAMES before it -- so both have to be clean, not just the one
    # this loader would read. Otherwise a config could pin lavapipe in the new variable
    # and the GPU in the old one, and which driver loads would depend on the image.
    listed = {name: os.environ[name] for name in ("VK_DRIVER_FILES", "VK_ICD_FILENAMES") if os.environ.get(name)}
    entries = [e for value in listed.values() for e in value.split(os.pathsep) if e]
    if not entries or any(not _icd_names_lavapipe(e) for e in entries):
        shown = ", ".join(f"{k}={v!r}" for k, v in listed.items()) or "both unset"
        raise RuntimeError(
            f"{benchmark} is in render: cpu, but the Vulkan loader is not restricted to "
            f"lavapipe ({shown}); a GPU device is still reachable, so this run would render "
            "on the device cpu mode exists to release. Check for a docker env: entry or an "
            "ICD bind-mount overriding what configure_render set."
        )
    return entries[0]


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

    Returns the env the benchmark actually applied — callers record it as
    provenance rather than re-deriving it from *mode*.
    """
    if not supports_render_mode(benchmark_cls, mode):
        raise ValueError(unsupported_render_message(name, benchmark_cls, mode))
    applied = benchmark_cls.configure_render(mode)
    if applied:
        logger.info("Render backend %s for %s: %s", mode, name, applied)
    return dict(applied)


def resolve_run_render_mode(config: dict[str, Any], override: str | None, cli_gpus: str | None = None) -> str:
    """Resolve ``render:`` (CLI wins over YAML) and reconcile it with ``docker.gpus``.

    ``render: cpu`` pins the container to no GPU at all. A CLI ``--render cpu`` is an explicit
    act that outranks a device spec sitting in the YAML, but a config asking for both at once
    contradicts itself and is rejected — as is ``--render cpu`` against an explicit ``--gpus``,
    where neither flag outranks the other.
    """
    if override is not None:
        config["render"] = override
    mode = normalize_render_mode(config.get("render"))

    docker_section = config.get("docker")
    if not isinstance(docker_section, dict):
        return mode

    gpus = docker_section.get("gpus")
    if override == "cpu" and cli_gpus is None and gpus is not None:
        logger.info("--render cpu overrides docker.gpus=%r; starting the container with no GPU", gpus)
        gpus = None
    check_gpu_spec_conflict(mode, gpus)
    if mode == "cpu":
        docker_section["gpus"] = NO_GPU_SPEC
    return mode


def check_run_render_support(config: dict[str, Any], mode: str) -> None:
    """Reject benchmarks that do not declare *mode*, before any docker pull.

    Falling back to GPU would reinstate the crash the caller is avoiding, so this
    raises instead of warning.
    """
    from vla_eval.registry import resolve_import_string

    offenders: list[str] = []
    for entry in config.get("benchmarks") or []:
        import_path = (entry or {}).get("benchmark", "")
        if not import_path:
            continue
        try:
            benchmark_cls = resolve_import_string(import_path)
        except Exception as exc:
            # Adapter deps often only exist in the benchmark image; the in-container
            # orchestrator re-checks authoritatively before any episode runs.
            logger.debug("Skipping host-side render check for %s: %s", import_path, exc)
            continue
        if not supports_render_mode(benchmark_cls, mode):
            offenders.append(
                unsupported_render_message(EvalConfig.from_dict(entry).resolved_name(), benchmark_cls, mode)
            )
    if offenders:
        raise ValueError("render: {} is not supported by:\n  {}".format(mode, "\n  ".join(offenders)))
