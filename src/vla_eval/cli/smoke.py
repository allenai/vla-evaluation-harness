"""Smoke test infrastructure for vla-eval CLI commands.

Discovers configs, checks resource prerequisites, runs tests, and reports results.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SERVER_CONFIGS_DIR = CONFIGS_DIR / "model_servers"


@dataclass
class SmokeTest:
    category: str  # "validate", "server", "benchmark"
    name: str
    config_path: Path | None
    description: str


@dataclass
class SmokeResult:
    test: SmokeTest
    status: str  # "pass", "fail", "skip"
    message: str
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_validate_tests() -> list[SmokeTest]:
    """Find all benchmark configs that have a 'benchmarks' key."""
    tests: list[SmokeTest] = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        data = _load_yaml(path)
        if isinstance(data.get("benchmarks"), list):
            n = len(data["benchmarks"])
            tests.append(SmokeTest("validate", path.stem, path, f"{n} benchmark(s)"))
    return tests


def _extract_model_id(data: dict[str, Any]) -> str:
    """Extract model identifier from a server config, checking common key names."""
    args = data.get("args", {})
    for key in ("model_path", "checkpoint", "pretrained_checkpoint", "checkpoint_dir"):
        if key in args:
            return str(args[key])
    return "unknown"


def discover_server_tests(name_filter: str | None = None) -> list[SmokeTest]:
    """Find model server configs, optionally filtered by name."""
    tests: list[SmokeTest] = []
    for path in sorted(SERVER_CONFIGS_DIR.glob("*.yaml")):
        if name_filter and name_filter != "*" and name_filter not in path.stem:
            continue
        data = _load_yaml(path)
        model = _extract_model_id(data)
        tests.append(SmokeTest("server", path.stem, path, model))
    return tests


def discover_benchmark_tests(name_filter: str | None = None) -> list[SmokeTest]:
    """Find benchmark configs that have a docker image, optionally filtered by name."""
    tests: list[SmokeTest] = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        if name_filter and name_filter != "*" and name_filter not in path.stem:
            continue
        data = _load_yaml(path)
        image = (data.get("docker") or {}).get("image")
        if not image:
            continue
        # Show short image name (last component before :tag)
        short = image.rsplit("/", 1)[-1] if "/" in image else image
        tests.append(SmokeTest("benchmark", path.stem, path, short))
    return tests


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def check_uv() -> tuple[bool, str]:
    return (True, "ok") if shutil.which("uv") else (False, "uv not found on PATH")


def check_docker() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if not docker:
        return False, "docker not found on PATH"
    result = subprocess.run([docker, "info"], capture_output=True)
    if result.returncode != 0:
        return False, "docker daemon not running"
    return True, "ok"


def check_docker_image(image: str) -> tuple[bool, str]:
    docker = shutil.which("docker")
    if not docker:
        return False, "docker not found"
    result = subprocess.run([docker, "image", "inspect", image], capture_output=True)
    if result.returncode != 0:
        return False, "not pulled"
    return True, "image ready"


def _benchmark_image(config_path: Path) -> str | None:
    """Extract docker.image from a benchmark config."""
    data = _load_yaml(config_path)
    return (data.get("docker") or {}).get("image")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_validate(tests: list[SmokeTest]) -> SmokeResult:
    """Validate all benchmark configs by resolving import strings."""
    from vla_eval.benchmarks.base import Benchmark
    from vla_eval.registry import resolve_import_string

    t0 = time.monotonic()
    errors: list[str] = []
    total = 0

    for test in tests:
        assert test.config_path is not None
        data = _load_yaml(test.config_path)
        for bench in data.get("benchmarks", []):
            total += 1
            import_path = bench.get("benchmark", "")
            if not import_path or ":" not in import_path:
                errors.append(f"{test.name}: invalid import path {import_path!r}")
                continue
            try:
                cls = resolve_import_string(import_path)
                if not (isinstance(cls, type) and issubclass(cls, Benchmark)):
                    errors.append(f"{test.name}: {import_path!r} is not a Benchmark subclass")
            except Exception as e:
                errors.append(f"{test.name}: {import_path!r} -> {e}")

    dt = time.monotonic() - t0
    valid = total - len(errors)
    dummy = SmokeTest("validate", "validate", None, "")

    if errors:
        msg = f"{valid}/{total} valid"
        for e in errors[:5]:
            msg += f"\n    {e}"
        if len(errors) > 5:
            msg += f"\n    ... and {len(errors) - 5} more"
        return SmokeResult(dummy, "fail", msg, dt)
    return SmokeResult(dummy, "pass", f"{valid}/{total} configs valid", dt)


def run_server_test(test: SmokeTest, timeout: int) -> SmokeResult:
    """Run vla-eval test-server as a subprocess."""
    assert test.config_path is not None
    uv_ok, uv_msg = check_uv()
    if not uv_ok:
        return SmokeResult(test, "skip", uv_msg)

    t0 = time.monotonic()
    uv = shutil.which("uv")
    assert uv is not None
    cmd = [uv, "run", "vla-eval", "test-server", "-c", str(test.config_path), "--timeout", str(timeout)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        return SmokeResult(test, "fail", f"subprocess timeout after {timeout + 30}s", dt)

    dt = time.monotonic() - t0
    if result.returncode == 0:
        # Extract last line of stdout for summary
        lines = result.stdout.strip().splitlines()
        msg = lines[-1] if lines else "ok"
        return SmokeResult(test, "pass", msg, dt)
    # Extract error from stderr
    err_lines = result.stderr.strip().splitlines()
    msg = err_lines[-1] if err_lines else f"exit code {result.returncode}"
    return SmokeResult(test, "fail", msg, dt)


def run_benchmark_test(test: SmokeTest) -> SmokeResult:
    """Run vla-eval test-benchmark as a subprocess."""
    assert test.config_path is not None
    image = _benchmark_image(test.config_path)
    if not image:
        return SmokeResult(test, "skip", "no docker.image in config")

    docker_ok, docker_msg = check_docker()
    if not docker_ok:
        return SmokeResult(test, "skip", docker_msg)

    img_ok, img_msg = check_docker_image(image)
    if not img_ok:
        return SmokeResult(test, "skip", f"{img_msg}: {test.description}")

    t0 = time.monotonic()
    uv = shutil.which("uv")
    cmd_parts: list[str] = [uv, "run"] if uv else [sys.executable, "-m"]
    cmd = [*cmd_parts, "vla-eval", "test-benchmark", "-c", str(test.config_path), "-y"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        return SmokeResult(test, "fail", "subprocess timeout after 600s", dt)

    dt = time.monotonic() - t0
    if result.returncode == 0:
        lines = result.stdout.strip().splitlines()
        msg = lines[-1] if lines else "ok"
        return SmokeResult(test, "pass", msg, dt)
    err_lines = result.stderr.strip().splitlines()
    msg = err_lines[-1] if err_lines else f"exit code {result.returncode}"
    return SmokeResult(test, "fail", msg, dt)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Status symbols
_SYM = {"pass": "\u2713", "fail": "\u2717", "skip": "-"}


def print_list(
    validate_tests: list[SmokeTest],
    server_tests: list[SmokeTest],
    benchmark_tests: list[SmokeTest],
) -> None:
    """Print inventory of available smoke tests with prerequisite status."""
    print("\nvla-eval smoke test inventory")
    print("=" * 40)

    # Validate
    print(f"\nVALIDATE -- import string resolution ({len(validate_tests)} benchmark configs)")
    if validate_tests:
        print("  All configs can be validated.")

    # Server
    uv_ok, uv_msg = check_uv()
    print(f"\nSERVER -- model weights + GPU ({len(server_tests)} configs)")
    if not uv_ok:
        print(f"  prerequisite: {uv_msg}")
    if server_tests:
        w = max(len(t.name) for t in server_tests) + 2
        for t in server_tests:
            print(f"  {t.name:<{w}s}{t.description}")

    # Benchmark
    docker_ok, docker_msg = check_docker()
    print(f"\nBENCHMARK -- Docker + GPU ({len(benchmark_tests)} configs)")
    if not docker_ok:
        print(f"  prerequisite: {docker_msg}")
    if benchmark_tests:
        w = max(len(t.name) for t in benchmark_tests) + 2
        dw = max(len(t.description) for t in benchmark_tests) + 2
        for t in benchmark_tests:
            assert t.config_path is not None
            image = _benchmark_image(t.config_path)
            if image and docker_ok:
                img_ok, img_msg = check_docker_image(image)
                status = f"[{img_msg}]"
            elif not docker_ok:
                status = "[docker unavailable]"
            else:
                status = "[no image]"
            print(f"  {t.name:<{w}s}{t.description:<{dw}s}{status}")

    # Summary
    print(f"\nPrerequisites: uv {'ok' if uv_ok else uv_msg}  |  docker {'ok' if docker_ok else docker_msg}")
    if docker_ok:
        images = set()
        for t in benchmark_tests:
            assert t.config_path is not None
            img = _benchmark_image(t.config_path)
            if img:
                images.add(img)
        pulled = sum(1 for img in images if check_docker_image(img)[0])
        print(f"  {pulled} of {len(images)} unique Docker images pulled")
    print()


def print_report(results: list[SmokeResult]) -> None:
    """Print execution report with pass/fail/skip counts."""
    print("\nvla-eval smoke tests")
    print("=" * 40)

    current_cat = ""
    for r in results:
        if r.test.category != current_cat:
            current_cat = r.test.category
            print(f"\n{current_cat.upper()}")
        sym = _SYM.get(r.status, "?")
        name = r.test.name
        dur = f"{r.duration:.1f}s" if r.duration > 0 else ""
        print(f"  {sym} {name:<24s}{r.message:<44s}{dur:>8s}")

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    total_time = sum(r.duration for r in results)

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped    total: {total_time:.1f}s\n")

    if failed > 0:
        sys.exit(1)
