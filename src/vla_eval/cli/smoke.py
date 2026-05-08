"""Smoke test infrastructure for vla-eval CLI commands.

Discovers configs, checks resource prerequisites, runs tests, and reports results.
All test logic (validate, server, benchmark) is self-contained here — no subprocess
delegation to other CLI subcommands.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
console = Console(highlight=False)
CONFIGS_DIR = REPO_ROOT / "configs"

logger = logging.getLogger(__name__)


@dataclass
class SmokeTest:
    category: str  # "validate", "server", "benchmark"
    name: str
    config_path: Path | None
    description: str
    image: str = ""  # full Docker image string (benchmark only)


@dataclass
class SmokeResult:
    test: SmokeTest
    status: str  # "pass", "fail", "skip"
    message: str
    duration: float = 0.0
    stderr: str = ""  # full stderr for failed tests (used for log files)


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    from vla_eval.cli.config_loader import load_config  # lazy to avoid circular import at collection time

    return load_config(str(path))


def _classify_data(data: dict[str, Any]) -> str:
    """Classify a loaded config dict as 'server', 'benchmark', or 'unknown'."""
    if "script" in data:
        return "server"
    if (data.get("docker") or {}).get("image"):
        return "benchmark"
    if isinstance(data.get("benchmarks"), list):
        return "benchmark"
    return "unknown"


# ---------------------------------------------------------------------------
# Smoke test registry — (name, config path) pairs
# ---------------------------------------------------------------------------


def _discover_registry(subdir: str) -> dict[str, str]:
    """Auto-discover smoke configs from README.md frontmatter.

    Scans ``configs/<subdir>/*/README.md`` for YAML frontmatter with a
    ``smoke_config`` key. Supported formats::

        smoke_config: eval.yaml          # single entry, key = directory name
        smoke_config:                    # multiple entries, key = registry name
          starvla_groot: groot_simpler.yaml
          starvla_oft: oft_simpler.yaml
        smoke_config: null               # skip (no smoke test)

    Directories without frontmatter are ignored.
    """
    registry: dict[str, str] = {}
    parent = CONFIGS_DIR / subdir
    if not parent.is_dir():
        return registry
    for readme in sorted(parent.glob("*/README.md")):
        dir_name = readme.parent.name
        try:
            text = readme.read_text(encoding="utf-8")
            if not text.startswith("---"):
                continue
            end = text.find("---", 3)
            if end == -1:
                continue
            front = yaml.safe_load(text[3:end])
            if not isinstance(front, dict) or "smoke_config" not in front:
                continue
            val = front["smoke_config"]
            if val is None:
                continue
        except Exception:
            logger.warning("Failed to parse frontmatter in %s, skipping", readme)
            continue
        entries: dict[str, str] = {}
        if isinstance(val, str):
            entries[dir_name] = val
        elif isinstance(val, dict):
            entries = val
        for entry_name, filename in entries.items():
            config_path = readme.parent / filename
            if config_path.exists():
                registry[entry_name] = str(config_path.relative_to(REPO_ROOT))
    return registry


BENCHMARK_REGISTRY: dict[str, str] = _discover_registry("benchmarks")
SERVER_REGISTRY: dict[str, str] = _discover_registry("model_servers")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _extract_model_id(data: dict[str, Any]) -> str:
    """Extract model identifier from a server config, checking common key names."""
    args = data.get("args", {})
    for key in ("model_path", "checkpoint", "pretrained_checkpoint", "checkpoint_dir"):
        if key in args:
            return str(args[key])
    return "unknown"


def discover_validate_tests() -> list[SmokeTest]:
    """Find all benchmark configs that have a 'benchmarks' key."""
    tests: list[SmokeTest] = []
    for path in sorted(CONFIGS_DIR.glob("benchmarks/**/*.yaml")):
        data = _load_yaml(path)
        if isinstance(data.get("benchmarks"), list):
            n = len(data["benchmarks"])
            tests.append(SmokeTest("validate", path.stem, path, f"{n} benchmark(s)"))
    return tests


def _server_test_from_registry(name: str, rel_path: str) -> SmokeTest | None:
    """Build a SmokeTest for one server registry entry, or None if config missing."""
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    data = _load_yaml(path)
    return SmokeTest("server", name, path, _extract_model_id(data))


def _benchmark_test_from_registry(name: str, rel_path: str) -> SmokeTest | None:
    """Build a SmokeTest for one benchmark registry entry, or None if config missing."""
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    data = _load_yaml(path)
    image = (data.get("docker") or {}).get("image", "")
    short = image.rsplit("/", 1)[-1] if "/" in image else image
    return SmokeTest("benchmark", name, path, short, image=image)


def discover_server_tests(name: str | None = None) -> list[SmokeTest]:
    """Return smoke tests from the server registry, optionally filtered by exact name."""
    if name is not None:
        rel = SERVER_REGISTRY.get(name)
        if rel is None:
            return []
        t = _server_test_from_registry(name, rel)
        return [t] if t else []
    return [t for n, r in SERVER_REGISTRY.items() if (t := _server_test_from_registry(n, r)) is not None]


def discover_benchmark_tests(name: str | None = None) -> list[SmokeTest]:
    """Return smoke tests from the benchmark registry, optionally filtered by exact name."""
    if name is not None:
        rel = BENCHMARK_REGISTRY.get(name)
        if rel is None:
            return []
        t = _benchmark_test_from_registry(name, rel)
        return [t] if t else []
    return [t for n, r in BENCHMARK_REGISTRY.items() if (t := _benchmark_test_from_registry(n, r)) is not None]


def smoke_test_from_path(path: Path) -> SmokeTest:
    """Create a SmokeTest from an explicit config path (auto-detects category)."""
    data = _load_yaml(path)
    cat = _classify_data(data)
    if cat == "server":
        return SmokeTest("server", path.stem, path, _extract_model_id(data))
    if cat == "benchmark":
        image = (data.get("docker") or {}).get("image", "")
        short = image.rsplit("/", 1)[-1] if "/" in image else image
        return SmokeTest("benchmark", path.stem, path, short, image=image)
    raise ValueError(f"Cannot classify config: {path}")


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


def _free_port(host: str = "127.0.0.1") -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Execution — validate
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


# ---------------------------------------------------------------------------
# Test doubles for smoke tests
# ---------------------------------------------------------------------------


def _make_stub_benchmark(task: dict[str, Any]) -> Any:
    """Create a StubBenchmark that sends realistic image observations."""
    import numpy as np

    from vla_eval.benchmarks.base import StepBenchmark, StepResult
    from vla_eval.types import Observation, Task

    _DUMMY_OBS: dict[str, Any] = {
        "images": {
            "agentview": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        "task_description": "smoke test",
        "state": np.zeros(8, dtype=np.float32),
    }

    class _StubBenchmark(StepBenchmark):
        def __init__(self) -> None:
            super().__init__()
            self._step = 0

        def get_tasks(self) -> list[dict[str, Any]]:
            return [task]

        def reset(self, task: Task) -> Any:
            self._step = 0
            return None

        def step(self, action: Any) -> StepResult:
            self._step += 1
            done = self._step >= 3
            return StepResult(obs=None, reward=1.0 if done else 0.0, done=done, info={})

        def make_obs(self, raw_obs: Any, task: Task) -> Observation:
            return _DUMMY_OBS

        def check_done(self, step_result: StepResult) -> bool:
            return step_result.done

        def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
            return {"success": step_result.done}

        def get_metadata(self) -> dict[str, Any]:
            return {"max_steps": 50}

    return _StubBenchmark()


def _make_echo_server(action_dim: int) -> Any:
    """Create an EchoModelServer that returns zero actions."""
    import numpy as np

    from vla_eval.model_servers.base import SessionContext
    from vla_eval.model_servers.predict import PredictModelServer

    class _EchoModelServer(PredictModelServer):
        def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
            return {"actions": np.zeros(action_dim, dtype=np.float32)}

    return _EchoModelServer()


# ---------------------------------------------------------------------------
# Execution — server smoke test
# ---------------------------------------------------------------------------


def run_server_test(test: SmokeTest, timeout: int, *, gpu_id: str | None = None) -> SmokeResult:
    """Smoke-test a model server: launch it, send dummy observations, check for actions."""
    assert test.config_path is not None

    uv_ok, uv_msg = check_uv()
    if not uv_ok:
        return SmokeResult(test, "skip", uv_msg)

    config = _load_yaml(test.config_path)
    script = Path(config.get("script", "")).resolve()
    if not script.exists():
        return SmokeResult(test, "fail", f"script not found: {script}")

    uv = shutil.which("uv")
    assert uv is not None

    port = _free_port()

    # Build uv run command with --port override
    cmd: list[str] = [uv, "run", str(script)]
    for key, value in config.get("args", {}).items():
        if key == "port":
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    cmd.extend(["--port", str(port)])

    # Extract suite for servers with chunk_size_map
    task: dict[str, Any] = {"name": "smoke_test"}
    args_cfg = config.get("args", {})
    chunk_map_raw = args_cfg.get("chunk_size_map")
    if chunk_map_raw:
        chunk_map = json.loads(chunk_map_raw) if isinstance(chunk_map_raw, str) else chunk_map_raw
        if chunk_map:
            task["suite"] = next(iter(chunk_map.keys()))

    import anyio

    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    t0 = time.monotonic()
    captured_stderr: list[bytes] = []  # shared with _run() closure

    async def _run() -> dict:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id} if gpu_id is not None else None
        proc = await anyio.open_process(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)

        async def _drain_stderr() -> None:
            assert proc.stderr is not None
            async for chunk in proc.stderr:
                captured_stderr.append(chunk)

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(_drain_stderr)

                url = f"ws://127.0.0.1:{port}"
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if proc.returncode is not None:
                        tg.cancel_scope.cancel()
                        stderr = b"".join(captured_stderr).decode(errors="replace")
                        raise RuntimeError(f"Model server exited early (rc={proc.returncode}):\n{stderr}")
                    try:
                        with anyio.fail_after(1.0):
                            stream = await anyio.connect_tcp("127.0.0.1", port)
                            await stream.aclose()
                        break
                    except (OSError, TimeoutError):
                        await anyio.sleep(1.0)
                else:
                    raise TimeoutError(f"Model server did not start within {timeout}s")

                benchmark = _make_stub_benchmark(task)
                runner = SyncEpisodeRunner()
                async with Connection(url) as conn:
                    result = await runner.run_episode(benchmark, task, conn, max_steps=50)
                tg.cancel_scope.cancel()
                return result
        finally:
            proc.terminate()
            try:
                with anyio.fail_after(10):
                    await proc.wait()
            except TimeoutError:
                proc.kill()

    try:
        result = anyio.run(_run)
        dt = time.monotonic() - t0
        success = result.get("metrics", {}).get("success", result.get("success", False))
        steps = result.get("steps", 0)
        if success:
            return SmokeResult(test, "pass", f"{steps} steps, success=True", dt)
        else:
            return SmokeResult(test, "fail", f"{steps} steps, success=False", dt)
    except Exception as e:
        dt = time.monotonic() - t0
        # Unwrap ExceptionGroup (anyio TaskGroup wraps errors)
        sub_exceptions = getattr(e, "exceptions", None)
        cause = sub_exceptions[0] if sub_exceptions else e
        parts = [str(cause)]
        stderr_text = b"".join(captured_stderr).decode(errors="replace").strip()
        if stderr_text:
            stderr_tail = stderr_text.splitlines()[-10:]
            parts.append("stderr:\n    " + "\n    ".join(stderr_tail))
        msg = "\n    ".join(parts)
        return SmokeResult(test, "fail", msg, dt, stderr=stderr_text)


# ---------------------------------------------------------------------------
# Execution — benchmark smoke test
# ---------------------------------------------------------------------------


def run_benchmark_test(test: SmokeTest, timeout: int = 600, *, gpu_id: str | None = None) -> SmokeResult:
    """Smoke-test a benchmark: start EchoModelServer, run benchmark via Docker for 1 episode."""
    assert test.config_path is not None

    config = _load_yaml(test.config_path)

    from vla_eval.config import DockerConfig

    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if not docker_cfg.image:
        return SmokeResult(test, "skip", "no docker.image in config")

    docker = shutil.which("docker")
    if not docker:
        return SmokeResult(test, "skip", "docker not found on PATH")

    docker_ok, docker_msg = check_docker()
    if not docker_ok:
        return SmokeResult(test, "skip", docker_msg)

    img_ok, img_msg = check_docker_image(docker_cfg.image)
    if not img_ok:
        return SmokeResult(test, "skip", f"{img_msg}: {test.description}")

    # Extract action_dim before mutating config
    action_dim = next((b.get("action_dim", 7) for b in config.get("benchmarks", [])), 7)

    t0 = time.monotonic()

    port = _free_port()

    # Write temp config: 1 task, 1 episode, capped steps, pointing to echo server
    smoke_config = dict(config)
    smoke_config["server"] = {"url": f"ws://127.0.0.1:{port}"}
    smoke_config.pop("docker", None)
    for bench in smoke_config.get("benchmarks", []):
        bench["episodes_per_task"] = 1
        bench["max_tasks"] = 1
        if bench.get("max_steps") is None or bench["max_steps"] > 50:
            bench["max_steps"] = 50

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="vla-eval-smoke-")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            yaml.dump(smoke_config, f)
    except Exception:
        os.close(tmp_fd)
        raise

    import anyio

    from vla_eval.model_servers.serve import serve_async

    echo_server = _make_echo_server(action_dim)

    # Suppress websocket noise
    logging.getLogger("websockets").setLevel(logging.CRITICAL)

    # Start echo server in daemon thread with graceful shutdown via Event
    echo_loop: asyncio.AbstractEventLoop | None = None
    shutdown_event: asyncio.Event | None = None

    def _run_echo_server() -> None:
        nonlocal echo_loop, shutdown_event
        echo_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(echo_loop)
        shutdown_event = asyncio.Event()

        async def _serve_until_shutdown() -> None:
            assert shutdown_event is not None
            async with anyio.create_task_group() as tg:
                tg.start_soon(serve_async, echo_server, "0.0.0.0", port)
                await shutdown_event.wait()
                tg.cancel_scope.cancel()

        echo_loop.run_until_complete(_serve_until_shutdown())

    server_thread = threading.Thread(target=_run_echo_server, daemon=True)
    server_thread.start()

    # Wait for echo server readiness
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.1)

    # Run Docker container
    results_dir = tempfile.mkdtemp(prefix="vla-eval-test-")
    container_name = f"vla-eval-test-{os.getpid()}-{test.name}"

    from vla_eval.docker_resources import gpu_docker_flag

    # fmt: off
    docker_cmd: list[str] = [
        docker, "run", "--rm",
        "--name", container_name,
        "--network", "host",
        "-v", f"{results_dir}:/workspace/results",
        "-v", f"{tmp_path}:/tmp/eval_config.yaml:ro",
    ]
    # fmt: on
    gpu_spec = gpu_id if gpu_id is not None else docker_cfg.gpus
    docker_cmd.extend(gpu_docker_flag(gpu_spec))
    for vol in docker_cfg.volumes:
        docker_cmd.extend(["-v", vol])
    for env_str in docker_cfg.env:
        docker_cmd.extend(["-e", env_str])
    docker_cmd.extend([docker_cfg.image, "run", "--no-docker", "--config", "/tmp/eval_config.yaml"])

    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        rc = result.returncode
    except subprocess.TimeoutExpired:
        dt = time.monotonic() - t0
        return SmokeResult(test, "fail", f"docker timeout after {timeout}s", dt)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Signal echo server to shut down gracefully
        if echo_loop is not None and shutdown_event is not None:
            echo_loop.call_soon_threadsafe(shutdown_event.set)
            server_thread.join(timeout=5)

    dt = time.monotonic() - t0

    try:
        if rc != 0:
            err_lines = result.stderr.strip().splitlines()
            tail = err_lines[-5:] if err_lines else [f"exit code {rc}"]
            msg = "\n    ".join(tail)
            return SmokeResult(test, "fail", msg, dt, stderr=result.stderr)

        json_files = _glob.glob(os.path.join(results_dir, "*.json"))
        if json_files:
            data = json.loads(Path(json_files[0]).read_text())
            rate = data.get("mean_success", 0)
            return SmokeResult(test, "pass", f"success_rate={rate:.0%}", dt)
        return SmokeResult(test, "pass", "completed (no result file)", dt)
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SYM = {"pass": "[green]\u2713[/green]", "fail": "[red]\u2717[/red]", "skip": "[yellow]-[/yellow]"}


def _prereq_str(ok: bool, label: str, msg: str) -> str:
    return f"{label} [green]ok[/green]" if ok else f"{label} [red]{msg}[/red]"


def print_list(
    validate_tests: list[SmokeTest],
    server_tests: list[SmokeTest],
    benchmark_tests: list[SmokeTest],
) -> None:
    """Print inventory of available smoke tests with prerequisite status."""
    console.print("\n[bold]vla-eval smoke test inventory[/bold]")
    console.print("=" * 40)

    # Validate
    console.print(f"\n[bold]VALIDATE[/bold] -- import string resolution ({len(validate_tests)} benchmark configs)")
    if validate_tests:
        console.print("  All configs can be validated.")

    # Server
    uv_ok, uv_msg = check_uv()
    console.print(f"\n[bold]SERVER[/bold] -- model weights + GPU ({len(server_tests)} configs)")
    if not uv_ok:
        console.print(f"  prerequisite: [red]{uv_msg}[/red]")
    if server_tests:
        w = max(len(t.name) for t in server_tests) + 2
        for t in server_tests:
            console.print(f"  {t.name:<{w}s}{t.description}")

    # Benchmark — check each unique image once
    docker_ok, docker_msg = check_docker()
    console.print(f"\n[bold]BENCHMARK[/bold] -- Docker + GPU ({len(benchmark_tests)} configs)")
    if not docker_ok:
        console.print(f"  prerequisite: [red]{docker_msg}[/red]")
    if benchmark_tests:
        # Cache image status to avoid repeated `docker image inspect` calls
        image_status: dict[str, tuple[bool, str]] = {}
        for t in benchmark_tests:
            if t.image and t.image not in image_status:
                if docker_ok:
                    image_status[t.image] = check_docker_image(t.image)
                else:
                    image_status[t.image] = (False, "docker unavailable")

        w = max(len(t.name) for t in benchmark_tests) + 2
        dw = max(len(t.description) for t in benchmark_tests) + 2
        for t in benchmark_tests:
            if t.image:
                ok, msg = image_status[t.image]
                status = f"[green]\\[{msg}][/green]" if ok else f"[red]\\[{msg}][/red]"
            else:
                status = "[red]\\[no image][/red]"
            console.print(f"  {t.name:<{w}s}{t.description:<{dw}s}{status}")

        # Summary
        pulled = sum(1 for ok, _ in image_status.values() if ok)

    console.print(
        f"\nPrerequisites: {_prereq_str(uv_ok, 'uv', uv_msg)}  |  {_prereq_str(docker_ok, 'docker', docker_msg)}"
    )
    if benchmark_tests:
        console.print(f"  {pulled} of {len(image_status)} unique Docker images pulled")
    console.print()


def print_report(results: list[SmokeResult]) -> None:
    """Print execution report with pass/fail/skip counts."""
    console.print("\n[bold]vla-eval smoke tests[/bold]")
    console.print("=" * 40)

    current_cat = ""
    for r in results:
        if r.test.category != current_cat:
            current_cat = r.test.category
            console.print(f"\n[bold]{current_cat.upper()}[/bold]")
        sym = _SYM.get(r.status, "?")
        name = r.test.name
        dur = f"{r.duration:.1f}s" if r.duration > 0 else ""
        console.print(f"  {sym} {name:<24s}{r.message:<44s}{dur:>8s}")

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    total_time = sum(r.duration for r in results)

    console.print(f"\n{'=' * 40}")
    console.print(
        f"Results: [green]{passed} passed[/green], [red]{failed} failed[/red], "
        f"[yellow]{skipped} skipped[/yellow]    total: {total_time:.1f}s\n"
    )

    if failed > 0:
        sys.exit(1)
