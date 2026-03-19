"""Smoke test infrastructure for vla-eval CLI commands.

Discovers configs, checks resource prerequisites, runs tests, and reports results.
All test logic (validate, server, benchmark) is self-contained here — no subprocess
delegation to other CLI subcommands.
"""

from __future__ import annotations

import asyncio
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

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SERVER_CONFIGS_DIR = CONFIGS_DIR / "model_servers"

logger = logging.getLogger(__name__)


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


def classify_config(path: Path) -> str:
    """Classify a config file as 'server', 'benchmark', or 'unknown'."""
    data = _load_yaml(path)
    if "script" in data:
        return "server"
    if (data.get("docker") or {}).get("image"):
        return "benchmark"
    if isinstance(data.get("benchmarks"), list):
        return "benchmark"
    return "unknown"


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


def discover_server_tests() -> list[SmokeTest]:
    """Find all model server configs in configs/model_servers/."""
    tests: list[SmokeTest] = []
    for path in sorted(SERVER_CONFIGS_DIR.glob("*.yaml")):
        data = _load_yaml(path)
        model = _extract_model_id(data)
        tests.append(SmokeTest("server", path.stem, path, model))
    return tests


def discover_benchmark_tests() -> list[SmokeTest]:
    """Find all benchmark configs that have a docker image."""
    tests: list[SmokeTest] = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        data = _load_yaml(path)
        image = (data.get("docker") or {}).get("image")
        if not image:
            continue
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
# Execution — server smoke test
# ---------------------------------------------------------------------------


def run_server_test(test: SmokeTest, timeout: int) -> SmokeResult:
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

    # Pick a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

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
    import json as _json

    from vla_eval.types import Task

    task: Task = {"name": "smoke_test"}
    args_cfg = config.get("args", {})
    chunk_map_raw = args_cfg.get("chunk_size_map")
    if chunk_map_raw:
        chunk_map = _json.loads(chunk_map_raw) if isinstance(chunk_map_raw, str) else chunk_map_raw
        if chunk_map:
            task["suite"] = next(iter(chunk_map.keys()))

    # Inline StubBenchmark
    import numpy as np

    from vla_eval.benchmarks.base import StepBenchmark, StepResult
    from vla_eval.types import Observation

    class _StubBenchmark(StepBenchmark):
        def __init__(self) -> None:
            super().__init__()
            self._step = 0

        @staticmethod
        def _dummy_obs() -> dict[str, Any]:
            return {
                "images": {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
                "task_description": "smoke test",
            }

        def get_tasks(self) -> list[dict[str, Any]]:
            return [task]

        def reset(self, task_: Task) -> Any:
            self._step = 0
            return None

        def step(self, action: Any) -> StepResult:
            self._step += 1
            done = self._step >= 3
            return StepResult(obs=None, reward=1.0 if done else 0.0, done=done, info={})

        def make_obs(self, raw_obs: Any, task_: Task) -> Observation:
            return self._dummy_obs()

        def check_done(self, step_result: StepResult) -> bool:
            return step_result.done

        def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
            return {"success": step_result.done}

        def get_metadata(self) -> dict[str, Any]:
            return {"max_steps": 50}

    import subprocess as _subprocess

    import anyio

    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    t0 = time.monotonic()

    async def _run() -> dict:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.PIPE,
        )
        stderr_chunks: list[bytes] = []

        async def _drain_stderr() -> None:
            assert proc.stderr
            async for chunk in proc.stderr:
                stderr_chunks.append(chunk)

        drain_task = asyncio.create_task(_drain_stderr())

        try:
            url = f"ws://127.0.0.1:{port}"
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.returncode is not None:
                    await drain_task
                    stderr = b"".join(stderr_chunks).decode(errors="replace")
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

            benchmark = _StubBenchmark()
            runner = SyncEpisodeRunner()
            async with Connection(url) as conn:
                return await runner.run_episode(benchmark, task, conn, max_steps=50)
        finally:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                with anyio.fail_after(10):
                    await proc.wait()
            except TimeoutError:
                proc.kill()
            drain_task.cancel()

    try:
        result = anyio.run(_run)
        dt = time.monotonic() - t0
        success = result.get("success", False)
        steps = result.get("steps", 0)
        if success:
            return SmokeResult(test, "pass", f"{steps} steps, success=True", dt)
        else:
            return SmokeResult(test, "fail", f"{steps} steps, success=False", dt)
    except Exception as e:
        dt = time.monotonic() - t0
        return SmokeResult(test, "fail", str(e), dt)


# ---------------------------------------------------------------------------
# Execution — benchmark smoke test
# ---------------------------------------------------------------------------


def run_benchmark_test(test: SmokeTest, timeout: int = 600) -> SmokeResult:
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

    t0 = time.monotonic()

    # Pick a free port for echo server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]

    # Write temp config: 1 task, 1 episode, pointing to echo server
    smoke_config = dict(config)
    smoke_config["server"] = {"url": f"ws://127.0.0.1:{port}"}
    smoke_config.pop("docker", None)
    for bench in smoke_config.get("benchmarks", []):
        bench["episodes_per_task"] = 1
        bench["max_tasks"] = 1

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="vla-eval-smoke-")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            yaml.dump(smoke_config, f)
    except Exception:
        os.close(tmp_fd)
        raise

    # Infer action_dim (default 7)
    action_dim = 7
    for bench in smoke_config.get("benchmarks", []):
        action_dim = bench.get("action_dim", action_dim)

    import numpy as np

    from vla_eval.model_servers.predict import PredictModelServer
    from vla_eval.model_servers.serve import serve_async

    class _EchoModelServer(PredictModelServer):
        def predict(self, obs: Any, ctx: Any) -> dict[str, Any]:
            return {"actions": np.zeros(action_dim, dtype=np.float32)}

    # Suppress websocket noise
    logging.getLogger("websockets").setLevel(logging.CRITICAL)

    # Start echo server in daemon thread
    def _run_echo_server() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve_async(_EchoModelServer(), host="0.0.0.0", port=port))

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
    container_name = f"vla-eval-test-{os.getpid()}"

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
    docker_cmd.extend(gpu_docker_flag(docker_cfg.gpus))
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

    dt = time.monotonic() - t0

    if rc != 0:
        err_lines = result.stderr.strip().splitlines()
        msg = err_lines[-1] if err_lines else f"exit code {rc}"
        return SmokeResult(test, "fail", msg, dt)

    # Check results
    import glob as _glob
    import json

    json_files = _glob.glob(os.path.join(results_dir, "*.json"))
    if json_files:
        data = json.loads(Path(json_files[0]).read_text())
        rate = data.get("overall_success_rate", 0)
        return SmokeResult(test, "pass", f"success_rate={rate:.0%}", dt)
    return SmokeResult(test, "pass", "completed (no result file)", dt)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

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
