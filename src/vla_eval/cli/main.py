"""CLI entry point for vla-evaluation-harness."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess as _subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from vla_eval.config import DockerConfig
from vla_eval.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def _load_config(path: str) -> dict[str, Any]:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _inside_docker() -> bool:
    """Check if we are already running inside a Docker container."""
    return Path("/.dockerenv").exists()


def _exec_subprocess(cmd: list[str]) -> None:
    """Run a subprocess with proper cleanup on KeyboardInterrupt."""
    import anyio

    async def _run() -> int:
        result = await anyio.run_process(cmd, check=False, stdout=None, stderr=None)
        return result.returncode

    try:
        sys.exit(anyio.run(_run))
    except KeyboardInterrupt:
        sys.exit(130)


def _exec_docker(docker: str, cmd: list[str], container_name: str) -> None:
    """Run a Docker container, stopping it on exit/signal to prevent orphans."""
    import atexit
    import signal
    import subprocess

    proc = subprocess.Popen(cmd)

    def _stop_container() -> None:
        try:
            subprocess.run([docker, "stop", "-t", "10", container_name], capture_output=True, timeout=15)
        except Exception:
            pass

    atexit.register(_stop_container)

    def _handle_signal(signum: int, _frame: object) -> None:
        _stop_container()
        sys.exit(128 + signum)

    signal.signal(signal.SIGHUP, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        rc = proc.wait()
        atexit.unregister(_stop_container)
        sys.exit(rc)
    except KeyboardInterrupt:
        _stop_container()
        sys.exit(130)


def _check_docker_daemon(docker: str) -> None:
    """Verify Docker daemon is reachable."""
    import subprocess

    result = subprocess.run([docker, "info"], capture_output=True)
    if result.returncode != 0:
        print(
            "ERROR: Docker daemon is not running.\n  Start it with: sudo systemctl start docker",
            file=sys.stderr,
        )
        sys.exit(1)


def _image_exists_locally(docker: str, image: str) -> bool:
    """Check if a Docker image exists locally."""
    import subprocess

    result = subprocess.run([docker, "image", "inspect", image], capture_output=True)
    return result.returncode == 0


def _ensure_docker_image(docker: str, image: str, auto_yes: bool) -> None:
    """Ensure Docker image is available, pulling with confirmation if needed."""
    import subprocess

    if _image_exists_locally(docker, image):
        return

    print(f"\n⚠  Docker image '{image}' not found locally.", file=sys.stderr)
    print("   Benchmark images are typically large (tens of GB).", file=sys.stderr)
    print("   This may take a while and use significant disk space.\n", file=sys.stderr)

    if not auto_yes:
        if not sys.stdin.isatty():
            print(
                "ERROR: Cannot confirm in non-interactive mode. Use --yes to skip confirmation.",
                file=sys.stderr,
            )
            sys.exit(1)
        answer = input("Proceed with docker pull? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.", file=sys.stderr)
            sys.exit(0)

    print(f"Pulling {image} ...", file=sys.stderr)
    ret = subprocess.call([docker, "pull", image])
    if ret != 0:
        print(f"ERROR: docker pull failed (exit code {ret}).", file=sys.stderr)
        sys.exit(1)


def _run_via_docker(
    config: dict[str, Any],
    *,
    auto_yes: bool = False,
    shard_id: int | None = None,
    num_shards: int | None = None,
) -> None:
    """Execute the evaluation inside a Docker container."""
    import shutil

    docker = shutil.which("docker")
    if docker is None:
        print("ERROR: 'docker' not found. Install Docker: https://docs.docker.com/get-docker/", file=sys.stderr)
        sys.exit(1)

    _check_docker_daemon(docker)

    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if docker_cfg.image is None:
        print("ERROR: 'docker.image' must be set in config", file=sys.stderr)
        sys.exit(1)

    _ensure_docker_image(docker, docker_cfg.image, auto_yes)

    results_dir = str(Path(config.get("output_dir", "./results")).resolve())
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Rewrite config for Docker: output_dir must point to the container-side mount,
    # not the host absolute path which doesn't exist inside the container.
    import tempfile

    docker_config = dict(config)
    docker_config["output_dir"] = "/workspace/results"
    docker_config_fd, docker_config_path = tempfile.mkstemp(suffix=".yaml", prefix="vla-eval-docker-")
    try:
        with os.fdopen(docker_config_fd, "w") as f:
            yaml.safe_dump(docker_config, f)
    except Exception:
        os.close(docker_config_fd)
        raise

    container_name = f"vla-eval-{os.getpid()}"

    from vla_eval.docker_resources import gpu_docker_flag, shard_docker_flags

    # fmt: off
    cmd: list[str] = [
        docker, "run", "--rm",
        "--name", container_name,
        "--network", "host",
        "-v", f"{results_dir}:/workspace/results",
        "-v", f"{docker_config_path}:/tmp/eval_config.yaml:ro",
    ]
    # fmt: on

    # Extra volumes from config
    for vol in docker_cfg.volumes:
        cmd.extend(["-v", vol])

    # Extra env vars
    for env_str in docker_cfg.env:
        cmd.extend(["-e", env_str])

    # Resource allocation
    if num_shards is not None:
        assert shard_id is not None
        cmd.extend(shard_docker_flags(shard_id, num_shards, cpus=docker_cfg.cpus, gpus=docker_cfg.gpus))
    else:
        cmd.extend(gpu_docker_flag(docker_cfg.gpus))

    cmd.extend([docker_cfg.image, "run", "--no-docker", "--config", "/tmp/eval_config.yaml"])
    if shard_id is not None:
        cmd.extend(["--shard-id", str(shard_id), "--num-shards", str(num_shards)])

    logger.info("Running via Docker: %s", " ".join(cmd))
    try:
        _exec_docker(docker, cmd, container_name)
    finally:
        Path(docker_config_path).unlink(missing_ok=True)


def cmd_run(args: argparse.Namespace) -> None:
    """Run evaluation."""
    config = _load_config(args.config)

    shard_id = getattr(args, "shard_id", None)
    num_shards = getattr(args, "num_shards", None)

    # Validate shard args
    if (shard_id is None) != (num_shards is None):
        print("ERROR: --shard-id and --num-shards must be used together", file=sys.stderr)
        sys.exit(1)
    if num_shards is not None:
        if num_shards < 1:
            print("ERROR: --num-shards must be >= 1", file=sys.stderr)
            sys.exit(1)
        assert shard_id is not None
        if shard_id < 0 or shard_id >= num_shards:
            print(f"ERROR: --shard-id must be in [0, {num_shards})", file=sys.stderr)
            sys.exit(1)

    # CLI overrides for docker resource allocation
    cli_gpus = getattr(args, "gpus", None)
    cli_cpus = getattr(args, "cpus", None)
    if cli_gpus is not None or cli_cpus is not None:
        docker_section = config.setdefault("docker", {})
        if cli_gpus is not None:
            docker_section["gpus"] = cli_gpus
        if cli_cpus is not None:
            docker_section["cpus"] = cli_cpus

    # Decide whether to run via Docker
    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    use_docker = bool(docker_cfg.image) and not getattr(args, "no_docker", False) and not _inside_docker()

    if use_docker:
        _run_via_docker(
            config,
            auto_yes=getattr(args, "yes", False),
            shard_id=shard_id,
            num_shards=num_shards,
        )
        return

    import anyio

    orchestrator = Orchestrator(config, shard_id=shard_id, num_shards=num_shards)
    results = anyio.run(orchestrator.run)

    # Print final summary
    for r in results:
        print(f"\n{r['benchmark']}: {r['overall_success_rate']:.1%}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch a model server from a YAML config via uv run."""
    import shutil

    uv = shutil.which("uv")
    if uv is None:
        print("ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    script = Path(config["script"]).resolve()
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    cmd: list[str] = [uv, "run", str(script)]
    for key, value in config.get("args", {}).items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    logger.info("Running: %s", " ".join(cmd))
    _exec_subprocess(cmd)


def _discover_shard_groups(config_path: str) -> dict[str, list[Path]]:
    """Auto-discover shard files from a config YAML, grouped by benchmark name.

    Returns a dict mapping ``safe_name`` to its shard file paths.
    """
    import re

    from vla_eval.config import EvalConfig

    config = _load_config(config_path)
    output_dir = Path(config.get("output_dir", "./results"))

    groups: dict[str, list[Path]] = {}
    for bench_cfg in config.get("benchmarks", []):
        cfg = EvalConfig.from_dict(bench_cfg)
        safe_name = re.sub(r"[^\w\-.]", "_", cfg.resolved_name())
        if safe_name in groups:
            continue
        matched = sorted(output_dir.glob(f"{safe_name}_shard*of*.json"))
        if not matched:
            print(f"WARNING: no shard files found for {safe_name} in {output_dir}", file=sys.stderr)
        groups[safe_name] = matched
    return groups


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge shard result files."""
    import glob
    import json

    from vla_eval.results.merge import load_shard_files, merge_shards, print_merge_report

    if not args.files and not args.config:
        print("ERROR: provide shard files or --config/-c to auto-discover", file=sys.stderr)
        sys.exit(1)

    # When --config is given, merge each sub-benchmark separately.
    if args.config:
        groups = _discover_shard_groups(args.config)
        # Also include any explicitly passed files as an extra group
        if args.files:
            extra: list[Path] = []
            for pattern in args.files:
                extra.extend(Path(p) for p in sorted(glob.glob(pattern)))
            if extra:
                groups["_extra"] = extra

        if not any(groups.values()):
            print("ERROR: no shard files found", file=sys.stderr)
            sys.exit(1)

        output_base = Path(args.output) if args.output else None
        merged_count = 0
        for name, paths in groups.items():
            if not paths:
                continue
            try:
                shards = load_shard_files(paths)
                merged = merge_shards(shards)
            except ValueError as e:
                print(f"ERROR ({name}): {e}", file=sys.stderr)
                sys.exit(1)
            print_merge_report(merged)
            if output_base:
                if len(groups) == 1:
                    out = output_base
                else:
                    out = output_base.parent / f"{output_base.stem}_{name}{output_base.suffix}"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(merged, indent=2, default=str))
                print(f"Merged result saved to {out}", file=sys.stderr)
            else:
                print(json.dumps(merged, indent=2, default=str))
            merged_count += 1

        if merged_count == 0:
            print("ERROR: no shard files found", file=sys.stderr)
            sys.exit(1)
        return

    # Legacy path: positional file args only
    paths: list[Path] = []
    for pattern in args.files:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"WARNING: no files matched: {pattern}", file=sys.stderr)
        paths.extend(Path(p) for p in matched)

    if not paths:
        print("ERROR: no shard files found", file=sys.stderr)
        sys.exit(1)

    try:
        shards = load_shard_files(paths)
        merged = merge_shards(shards)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print_merge_report(merged)

    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(merged, indent=2, default=str))
        print(f"Merged result saved to {output}", file=sys.stderr)
    else:
        print(json.dumps(merged, indent=2, default=str))


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate config file."""
    config = _load_config(args.config)
    from vla_eval.benchmarks.base import Benchmark
    from vla_eval.registry import resolve_import_string

    errors = []
    for bench in config.get("benchmarks", []):
        import_path = bench.get("benchmark", "")
        if not import_path or ":" not in import_path:
            errors.append(f"Missing or invalid 'benchmark' import path: {import_path!r}")
            continue
        try:
            cls = resolve_import_string(import_path)
            if not (isinstance(cls, type) and issubclass(cls, Benchmark)):
                errors.append(f"{import_path!r} is not a Benchmark subclass")
        except Exception as e:
            errors.append(f"Cannot resolve {import_path!r}: {e}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Config is valid.")


def cmd_test_benchmark(args: argparse.Namespace) -> None:
    """Smoke-test a benchmark: start EchoModelServer on host, run benchmark via Docker for 1 episode."""
    import shutil
    import socket
    import subprocess
    import tempfile

    from vla_eval.model_servers.predict import PredictModelServer
    from vla_eval.model_servers.serve import serve_async

    config = _load_config(args.config)

    # Require docker config
    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if not docker_cfg.image:
        print("ERROR: Config has no docker.image — test-benchmark requires a Docker benchmark.", file=sys.stderr)
        sys.exit(1)

    docker = shutil.which("docker")
    if docker is None:
        print("ERROR: 'docker' not found. Install Docker: https://docs.docker.com/get-docker/", file=sys.stderr)
        sys.exit(1)

    _check_docker_daemon(docker)
    _ensure_docker_image(docker, docker_cfg.image, auto_yes=getattr(args, "yes", False))

    # Pick a free port for the echo server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]

    # Write a temporary config with 1 task, 1 episode, pointing to our echo server
    smoke_config = dict(config)
    # --network host: container shares host network, so 127.0.0.1 works
    smoke_config["server"] = {"url": f"ws://127.0.0.1:{port}"}
    smoke_config.pop("docker", None)
    for bench in smoke_config.get("benchmarks", []):
        bench["episodes_per_task"] = 1
        bench["max_tasks"] = 1

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(smoke_config, tmp)
    tmp.close()

    # Infer action_dim from benchmark config (default 7)
    action_dim = 7
    for bench in smoke_config.get("benchmarks", []):
        action_dim = bench.get("action_dim", action_dim)

    class _EchoModelServer(PredictModelServer):
        def predict(self, obs, ctx):
            import numpy as np

            return {"actions": np.zeros(action_dim, dtype=np.float32)}

    # Suppress websocket noise from the echo server (handshake errors from TCP
    # readiness probes, connection lifecycle messages).  The echo server is a
    # short-lived test helper, so its websocket logs are never useful.
    import time

    logging.getLogger("websockets").setLevel(logging.CRITICAL)

    # Start echo server in a daemon thread so it dies automatically when
    # the main thread exits — no portal cleanup needed.
    import asyncio
    import threading

    def _run_echo_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve_async(_EchoModelServer(), host="0.0.0.0", port=port))

    server_thread = threading.Thread(target=_run_echo_server, daemon=True)
    server_thread.start()

    # Wait for echo server to be ready
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

    cmd: list[str] = [
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
        f"{tmp.name}:/tmp/eval_config.yaml:ro",
    ]
    cmd.extend(gpu_docker_flag(docker_cfg.gpus))
    for vol in docker_cfg.volumes:
        cmd.extend(["-v", vol])
    for env_str in docker_cfg.env:
        cmd.extend(["-e", env_str])
    cmd.extend([docker_cfg.image, "run", "--no-docker", "--config", "/tmp/eval_config.yaml"])

    print(f"Starting echo server on port {port}")
    print(f"Running: {' '.join(cmd)}")

    try:
        rc = subprocess.call(cmd)
    finally:
        os.unlink(tmp.name)

    if rc != 0:
        print(f"❌ Benchmark test failed (exit code {rc})", file=sys.stderr)
        sys.exit(1)

    # Check results
    import glob
    import json

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if json_files:
        result = json.loads(Path(json_files[0]).read_text())
        rate = result.get("overall_success_rate", 0)
        print(f"✅ Benchmark test passed: success_rate={rate:.0%}")
    else:
        print("✅ Benchmark test completed (no result file — check benchmark output above)")


def cmd_test(args: argparse.Namespace) -> None:
    """Run smoke tests across CLI commands."""
    from vla_eval.cli.smoke import (
        discover_benchmark_tests,
        discover_server_tests,
        discover_validate_tests,
        print_list,
        print_report,
        run_benchmark_test,
        run_server_test,
        run_validate,
    )

    # Which categories to run
    run_all = not args.validate_only and args.server is None and args.benchmark is None
    do_validate = run_all or args.validate_only
    do_server = run_all or args.server is not None
    do_benchmark = run_all or args.benchmark is not None

    # Discover tests
    validate_tests = discover_validate_tests() if do_validate else []
    server_tests = discover_server_tests(args.server) if do_server else []
    benchmark_tests = discover_benchmark_tests(args.benchmark) if do_benchmark else []

    if args.list or args.dry_run:
        print_list(validate_tests, server_tests, benchmark_tests)
        if args.dry_run and not args.list:
            # Show what would run (vs --list which shows all regardless)
            total = len(validate_tests) + len(server_tests) + len(benchmark_tests)
            print(f"Would run {total} test(s). Use without --dry-run to execute.")
        return

    results = []

    if validate_tests:
        results.append(run_validate(validate_tests))

    for t in server_tests:
        results.append(run_server_test(t, args.timeout))

    for t in benchmark_tests:
        results.append(run_benchmark_test(t))

    if not results:
        print("No tests to run. Use --list to see available tests.", file=sys.stderr)
        sys.exit(1)

    print_report(results)


def cmd_test_server(args: argparse.Namespace) -> None:
    """Smoke-test a model server: launch it via uv run, run StubBenchmark for 1 episode."""
    import shutil
    import socket
    import time

    from vla_eval.benchmarks.base import StepBenchmark, StepResult
    from vla_eval.types import Observation, Task
    from vla_eval.connection import Connection
    from vla_eval.runners.sync_runner import SyncEpisodeRunner

    uv = shutil.which("uv")
    if uv is None:
        print("ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    script = Path(config["script"]).resolve()
    if not script.exists():
        print(f"ERROR: Script not found: {script}", file=sys.stderr)
        sys.exit(1)

    # Pick a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # Build uv run command with --port override
    cmd: list[str] = [uv, "run", str(script)]
    for key, value in config.get("args", {}).items():
        flag = f"--{key}"
        if key == "port":
            continue  # we override port
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    cmd.extend(["--port", str(port)])

    print(f"Starting model server: {' '.join(cmd)}")

    # Extract suite from config for servers that use chunk_size_map
    import json as _json

    task: Task = {"name": "smoke_test"}
    args_cfg = config.get("args", {})
    chunk_map_raw = args_cfg.get("chunk_size_map")
    if chunk_map_raw:
        chunk_map = _json.loads(chunk_map_raw) if isinstance(chunk_map_raw, str) else chunk_map_raw
        if chunk_map:
            task["suite"] = next(iter(chunk_map.keys()))

    # Minimal benchmark that sends realistic observations
    class _StubBenchmark(StepBenchmark):
        def __init__(self):
            super().__init__()
            self._step = 0

        @staticmethod
        def _dummy_obs() -> dict[str, Any]:
            import numpy as np

            return {
                "images": {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
                "task_description": "smoke test",
            }

        def get_tasks(self):
            return [task]

        def reset(self, task: Task) -> Any:
            self._step = 0
            return None

        def step(self, action):
            self._step += 1
            done = self._step >= 3
            return StepResult(obs=None, reward=1.0 if done else 0.0, done=done, info={})

        def make_obs(self, raw_obs: Any, task: Task) -> Observation:
            return self._dummy_obs()

        def check_done(self, step_result):
            return step_result.done

        def get_step_result(self, step_result):
            return {"success": step_result.done}

        def get_metadata(self):
            return {"max_steps": 50}

    import anyio

    async def _run() -> dict:
        # Use async subprocess to avoid pipe buffer deadlock — model servers
        # can produce significant stdout/stderr during model loading.
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
            # Wait for server to be ready via TCP check (no WebSocket handshake)
            timeout = getattr(args, "timeout", 300)
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
        success = result.get("success", False)
        steps = result.get("steps", 0)
        if success:
            print(f"✅ Model server OK: {steps} steps, success={success}")
        else:
            print(f"❌ Model server FAIL: {steps} steps, success={success}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"❌ Model server test failed: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vla-eval",
        description="VLA Evaluation Harness — benchmark Vision-Language-Action models in simulation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = sub.add_parser(
        "run",
        help="Run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
execution flow:
  By default, if the config contains a 'docker.image' key, the CLI
  launches a Docker container and re-invokes itself inside it with
  --no-docker.  Use --no-docker to skip this and run directly.

  Docker container settings:
    --gpus all --network host (model server on host is reachable at localhost)
    Config file is bind-mounted read-only; results dir is bind-mounted read-write.
    Extra volumes/env vars can be added via docker.volumes and docker.env in config.

  max_steps resolution:
    If max_steps is omitted from the config, the benchmark's own default
    is used (e.g. libero_spatial=220, libero_10=520).
    Setting max_steps explicitly in config always takes precedence.

  sharding (--shard-id / --num-shards):
    Work items (task × episode pairs) are distributed round-robin across shards.
    Each shard writes a deterministic output file: {name}_shard{id}of{total}.json.
    Use 'vla-eval merge' to combine shard results.

  error recovery:
    Episodes are isolated — one failure does not abort the run.
    On server disconnect, the harness retries (5× exponential backoff)
    then continues.  Partial results are saved automatically.
""",
    )
    run_parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    run_parser.add_argument(
        "--no-docker", action="store_true", help="Run directly without Docker (for dev/debug or inside-container use)"
    )
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts (e.g. docker pull)")
    run_parser.add_argument(
        "--shard-id", type=int, default=None, help="Shard index (0-based). Must use with --num-shards."
    )
    run_parser.add_argument(
        "--num-shards", type=int, default=None, help="Total number of shards. Must use with --shard-id."
    )
    run_parser.add_argument(
        "--gpus",
        default=None,
        help="GPU devices for benchmark containers, e.g. '0,1' (overrides docker.gpus in config)",
    )
    run_parser.add_argument(
        "--cpus",
        default=None,
        help="CPU range for benchmark containers, e.g. '0-31' (overrides docker.cpus in config)",
    )
    run_parser.add_argument("--verbose", "-v", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    # serve command
    serve_parser = sub.add_parser(
        "serve",
        help="Launch model server from config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Launches a model server script via 'uv run <script>'.
Requires 'uv' (https://docs.astral.sh/uv/) on PATH.

The config YAML must contain:
  script: path/to/server_script.py   # resolved relative to cwd
  args:                               # converted to --key value flags
    model_path: Org/model-name
    port: 8000

Bool args become flags (--use_text_template), others become --key value.
""",
    )
    serve_parser.add_argument("--config", "-c", required=True, help="Path to model server YAML config")
    serve_parser.add_argument("--verbose", "-v", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    # merge command
    merge_parser = sub.add_parser(
        "merge",
        help="Merge shard result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Combines shard JSON files produced by --shard-id/--num-shards runs.

  Expects files named {name}_shard{id}of{total}.json.
  Missing shards are allowed — the merged result is marked partial.
  Duplicate episode IDs across shards: last file wins (a warning is logged).

examples:
  vla-eval merge -c configs/libero_spatial.yaml -o results/libero_spatial.json
  vla-eval merge results/LIBEROBenchmark_shard*of4.json -o merged.json
  vla-eval merge results/*.json  # merges all shard files found
""",
    )
    merge_parser.add_argument("files", nargs="*", help="Shard result JSON files (supports glob patterns)")
    merge_parser.add_argument(
        "--config", "-c", default=None, help="Config YAML — auto-discover shard files from output_dir"
    )
    merge_parser.add_argument("--output", "-o", default=None, help="Output path for merged JSON (default: stdout)")
    merge_parser.add_argument("--verbose", "-v", action="store_true")
    merge_parser.set_defaults(func=cmd_merge)

    # validate command
    val_parser = sub.add_parser(
        "validate",
        help="Validate config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Checks that all 'benchmark' import strings (module:Class format) in the
config resolve to valid Benchmark subclasses.

Note: this does NOT check whether benchmark dependencies (e.g. robosuite,
mani_skill2) are installed — only that the import path is well-formed and
the class exists in the current Python environment.
""",
    )
    val_parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    val_parser.set_defaults(func=cmd_validate)

    # test-benchmark command
    tb_parser = sub.add_parser(
        "test-benchmark",
        help="Smoke-test a benchmark (EchoModelServer + 1 episode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Verifies that a benchmark Docker image works end-to-end.

  What it does:
    1. Starts an EchoModelServer on the host (returns zero actions)
    2. Launches the benchmark Docker container (1 task, 1 episode)
    3. Reports pass/fail based on whether the episode completes

  Requires: Docker (docker.image must be set in config).
  Does NOT require: a real model or GPU inference.
""",
    )
    tb_parser.add_argument("--config", "-c", required=True, help="Path to benchmark YAML config")
    tb_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts (e.g. docker pull)")
    tb_parser.add_argument("--verbose", "-v", action="store_true")
    tb_parser.set_defaults(func=cmd_test_benchmark)

    # test-server command
    ts_parser = sub.add_parser(
        "test-server",
        help="Smoke-test a model server (StubBenchmark + 1 episode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Verifies that a model server starts and responds to observations.

  What it does:
    1. Launches the model server via 'uv run <script>'
    2. Waits for TCP readiness (up to --timeout seconds)
    3. Sends dummy observations from a StubBenchmark (3 steps)
    4. Reports pass/fail based on whether actions are received

  Requires: uv on PATH, model server config with 'script' key.
  Does NOT require: Docker or a real benchmark environment.
""",
    )
    ts_parser.add_argument("--config", "-c", required=True, help="Path to model server YAML config")
    ts_parser.add_argument("--timeout", "-t", type=int, default=300, help="Seconds to wait for server startup")
    ts_parser.add_argument("--verbose", "-v", action="store_true")
    ts_parser.set_defaults(func=cmd_test_server)

    # test command (umbrella smoke tests)
    test_parser = sub.add_parser(
        "test",
        help="Run smoke tests across all CLI commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Discovers configs, checks resource prerequisites, and runs smoke tests.

  categories:
    validate   — resolve import strings in all benchmark configs (fast, no deps)
    server     — run test-server for each model server config (needs uv + model + GPU)
    benchmark  — run test-benchmark for each benchmark config (needs Docker + image + GPU)

  By default, runs all categories. Use flags to select specific ones.

examples:
  vla-eval test --list                     show available tests + readiness
  vla-eval test --validate                 validate all benchmark configs
  vla-eval test --server cogact            test model servers matching 'cogact'
  vla-eval test --benchmark libero         test benchmarks matching 'libero'
  vla-eval test --dry-run                  preview what would run
""",
    )
    test_parser.add_argument("--list", action="store_true", help="Show available tests and prerequisites")
    test_parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    test_parser.add_argument("--validate", dest="validate_only", action="store_true", help="Validate configs only")
    test_parser.add_argument(
        "--server",
        nargs="?",
        const="*",
        default=None,
        metavar="NAME",
        help="Run server smoke tests (optional name filter, e.g. 'cogact')",
    )
    test_parser.add_argument(
        "--benchmark",
        nargs="?",
        const="*",
        default=None,
        metavar="NAME",
        help="Run benchmark smoke tests (optional name filter, e.g. 'libero')",
    )
    test_parser.add_argument("--timeout", type=int, default=300, help="Server startup timeout in seconds")
    test_parser.add_argument("--verbose", "-v", action="store_true")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
