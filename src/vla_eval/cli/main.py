"""CLI entry point for vla-evaluation-harness."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from vla_eval import watchdog
from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.cli._docker import (
    check_docker_daemon as _check_docker_daemon,
    ensure_image_local as _ensure_docker_image,
)
from vla_eval.cli.config_loader import load_config as _load_config
from vla_eval.config import DockerConfig
from vla_eval.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("vla_eval").setLevel(level)


def _inside_docker() -> bool:
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


def _resolve_dev_src() -> Path:
    """Find the host ``src/`` directory for ``--dev`` bind-mount."""
    cwd_src = Path.cwd() / "src"
    if (cwd_src / "vla_eval").is_dir():
        return cwd_src.resolve()
    # Editable install: ``vla_eval.__file__`` lives under ``src/vla_eval/``.
    import vla_eval

    pkg_parent = Path(vla_eval.__file__).resolve().parent.parent
    if pkg_parent.name == "src" and (pkg_parent / "vla_eval").is_dir():
        return pkg_parent

    print("ERROR: --dev: cannot find src/vla_eval/ in cwd or via editable install", file=sys.stderr)
    sys.exit(1)


def _apply_record_video_override(config: dict[str, Any], *, enabled: bool) -> None:
    """Apply the run-level video override to per-benchmark recording blocks, creating them as needed."""
    for idx, bench in enumerate(config.get("benchmarks") or []):
        if not isinstance(bench, dict):
            raise ValueError(f"benchmarks[{idx}] must be a mapping")
        rec = bench.get("recording")
        if rec is None:
            rec = bench["recording"] = {}
        if not isinstance(rec, dict):
            raise ValueError(f"benchmarks[{idx}].recording must be a mapping or null")
        rec["record_video"] = enabled


def _run_via_docker(
    config: dict[str, Any],
    *,
    auto_yes: bool = False,
    dev: bool = False,
    shard_id: int | None = None,
    num_shards: int | None = None,
    accept_license: list[str] | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
) -> None:
    """Execute the evaluation inside a Docker container."""
    import shutil

    docker = shutil.which("docker")
    if docker is None:
        _stderr_console().print(
            "[red]ERROR: 'docker' not found. Install Docker: https://docs.docker.com/get-docker/[/red]"
        )
        sys.exit(1)

    _check_docker_daemon(docker)

    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if docker_cfg.image is None:
        _stderr_console().print("[red]ERROR: 'docker.image' must be set in config[/red]")
        sys.exit(1)

    _ensure_docker_image(docker, docker_cfg.image, auto_yes)

    results_dir = str(Path(config.get("output_dir", "./results")).resolve())
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # output_dir must point to the container mount; the host absolute path doesn't exist inside.
    import tempfile

    docker_config = dict(config)
    docker_config["output_dir"] = "/workspace/results"
    # Also remap any per-benchmark `recording.output_dir` that points under the
    # host results_dir — otherwise the recorder writes mp4/jsonl inside the
    # container at the host path and they vanish when the container exits.
    benchmarks = docker_config.get("benchmarks") or []
    remapped_benchmarks = []
    for entry in benchmarks:
        rec = (entry or {}).get("recording")
        if isinstance(rec, dict) and rec.get("output_dir"):
            host_path = Path(rec["output_dir"]).resolve()
            try:
                rel = host_path.relative_to(results_dir)
                new_entry = dict(entry)
                new_rec = dict(rec)
                new_rec["output_dir"] = str(Path("/workspace/results") / rel)
                new_entry["recording"] = new_rec
                remapped_benchmarks.append(new_entry)
                continue
            except ValueError:
                logger.warning(
                    "recording.output_dir=%s is outside output_dir=%s; container writes will not persist on the host",
                    host_path,
                    results_dir,
                )
        remapped_benchmarks.append(entry)
    docker_config["benchmarks"] = remapped_benchmarks
    docker_config_fd, docker_config_path = tempfile.mkstemp(suffix=".yaml", prefix="vla-eval-docker-")
    try:
        with os.fdopen(docker_config_fd, "w") as f:
            yaml.safe_dump(docker_config, f)
    except Exception:
        os.close(docker_config_fd)
        raise

    container_name = f"vla-eval-{os.getpid()}"

    from vla_eval.docker_resources import gpu_docker_flag, shard_docker_flags, tty_docker_flags

    # fmt: off
    cmd: list[str] = [
        docker, "run", "--rm",
        "--name", container_name,
        "--network", "host",
        "-v", f"{results_dir}:/workspace/results",
        "-v", f"{docker_config_path}:/tmp/eval_config.yaml:ro",
    ]
    # fmt: on

    # Opt-in --user (see DockerConfig.user).
    if docker_cfg.user == "host":
        if not hasattr(os, "getuid"):
            _stderr_console().print(
                "[red]ERROR: docker.user='host' needs a POSIX host; pin user: '<uid>:<gid>' instead.[/red]"
            )
            sys.exit(1)
        cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    elif docker_cfg.user:
        cmd.extend(["--user", docker_cfg.user])

    # Forward host-side results_dir for recorder._host_translate.
    cmd.extend(["-e", f"VLA_EVAL_HOST_OUTPUT_DIR={results_dir}"])

    # Forward stdin/TTY for in-container licence prompts.
    cmd.extend(tty_docker_flags())

    # Dev mode: mount host src/ into container (requires editable install in image).
    if dev:
        src_dir = _resolve_dev_src()
        cmd.extend(["-v", f"{src_dir}:/workspace/src"])
        logger.info("Dev mode: mounting %s -> /workspace/src", src_dir)

    # Extra volumes / env vars from config
    for vol in docker_cfg.volumes:
        cmd.extend(["-v", vol])
    for env_str in docker_cfg.env:
        cmd.extend(["-e", env_str])

    # Forward licence acceptance into the container so ``ensure_license`` can skip the prompt.
    if accept_license:
        cmd.extend(["-e", f"VLA_EVAL_ACCEPTED_LICENSES={','.join(accept_license)}"])

    # Resource allocation
    if num_shards is not None:
        assert shard_id is not None
        cmd.extend(shard_docker_flags(shard_id, num_shards, cpus=docker_cfg.cpus, gpus=docker_cfg.gpus))
    else:
        cmd.extend(gpu_docker_flag(docker_cfg.gpus))

    cmd.extend([docker_cfg.image, "run", "--no-docker", "--config", "/tmp/eval_config.yaml"])
    if shard_id is not None:
        cmd.extend(["--shard-id", str(shard_id), "--num-shards", str(num_shards)])
    if eval_id:
        cmd.extend(["--eval-id", eval_id])
    if no_save:
        cmd.append("--no-save")

    logger.info("Running via Docker: %s", " ".join(cmd))
    try:
        _exec_docker(docker, cmd, container_name)
    finally:
        Path(docker_config_path).unlink(missing_ok=True)


def cmd_run(args: argparse.Namespace) -> None:
    """Run evaluation."""
    config = _load_config(args.config)

    # CLI override for server URL
    server_url = getattr(args, "server_url", None)
    if server_url is not None:
        config.setdefault("server", {})["url"] = server_url

    # CLI override for output directory
    output_dir = getattr(args, "output_dir", None)
    if output_dir is not None:
        config["output_dir"] = output_dir

    # CLI overrides for benchmark params (applied to all benchmark entries)
    param_overrides = getattr(args, "param", None)
    if param_overrides:
        from omegaconf import OmegaConf

        overrides = OmegaConf.to_container(OmegaConf.from_dotlist(param_overrides))
        for bench in config.get("benchmarks", []):
            bench.setdefault("params", {}).update(overrides)

    shard_id = getattr(args, "shard_id", None)
    num_shards = getattr(args, "num_shards", None)
    eval_id = getattr(args, "eval_id", None)
    no_save = getattr(args, "no_save", False)

    record_video_override = getattr(args, "record_video", None)
    if record_video_override and no_save:
        _stderr_console().print("[red]ERROR: --record-video cannot be used with --no-save[/red]")
        sys.exit(1)
    if record_video_override is not None:
        try:
            _apply_record_video_override(config, enabled=record_video_override)
        except ValueError as exc:
            _stderr_console().print(f"[red]ERROR: {exc}[/red]")
            sys.exit(1)

    # Validate shard args
    if (shard_id is None) != (num_shards is None):
        _stderr_console().print("[red]ERROR: --shard-id and --num-shards must be used together[/red]")
        sys.exit(1)
    if num_shards is not None:
        if num_shards < 1:
            _stderr_console().print("[red]ERROR: --num-shards must be >= 1[/red]")
            sys.exit(1)
        assert shard_id is not None
        if shard_id < 0 or shard_id >= num_shards:
            _stderr_console().print(f"[red]ERROR: --shard-id must be in [0, {num_shards})[/red]")
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
            dev=getattr(args, "dev", False),
            shard_id=shard_id,
            num_shards=num_shards,
            accept_license=getattr(args, "accept_license", None),
            eval_id=eval_id,
            no_save=no_save,
        )
        return

    import anyio

    watchdog.start(float(os.environ.get("VLA_EVAL_WATCHDOG_TIMEOUT_S", "1200")))
    orchestrator = Orchestrator(
        config,
        shard_id=shard_id,
        num_shards=num_shards,
        eval_id=eval_id,
        no_save=no_save,
    )
    results = anyio.run(orchestrator.run)

    # Print final summary
    for r in results:
        print(f"\n{r['benchmark']}: {r.get('mean_success', 0.0):.1%}")

    # Single-shard runs auto-merge: write per-episode jsonl + aggregate JSON from
    # the SQLite recording, since there are no other shard processes to coordinate
    # with. Sharded runs leave the merge to the launcher (run_sharded.sh) so it
    # only runs once after all shards exit.
    if not no_save and shard_id is None:
        from vla_eval.results.merge import merge_eval

        output_dir = Path(config.get("output_dir", "./results")).resolve()
        try:
            merge_eval(output_dir, orchestrator.eval_id)
        except FileNotFoundError:
            logger.info("No recording DB to merge")
        except Exception:
            logger.exception("vla-eval merge failed for eval_id=%s", orchestrator.eval_id)


# yaml convention puts these under ``args:`` but they belong to the WS server,
# not ModelServer.__init__; emitted at the inner root so jsonargparse routes
# them correctly.
_SERVER_LEVEL_KEYS = {"port", "host"}


def _stringify_arg(value: Any) -> str:
    """Render a yaml value as one argv token for the inner jsonargparse parser.
    list/dict round-trip as JSON literals; primitives go through ``str`` (jsonargparse
    parses ``"null"`` / ``"8000"`` back to the typed value)."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if value is None:
        return "null"
    return str(value)


def _build_serve_cmd(
    uv: str,
    script: Path,
    config: dict[str, Any],
    *,
    address: str | None = None,
    port: int | None = None,
    overrides: list[str] | None = None,
) -> list[str]:
    """Build ``uv run <script> --<server_key>=v --args.<class_key>=v ...``.

    Server-level keys (``port`` / ``host``) emit at the inner root; everything
    in the yaml's ``args:`` block emits under ``--args.*`` so jsonargparse maps
    them onto ``server_cls.__init__``.
    """
    cmd: list[str] = [uv, "run", str(script)]
    args_block = dict(config.get("args") or {})
    for k in _SERVER_LEVEL_KEYS:
        if k in args_block:
            cmd.append(f"--{k}={_stringify_arg(args_block.pop(k))}")
    for k, v in args_block.items():
        cmd.append(f"--args.{k}={_stringify_arg(v)}")
    if port is not None:
        cmd.append(f"--port={port}")
    if address:
        cmd.extend(["--address", address])
    for override in overrides or []:
        key, sep, value = override.partition("=")
        if not key or not sep:
            raise ValueError(f"--arg must be KEY=VALUE, got {override!r}")
        prefix = "" if key in _SERVER_LEVEL_KEYS else "args."
        cmd.append(f"--{prefix}{key}={value}")
    return cmd


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch a model server from a YAML config via uv run."""
    import shutil

    uv = shutil.which("uv")
    if uv is None:
        _stderr_console().print("[red]ERROR: 'uv' not found. Install it: https://docs.astral.sh/uv/[/red]")
        sys.exit(1)

    config = _load_config(args.config)
    script = Path(config["script"]).resolve()
    if not script.exists():
        _stderr_console().print(f"[red]ERROR: Script not found: {script}[/red]")
        sys.exit(1)

    try:
        cmd = _build_serve_cmd(
            uv,
            script,
            config,
            address=getattr(args, "address", None),
            overrides=getattr(args, "arg", None),
        )
    except ValueError as exc:
        _stderr_console().print(f"[red]ERROR: {exc}[/red]")
        sys.exit(1)
    logger.info("Running: %s", " ".join(cmd))
    _exec_subprocess(cmd)


def cmd_merge(args: argparse.Namespace) -> None:
    """Materialize per-episode jsonl + aggregate JSON from a recording SQLite.

    Two ways to specify which DB to merge:

    - ``--config -c <yaml>``: derive ``output_dir`` from the YAML, then
      either use ``--eval-id <id>`` to pick a specific DB or merge every
      ``recording-*.sqlite`` under it. The launcher script
      (``run_sharded.sh``) calls this with both.
    - ``--db <path>``: direct DB path. Output goes to
      ``--output-dir`` (or the DB's parent dir).
    """
    from vla_eval.results.merge import merge_db, print_merge_summary
    from vla_eval.tracking import call_each, get_reporting_trackers

    db_paths: list[Path] = []
    output_dir: Path
    config: dict[str, Any] = {}

    if getattr(args, "db", None):
        db_paths = [Path(args.db)]
        output_dir = Path(getattr(args, "output_dir", None) or db_paths[0].parent).resolve()
    elif getattr(args, "config", None):
        config = _load_config(args.config)
        output_dir = Path(getattr(args, "output_dir", None) or config.get("output_dir", "./results")).resolve()
        if getattr(args, "eval_id", None):
            from vla_eval.recording import db_path_for_eval

            db_paths = [db_path_for_eval(output_dir, args.eval_id)]
        else:
            db_paths = sorted(output_dir.glob("recording-*.sqlite"))
            if not db_paths:
                _stderr_console().print(f"[red]ERROR: no recording-*.sqlite found under {output_dir}[/red]")
                sys.exit(1)
    else:
        _stderr_console().print(
            "[red]ERROR: pass --config / -c <yaml> (optionally with --eval-id) or --db <path>[/red]"
        )
        sys.exit(1)

    # Tracker run identity needs the same eval_id the orchestrator used so
    # id+resume converges live + merge on one run. Sniff from the DB filename
    # if --eval-id wasn't passed; skip emission entirely otherwise (orphan
    # hooks would raise on backends that require init first).
    from vla_eval.recording import eval_id_from_db_path

    trackers = get_reporting_trackers((config.get("tracking") or {}).get("report_to"))
    eval_id_for_trackers = getattr(args, "eval_id", None)
    if trackers and not eval_id_for_trackers and db_paths:
        eval_id_for_trackers = eval_id_from_db_path(db_paths[0])
    if not eval_id_for_trackers:
        trackers = []
    call_each(trackers, "on_eval_begin", eval_id_for_trackers, config)

    all_aggregates: list[dict[str, Any]] = []
    for db in db_paths:
        try:
            aggs = merge_db(db, output_dir)
        except FileNotFoundError:
            _stderr_console().print(f"[yellow]WARNING: skipping missing DB {db}[/yellow]")
            continue
        except Exception as exc:
            _stderr_console().print(f"[red]ERROR merging {db}: {exc}[/red]")
            sys.exit(1)
        for agg in aggs:
            call_each(trackers, "on_benchmark_begin", agg.get("benchmark", ""), {})
            call_each(trackers, "on_benchmark_end", agg.get("benchmark", ""), agg)
        all_aggregates.extend(aggs)

    call_each(trackers, "on_eval_end", all_aggregates)
    call_each(trackers, "close")

    print_merge_summary(all_aggregates)


def cmd_test(args: argparse.Namespace) -> None:
    """Run smoke tests across CLI commands."""
    from vla_eval.cli.smoke import (
        BENCHMARK_REGISTRY,
        SERVER_REGISTRY,
        SmokeResult,
        SmokeTest,
        check_docker,
        check_uv,
        discover_benchmark_tests,
        discover_server_tests,
        discover_validate_tests,
        print_list,
        print_report,
        run_benchmark_test,
        run_server_test,
        run_validate,
        smoke_test_from_path,
    )

    # Explicit config paths via -c
    if args.config:
        validate_tests: list[SmokeTest] = []
        server_tests: list[SmokeTest] = []
        benchmark_tests: list[SmokeTest] = []
        for config_path_str in args.config:
            path = Path(config_path_str).resolve()
            if not path.exists():
                _stderr_console().print(f"[red]ERROR: config not found: {config_path_str}[/red]")
                sys.exit(1)
            try:
                t = smoke_test_from_path(path)
            except ValueError as e:
                _stderr_console().print(f"[red]ERROR: {e}[/red]")
                sys.exit(1)
            if t.category == "server":
                server_tests.append(t)
            else:
                benchmark_tests.append(t)
    else:
        # Normalize: --server/--benchmark with no value → all; None → not requested
        server_name = None if args.server is None else (args.server if args.server != "*" else None)
        benchmark_name = None if args.benchmark is None else (args.benchmark if args.benchmark != "*" else None)
        has_filter = args.all or args.validate_only or args.server is not None or args.benchmark is not None

        # --list/--dry-run always discover everything; otherwise default to validate only
        show_all = args.list or args.dry_run
        run_validate_flag = show_all or args.all or args.validate_only or not has_filter
        run_server_flag = show_all or args.all or args.server is not None
        run_benchmark_flag = show_all or args.all or args.benchmark is not None

        validate_tests = discover_validate_tests() if run_validate_flag else []

        if run_server_flag:
            if server_name and server_name not in SERVER_REGISTRY:
                names = ", ".join(SERVER_REGISTRY.keys())
                _stderr_console().print(f"[red]ERROR: unknown server '{server_name}'. Available: {names}[/red]")
                sys.exit(1)
            server_tests = discover_server_tests(name=server_name)
        else:
            server_tests = []

        if run_benchmark_flag:
            if benchmark_name and benchmark_name not in BENCHMARK_REGISTRY:
                names = ", ".join(BENCHMARK_REGISTRY.keys())
                _stderr_console().print(f"[red]ERROR: unknown benchmark '{benchmark_name}'. Available: {names}[/red]")
                sys.exit(1)
            benchmark_tests = discover_benchmark_tests(name=benchmark_name)
        else:
            benchmark_tests = []

    if args.list or args.dry_run:
        print_list(validate_tests, server_tests, benchmark_tests)
        if args.dry_run and not args.list:
            total = len(validate_tests) + len(server_tests) + len(benchmark_tests)
            print(f"Would run {total} test(s). Use without --dry-run to execute.")
        return

    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from contextlib import nullcontext

    from vla_eval.docker_resources import parse_gpus

    # --- resolve parallelism ---
    gpu_queue: queue.Queue[str] | None = None
    if args.parallel is not None:
        gpu_ids = parse_gpus(None)  # auto-detect via the active GPU runtime
        if args.parallel == "auto":
            workers = len(gpu_ids)
        else:
            try:
                n = int(args.parallel)
                if n <= 0:
                    raise ValueError("must be positive")
                workers = min(n, len(gpu_ids))
            except ValueError:
                print(
                    f"ERROR: --parallel must be 'auto' or a positive integer, got '{args.parallel}'", file=sys.stderr
                )
                sys.exit(1)
        if workers > 1:
            gpu_queue = queue.Queue()
            for gid in gpu_ids[:workers]:
                gpu_queue.put(gid)
    else:
        workers = 1

    from vla_eval.cli.smoke import REPO_ROOT as _REPO_ROOT
    from vla_eval.cli.smoke import _SYM, console

    results: list[SmokeResult] = []
    print_lock = threading.Lock() if workers > 1 else nullcontext()
    log_dir: Path | None = None

    def _ensure_log_dir() -> Path:
        """Lazily create and return the smoke-log directory."""
        nonlocal log_dir
        if log_dir is None:
            log_dir = _REPO_ROOT / "results" / "smoke-logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _record(r: SmokeResult) -> bool:
        """Record result, print progress, save log on failure."""
        sym = _SYM.get(r.status, "?")
        dur = f" ({r.duration:.1f}s)" if r.duration > 0 else ""
        log_path: Path | None = None
        if r.status == "fail" and r.stderr:
            d = _ensure_log_dir()
            log_path = d / f"{r.test.category}_{r.test.name}.log"
        with print_lock:
            results.append(r)
            console.print(f"  {sym} {r.test.category}/{r.test.name}: {r.message}{dur}")
            if log_path is not None:
                console.print(f"    [dim]\u2192 log: {log_path.relative_to(_REPO_ROOT)}[/dim]")
        # Write file outside lock to avoid blocking other threads
        if log_path is not None:
            log_path.write_text(r.stderr)
        return r.status == "fail" and args.fail_fast

    def _run_with_gpu(runner, test, timeout):
        """Acquire a GPU slot, run the test, release the slot."""
        if gpu_queue is not None:
            gid = gpu_queue.get()
            try:
                return runner(test, timeout, gpu_id=gid)
            finally:
                gpu_queue.put(gid)
        return runner(test, timeout)

    def _run_parallel(tests: list[SmokeTest], runner) -> bool:
        """Run tests in parallel via thread pool, or sequentially if workers <= 1."""
        if workers <= 1:
            for t in tests:
                r = _run_with_gpu(runner, t, args.timeout)
                if _record(r):
                    return True
            return False

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {pool.submit(_run_with_gpu, runner, t, args.timeout): t for t in tests}
            stopped = False
            for future in as_completed(futures):
                if stopped:
                    break
                r = future.result()
                if _record(r):
                    stopped = True
                    for f in futures:
                        f.cancel()
            return stopped
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    stopped = False
    try:
        # --- validate ---
        if validate_tests:
            console.print("[bold]Running validate tests...[/bold]")
            r = run_validate(validate_tests)
            stopped = _record(r)

        # --- server (prerequisite: uv) ---
        if server_tests and not stopped:
            uv_ok, uv_msg = check_uv()
            if not uv_ok:
                console.print(f"[yellow]Skipping {len(server_tests)} server test(s): {uv_msg}[/yellow]")
                for t in server_tests:
                    results.append(SmokeResult(t, "skip", uv_msg))
            else:
                par = f", {workers} parallel" if workers > 1 else ""
                console.print(f"[bold]Running {len(server_tests)} server test(s){par}...[/bold]")
                stopped = _run_parallel(server_tests, run_server_test)

        # --- benchmark (prerequisite: docker) ---
        if benchmark_tests and not stopped:
            docker_ok, docker_msg = check_docker()
            if not docker_ok:
                console.print(f"[yellow]Skipping {len(benchmark_tests)} benchmark test(s): {docker_msg}[/yellow]")
                for t in benchmark_tests:
                    results.append(SmokeResult(t, "skip", docker_msg))
            else:
                par = f", {workers} parallel" if workers > 1 else ""
                console.print(f"[bold]Running {len(benchmark_tests)} benchmark test(s){par}...[/bold]")
                _run_parallel(benchmark_tests, run_benchmark_test)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user.[/yellow]")

    if not results:
        _stderr_console().print("[red]No tests to run. Use --list to see available tests.[/red]")
        sys.exit(1)

    print_report(results)


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
    All shards share a single recording-<eval-id>.sqlite via WAL mode.
    Pass the same --eval-id to every shard, then run 'vla-eval merge' once at
    the end (scripts/run_sharded.sh does this for you).

  recording:
    By default, benchmark entries write episode results + step rows to
    <output_dir>/recording-<eval-id>.sqlite with videos off. A recording:
    block overrides those defaults per benchmark; use --record-video to
    enable per-episode mp4s for the run. Single-shard runs auto-merge.
    Use --no-save for in-memory summary only.

  error recovery:
    Episodes are isolated — one failure does not abort the run.
    On server disconnect, the harness retries (5× exponential backoff)
    then continues.  Partial results are saved automatically.
""",
    )
    run_parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    run_parser.add_argument(
        "--server-url",
        default=None,
        help="Override server URL (e.g. ws://my-host:8000). Avoids per-host config files.",
    )
    run_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: from config YAML, or ./results/)",
    )
    run_parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Override benchmark params (applied to all benchmarks). Repeatable. "
        "e.g. --param send_wrist_image=true --param send_state=true",
    )
    run_parser.add_argument(
        "--no-docker", action="store_true", help="Run directly without Docker (for dev/debug or inside-container use)"
    )
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts (e.g. docker pull)")
    run_parser.add_argument(
        "--accept-license",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Accept a benchmark licence non-interactively (repeatable). Forwarded into the eval "
            "container as VLA_EVAL_ACCEPTED_LICENSES so vla_eval.dirs.ensure_license skips the "
            "stdin prompt. Example: --accept-license behavior-dataset-tos."
        ),
    )
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
    run_parser.add_argument(
        "--dev", action="store_true", help="Mount local src/ into the container (skip image rebuild on code changes)"
    )
    run_parser.add_argument(
        "--eval-id",
        default=None,
        help=(
            "Run-level identifier shared across shards of the same evaluation. "
            "All shards with the same --eval-id write to "
            "<output_dir>/recording-<eval-id>.sqlite. Defaults to a fresh uuid; "
            "supply explicitly to fan multiple shards into one DB."
        ),
    )
    run_parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Run without writing anything to disk: no SQLite recording, no per-episode "
            "mp4/jsonl, no aggregate JSON. The eval still executes and prints its "
            "summary to stdout. Use for quick local checks; omit for persisted results."
        ),
    )
    run_parser.add_argument(
        "--record-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable (or disable with --no-record-video) per-episode mp4 recording for all benchmarks.",
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
  args:                               # passed to the server class __init__
    model_path: Org/model-name
    port: 8000

The yaml is handed to the inner script via --config; jsonargparse maps
args.* onto the class signature with full type validation. CLI overrides
--arg KEY=VALUE compose with the yaml; KEY may be a class kwarg or a
server-level key (host/port).
""",
    )
    serve_parser.add_argument("--config", "-c", required=True, help="Path to model server YAML config")
    serve_parser.add_argument("--address", default=None, help="Override host:port (e.g. 0.0.0.0:8001)")
    serve_parser.add_argument(
        "--arg",
        action="append",
        metavar="KEY=VALUE",
        help="Override model server args (applied on top of config). Repeatable. "
        "e.g. --arg inference_delay=0.1 --arg ci=true",
    )
    serve_parser.add_argument("--verbose", "-v", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    # merge command
    merge_parser = sub.add_parser(
        "merge",
        help="Materialize per-episode jsonl + aggregate JSON from a recording SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Reads <output_dir>/recording-<eval-id>.sqlite written by `vla-eval run` and
emits the human-readable per-episode jsonl + per-benchmark aggregate JSON.

Multi-shard runs all write to one DB (same --eval-id). Run merge once after
all shards exit (run_sharded.sh does this automatically). Single-shard `vla-eval
run` invokes merge inline at the end, so manual merge is only needed for sharded
runs or to re-render outputs.

examples:
  vla-eval merge -c configs/benchmarks/libero/spatial.yaml --eval-id abc
  vla-eval merge -c configs/benchmarks/libero/spatial.yaml  # merge every DB
  vla-eval merge --db /path/to/recording-abc.sqlite
""",
    )
    merge_parser.add_argument("--config", "-c", default=None, help="Config YAML (provides output_dir)")
    merge_parser.add_argument(
        "--eval-id",
        default=None,
        help="Specific eval id (= specific DB file). Omit with --config to merge every DB under output_dir.",
    )
    merge_parser.add_argument(
        "--db",
        default=None,
        help="Direct path to a recording-*.sqlite. Bypasses --config.",
    )
    merge_parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the directory the materialised files land in (default: config output_dir or DB parent).",
    )
    merge_parser.add_argument("--verbose", "-v", action="store_true")
    merge_parser.set_defaults(func=cmd_merge)

    # test command
    test_parser = sub.add_parser(
        "test",
        help="Run smoke tests (validate configs, test servers, test benchmarks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Discovers configs, checks resource prerequisites, and runs smoke tests.

  categories:
    validate   — resolve import strings in all benchmark configs (fast, no deps)
    server     — launch model server, send dummy observations, check actions
                 (needs uv + model weights + GPU)
    benchmark  — start EchoModelServer, run benchmark in Docker for 1 episode
                 (needs Docker + image + GPU)

  By default, runs only fast validation. Use --all for everything, or
  --server / --benchmark to select expensive categories explicitly.
  Use -c to test specific config files (auto-detects server vs benchmark).

examples:
  vla-eval test                                     validate configs (fast, default)
  vla-eval test --all                               run all categories
  vla-eval test --all -x                            run all, stop at first failure
  vla-eval test --server --parallel                 test servers in parallel (one per GPU)
  vla-eval test --server --parallel 2               test servers, max 2 at a time
  vla-eval test --list                              show available tests
  vla-eval test --server                            test all model servers
  vla-eval test --server cogact                     test a specific server by registry name
  vla-eval test --benchmark libero                  test a specific benchmark by registry name
  vla-eval test -c configs/model_servers/cogact.yaml   test an arbitrary config file
  vla-eval test --dry-run                           preview what would run
""",
    )
    test_parser.add_argument(
        "-c", "--config", action="append", default=None, metavar="PATH", help="Config YAML path(s) to test"
    )
    test_parser.add_argument("--list", action="store_true", help="Show available tests and prerequisites")
    test_parser.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    test_parser.add_argument("--all", action="store_true", help="Run all categories (validate + server + benchmark)")
    test_parser.add_argument("--validate", dest="validate_only", action="store_true", help="Validate configs only")
    test_parser.add_argument(
        "--server", nargs="?", const="*", default=None, metavar="NAME", help="Server tests (exact registry name)"
    )
    test_parser.add_argument(
        "--benchmark", nargs="?", const="*", default=None, metavar="NAME", help="Benchmark tests (exact registry name)"
    )
    test_parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for server/benchmark tests")
    test_parser.add_argument(
        "--parallel",
        nargs="?",
        const="auto",
        default=None,
        metavar="N",
        help="Run server/benchmark tests in parallel (default: one per GPU, auto-detected)",
    )
    test_parser.add_argument("-x", "--fail-fast", action="store_true", help="Stop at first failure")
    test_parser.add_argument("--verbose", "-v", action="store_true")
    test_parser.set_defaults(func=cmd_test)

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
