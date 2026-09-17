"""CLI entry point for vla-evaluation-harness."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


from vla_eval import watchdog
from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.cli._docker import (
    RUNTIMES,
    resolve_runtime as _resolve_container_runtime,
    inside_docker as _inside_docker,
    run_in_container as _run_in_container,
)
from vla_eval.cli.config_loader import load_config as _load_config
from vla_eval.config import DockerConfig
from vla_eval.orchestrator import Orchestrator
from vla_eval.render import (
    RENDER_MODES,
    check_run_render_support as _check_render_support,
    resolve_run_render_mode as _resolve_render_mode,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("vla_eval").setLevel(level)


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


class _RecordVideoAction(argparse.Action):
    """Python 3.8-compatible boolean optional action for --record-video."""

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, option_string != "--no-record-video")


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

    # Render backend: resolved after --gpus so an explicit device spec counts as deliberate.
    try:
        render_mode = _resolve_render_mode(config, getattr(args, "render", None), cli_gpus)
        _check_render_support(config, render_mode)
    except ValueError as exc:
        _stderr_console().print(f"[red]ERROR: {exc}[/red]")
        sys.exit(1)

    # Decide whether to run via Docker
    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    use_docker = bool(docker_cfg.image) and not getattr(args, "no_docker", False) and not _inside_docker()

    if use_docker:
        try:
            runtime = _resolve_container_runtime(config, getattr(args, "runtime", None))
        except ValueError as exc:
            _stderr_console().print(f"[red]ERROR: {exc}[/red]")
            sys.exit(1)
        rc = _run_in_container(
            config,
            runtime=runtime,
            auto_yes=getattr(args, "yes", False),
            dev=getattr(args, "dev", False),
            shard_id=shard_id,
            num_shards=num_shards,
            accept_license=getattr(args, "accept_license", None),
            eval_id=eval_id,
            no_save=no_save,
            force_build=getattr(args, "build", False),
        )
        if rc != 0:
            sys.exit(rc)
        return

    import anyio

    # Pin output_dir absolute before benchmark code runs: some benchmarks (robotwin,
    # robodojo) chdir into their sim checkout, silently re-anchoring relative paths.
    config["output_dir"] = str(Path(config.get("output_dir") or "./results").resolve())

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
        errs = r.get("num_errors", 0)
        tail = f"  (⚠ {errs} episodes errored)" if errs else ""
        print(f"\n{r['benchmark']}: {r.get('mean_success', 0.0):.1%}{tail}")

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

    - ``--config / -c <yaml>`` or ``--output-dir <dir>``: the YAML supplies
      ``output_dir`` (``--output-dir`` overrides it, and stands alone when
      there is no config to hand). ``--eval-id <id>`` then picks a specific
      DB; without it every ``recording-*.sqlite`` under the directory is
      merged. The launcher script (``run_sharded.sh``) passes config + id.
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
    elif getattr(args, "config", None) or getattr(args, "output_dir", None):
        if getattr(args, "config", None):
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
            "[red]ERROR: pass --config / -c <yaml> or --output-dir <dir> "
            "(either one optionally with --eval-id), or --db <path>[/red]"
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
    from functools import partial

    from vla_eval.docker_resources import parse_gpus

    # --- resolve parallelism ---
    # Workers are sized by (and handed) GPUs — except under --render cpu, where no
    # device is attached and the GPU count would cap or serialise for no reason.
    cpu_render = getattr(args, "render", None) == "cpu"
    gpu_queue: queue.Queue[str] | None = None
    if args.parallel is not None:
        gpu_ids = [] if cpu_render else parse_gpus(None)  # auto-detect via the active GPU runtime
        if args.parallel == "auto":
            if cpu_render:
                print("--parallel auto sizes workers by GPU count; with --render cpu pass an explicit number")
                workers = 1
            else:
                workers = len(gpu_ids)
        else:
            try:
                n = int(args.parallel)
                if n <= 0:
                    raise ValueError("must be positive")
                workers = n if cpu_render else min(n, len(gpu_ids))
            except ValueError:
                print(
                    f"ERROR: --parallel must be 'auto' or a positive integer, got '{args.parallel}'", file=sys.stderr
                )
                sys.exit(1)
        if workers > 1 and not cpu_render:
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
                runner = partial(
                    run_benchmark_test, render=getattr(args, "render", None), dev=getattr(args, "dev", False)
                )
                _run_parallel(benchmark_tests, runner)
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

  render backend (render: gpu|cpu, --render):
    Run-level: the renderer binds at the first simulator import, so it cannot
    vary per benchmark entry. 'cpu' software-renders the simulator and attaches
    no GPU to the container, which keeps simulator readback off the device the
    model server computes on. Only benchmarks that declare CPU support accept
    it; the rest fail fast rather than silently falling back to the GPU.

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
    run_parser.add_argument(
        "--runtime",
        choices=RUNTIMES,
        default=None,
        help="Container runtime (default: docker.runtime, $VLA_EVAL_RUNTIME, or docker); see docs/runtimes.md",
    )
    run_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts (e.g. docker pull)")
    run_parser.add_argument(
        "--build", action="store_true", help="Rebuild docker.image from docker.build even if it exists locally"
    )
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
        "--no-record-video",
        action=_RecordVideoAction,
        default=None,
        help="Enable (or disable with --no-record-video) per-episode mp4 recording for all benchmarks.",
    )
    run_parser.add_argument(
        "--render",
        choices=RENDER_MODES,
        default=None,
        help=(
            "Where the simulator renders (overrides 'render:' in config; default: gpu). "
            "'cpu' runs the benchmark software-rendered with no GPU attached — slower, but it "
            "frees the GPU for the model server. Benchmarks that have not declared CPU support "
            "are rejected up front."
        ),
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

--config and --output-dir are interchangeable ways to name the directory holding
the DBs; pass --output-dir on its own when the config is not at hand.

examples:
  vla-eval merge -c configs/benchmarks/libero/spatial.yaml --eval-id abc
  vla-eval merge -c configs/benchmarks/libero/spatial.yaml  # merge every DB
  vla-eval merge --output-dir ./results --eval-id abc
  vla-eval merge --db /path/to/recording-abc.sqlite
""",
    )
    merge_parser.add_argument("--config", "-c", default=None, help="Config YAML (provides output_dir)")
    merge_parser.add_argument(
        "--eval-id",
        default=None,
        help="Specific eval id (= specific DB file). Omit to merge every DB under output_dir.",
    )
    merge_parser.add_argument(
        "--db",
        default=None,
        help="Direct path to a recording-*.sqlite. Bypasses --config.",
    )
    merge_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory holding the recording DBs, and where materialised files land. Usable without "
        "--config (default: config output_dir or DB parent).",
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
  vla-eval test --benchmark libero --render cpu     test it software-rendered, with no GPU attached
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
        "--render",
        choices=RENDER_MODES,
        default=None,
        help="Render backend for benchmark tests (default: the config's). 'cpu' attaches no GPU.",
    )
    test_parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Mount local src/ into the benchmark container (as in 'vla-eval run --dev'). "
            "Without this, benchmark tests run the harness baked into the image — the right "
            "thing for validating a shipped image, the wrong thing for validating local changes."
        ),
    )
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
