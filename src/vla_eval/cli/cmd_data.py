"""``vla-eval data`` subcommand: fetch externally-licensed benchmark datasets."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from vla_eval.benchmarks.base import Benchmark, DataRequirement
from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.cli._docker import check_docker_daemon, image_exists_locally
from vla_eval.cli.config_loader import load_config as _load_config
from vla_eval.config import DockerConfig
from vla_eval.docker_resources import gpu_docker_flag
from vla_eval.registry import resolve_import_string


def _resolve_benchmark_class(config: dict) -> type[Benchmark]:
    """Resolve the first benchmark entry to its class.

    Multi-benchmark configs are permitted but only the first entry's
    ``data_requirements()`` is consulted; a warning is printed so the
    user knows other entries weren't fetched.
    """
    benchmarks = config.get("benchmarks") or []
    if not benchmarks:
        raise ValueError("config has no 'benchmarks' entries")
    if len(benchmarks) > 1:
        _stderr_console().print(
            f"[yellow]warning: config has {len(benchmarks)} benchmarks; "
            "only the first entry's data_requirements() will be fetched.[/yellow]"
        )
    import_string = benchmarks[0].get("benchmark")
    if not import_string:
        raise ValueError("first benchmark entry is missing 'benchmark' import string")
    cls = resolve_import_string(import_string)
    if not (isinstance(cls, type) and issubclass(cls, Benchmark)):
        raise TypeError(f"resolved {import_string} to {cls!r}, which is not a Benchmark subclass")
    return cls


def _default_host_data_dir(cache_key: str) -> Path:
    """``${VLA_EVAL_DATA_DIR}/<cache_key>`` or the XDG-style default."""
    base = os.environ.get("VLA_EVAL_DATA_DIR")
    if base:
        return Path(base).expanduser() / cache_key
    return Path.home() / ".cache" / "vla-eval" / cache_key


def _build_docker_argv(
    docker_cfg: DockerConfig,
    host_dir: Path,
    requirement: DataRequirement,
    extra_gpus: str | None,
) -> list[str]:
    argv: list[str] = ["docker", "run", "--rm"]
    argv.extend(gpu_docker_flag(extra_gpus or docker_cfg.gpus))
    for env_pair in docker_cfg.env:
        argv.extend(["-e", env_pair])
    argv.extend(["-v", f"{host_dir}:{requirement.container_data_path}"])
    argv.append(docker_cfg.image or "")
    argv.extend(requirement.download_command)
    return argv


def cmd_data_fetch(args: argparse.Namespace) -> None:
    con = _stderr_console()
    config = _load_config(args.config)

    try:
        bench_cls = _resolve_benchmark_class(config)
    except (TypeError, ValueError) as exc:
        con.print(f"[red]ERROR: {exc}[/red]")
        sys.exit(1)

    requirement = bench_cls.data_requirements()
    if requirement is None:
        con.print(f"[yellow]{bench_cls.__name__} declares no external data requirement; nothing to fetch.[/yellow]")
        return

    if requirement.license_id not in set(args.accept_license):
        con.print(
            f"[red]ERROR: this dataset requires accepting licence '{requirement.license_id}'.[/red]\n"
            f"  Read: {requirement.license_url}\n"
            f"  Re-run: vla-eval data fetch -c {args.config} --accept-license {requirement.license_id}"
        )
        sys.exit(1)

    host_dir = _default_host_data_dir(requirement.cache_key)
    host_dir.mkdir(parents=True, exist_ok=True)

    marker = host_dir / requirement.marker
    if marker.exists() and not args.force:
        con.print(
            f"[green]Data already present at {host_dir} (marker: {requirement.marker}). "
            "Use --force to refetch.[/green]"
        )
        return

    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if not docker_cfg.image:
        con.print("[red]ERROR: 'docker.image' must be set in the config to fetch data[/red]")
        sys.exit(1)
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        con.print("[red]ERROR: 'docker' not found on PATH[/red]")
        sys.exit(1)
    check_docker_daemon(docker_bin)
    if not image_exists_locally(docker_bin, docker_cfg.image):
        # Don't try to pull — benchmarks with external data are typically
        # NO_REDIST images that aren't published.  Point at the build path.
        cache_key = requirement.cache_key
        con.print(
            f"[red]ERROR: image '{docker_cfg.image}' is not present locally.[/red]\n"
            f"  Build it first: docker/build.sh {cache_key} --accept-license {cache_key}"
        )
        sys.exit(1)

    argv = _build_docker_argv(docker_cfg, host_dir, requirement, extra_gpus=getattr(args, "gpus", None))

    con.print(f"[bold]Fetching data → {host_dir}[/bold]")
    con.print(f"  image: {docker_cfg.image}")
    con.print(f"  mount: {host_dir} → {requirement.container_data_path}")
    if args.dry_run:
        con.print("  [yellow]--dry-run[/yellow]: would run:")
        con.print(f"    {' '.join(argv)}")
        return

    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        con.print(f"[red]ERROR: docker run exited with {completed.returncode}[/red]")
        sys.exit(completed.returncode)
    con.print(f"[green]Done. Dataset available at {host_dir}.[/green]")


def register(subparsers: argparse._SubParsersAction) -> None:
    data_parser = subparsers.add_parser(
        "data",
        help="Manage external benchmark datasets",
        description=(
            "Fetch external datasets that aren't redistributable in the docker image. "
            "Each benchmark's data requirements are declared in its Benchmark class via "
            "data_requirements(); see vla_eval.benchmarks.base.DataRequirement."
        ),
    )
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    fetch_parser = data_sub.add_parser(
        "fetch",
        help="Download a benchmark's external data into the local cache",
        description=(
            "Resolves the benchmark class from the config, runs its download command "
            "inside the benchmark's docker image with the host cache mounted "
            "read-write at the container's data path. Idempotent: skips if the "
            "marker file already exists."
        ),
    )
    fetch_parser.add_argument("--config", "-c", required=True, help="Path to a benchmark eval config YAML.")
    fetch_parser.add_argument(
        "--accept-license",
        action="append",
        default=[],
        metavar="ID",
        help="License ID to opt into (e.g. 'behavior-dataset-tos'). Repeatable.",
    )
    fetch_parser.add_argument(
        "--gpus",
        default=None,
        help="GPU devices for the fetch container (e.g. '0,1'). Defaults to docker.gpus or 'all'.",
    )
    fetch_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the download even if the marker file is already present.",
    )
    fetch_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the docker command that would run and exit.",
    )
    fetch_parser.set_defaults(func=cmd_data_fetch)
