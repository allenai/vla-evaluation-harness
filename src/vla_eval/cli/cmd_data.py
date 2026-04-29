"""``vla-eval data`` subcommand handlers.

Provides a uniform fetch flow for benchmarks whose dataset is licensed
independently of the harness (e.g. BEHAVIOR-1K's BEHAVIOR Dataset
ToS).  See :class:`vla_eval.benchmarks.base.DataRequirement` and
:meth:`vla_eval.benchmarks.base.Benchmark.data_requirements`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from vla_eval.benchmarks.base import Benchmark, DataRequirement
from vla_eval.cli.config_loader import load_config as _load_config
from vla_eval.config import DockerConfig
from vla_eval.registry import resolve_import_string


def _stderr_console():  # pragma: no cover — same shim cmd_run uses
    from rich.console import Console

    return Console(stderr=True, soft_wrap=True)


def _resolve_benchmark_class(config: dict) -> tuple[type[Benchmark], str]:
    """Return ``(class, cache_subdir)`` for the first benchmark in config.

    ``cache_subdir`` is the module-path's last package segment, e.g.
    ``vla_eval.benchmarks.behavior1k.benchmark:X`` → ``behavior1k``.
    """
    benchmarks = config.get("benchmarks") or []
    if not benchmarks:
        raise ValueError("config has no 'benchmarks' entries")
    import_string = benchmarks[0].get("benchmark")
    if not import_string:
        raise ValueError("first benchmark entry is missing 'benchmark' import string")
    cls = resolve_import_string(import_string)
    if not (isinstance(cls, type) and issubclass(cls, Benchmark)):
        raise TypeError(f"resolved {import_string} to {cls!r}, which is not a Benchmark subclass")
    module_path = import_string.split(":", 1)[0]
    parts = module_path.split(".")
    # Expect …benchmarks.<key>.benchmark — take the second-to-last part.
    cache_subdir = parts[-2] if len(parts) >= 2 else parts[-1]
    return cls, cache_subdir


def _default_host_data_dir(cache_subdir: str) -> Path:
    """Return ``${VLA_EVAL_DATA_DIR}/<cache_subdir>`` or the XDG-style default."""
    base = os.environ.get("VLA_EVAL_DATA_DIR")
    if base:
        return Path(base).expanduser() / cache_subdir
    return Path.home() / ".cache" / "vla-eval" / cache_subdir


def _build_docker_argv(
    image: str,
    docker_cfg: DockerConfig,
    host_dir: Path,
    requirement: DataRequirement,
    extra_gpus: str | None,
) -> list[str]:
    """Build the ``docker run`` argv that downloads the dataset."""
    argv: list[str] = ["docker", "run", "--rm"]
    gpus = extra_gpus or docker_cfg.gpus or "all"
    argv.extend(["--gpus", gpus])
    for env_pair in docker_cfg.env:
        argv.extend(["-e", env_pair])
    argv.extend(["-v", f"{host_dir}:{requirement.container_data_path}"])
    argv.append(image)
    argv.extend(requirement.download_command)
    return argv


def cmd_data_fetch(args: argparse.Namespace) -> None:
    """Fetch the external dataset for a benchmark, mounted at the
    canonical host cache directory."""
    con = _stderr_console()
    config = _load_config(args.config)

    try:
        bench_cls, cache_subdir = _resolve_benchmark_class(config)
    except (TypeError, ValueError) as exc:
        con.print(f"[red]ERROR: {exc}[/red]")
        sys.exit(1)

    requirement = bench_cls.data_requirements()
    if requirement is None:
        con.print(f"[yellow]{bench_cls.__name__} declares no external data requirement; nothing to fetch.[/yellow]")
        return

    accepted = set(args.accept_license or [])
    if requirement.license_id not in accepted:
        con.print(
            f"[red]ERROR: this dataset requires accepting licence '{requirement.license_id}'.[/red]\n"
            f"  Read: {requirement.license_url}\n"
            f"  Re-run: vla-eval data fetch -c {args.config} --accept-license {requirement.license_id}"
        )
        sys.exit(1)

    host_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else _default_host_data_dir(cache_subdir)
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
    if shutil.which("docker") is None:
        con.print("[red]ERROR: 'docker' not found on PATH[/red]")
        sys.exit(1)

    argv = _build_docker_argv(
        docker_cfg.image,
        docker_cfg,
        host_dir,
        requirement,
        extra_gpus=getattr(args, "gpus", None),
    )

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
    """Wire ``data fetch`` into the top-level ``vla-eval`` parser."""
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
        "--data-dir",
        default=None,
        help="Override host data directory. Defaults to "
        "${VLA_EVAL_DATA_DIR}/<benchmark> or ~/.cache/vla-eval/<benchmark>.",
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
