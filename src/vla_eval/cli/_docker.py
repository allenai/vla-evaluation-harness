"""Docker subprocess helpers."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.config import DockerConfig

logger = logging.getLogger(__name__)

# Public mirror for images the allenai org has not granted public read on (issue #73).
_REGISTRY_MIRRORS = {"ghcr.io/allenai/vla-evaluation-harness/": "ghcr.io/worv-ai/vla-evaluation-harness-public/"}


def dev_src_mount_flags() -> list[str]:
    """Volume flags mounting the host checkout's ``src/`` over the image's (``--dev``).

    Raises ``RuntimeError`` when no checkout is found; the caller decides whether that
    exits the CLI (``vla-eval run``) or fails one test (the smoke runner).
    """
    cwd_src = Path.cwd() / "src"
    if (cwd_src / "vla_eval").is_dir():
        src = cwd_src.resolve()
    else:
        # Editable install: ``vla_eval.__file__`` lives under ``src/vla_eval/``.
        import vla_eval

        pkg_parent = Path(vla_eval.__file__).resolve().parent.parent
        if pkg_parent.name != "src" or not (pkg_parent / "vla_eval").is_dir():
            raise RuntimeError("--dev: cannot find src/vla_eval/ in cwd or via editable install")
        src = pkg_parent
    return ["-v", f"{src}:/workspace/src"]


def check_docker_daemon(docker: str) -> None:
    """Exit 1 with a clear message if the docker daemon is unreachable."""
    if subprocess.run([docker, "info"], capture_output=True).returncode != 0:
        _stderr_console().print(
            "[red]ERROR: Docker daemon is not running.[/red]\n  Start it with: sudo systemctl start docker",
        )
        sys.exit(1)


def image_exists_locally(docker: str, image: str) -> bool:
    """Return True if a docker image is present in the local store."""
    return subprocess.run([docker, "image", "inspect", image], capture_output=True).returncode == 0


def ensure_image_local(docker: str, image: str, auto_yes: bool) -> None:
    """Make sure ``image`` is available locally, prompting for ``docker pull`` when missing."""
    if image_exists_locally(docker, image):
        return

    con = _stderr_console()
    con.print(f"\n[yellow]⚠  Docker image '{image}' not found locally.[/yellow]")
    con.print("   Benchmark images are typically large (tens of GB).")
    con.print("   This may take a while and use significant disk space.\n")

    if not auto_yes:
        if not sys.stdin.isatty():
            con.print("[red]ERROR: Cannot confirm in non-interactive mode. Use --yes to skip confirmation.[/red]")
            sys.exit(1)
        answer = input("Proceed with docker pull? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            con.print("Aborted.")
            sys.exit(0)

    con.print(f"Pulling {image} ...")
    if subprocess.call([docker, "pull", image]) == 0:
        return
    for primary, mirror_prefix in _REGISTRY_MIRRORS.items():
        if image.startswith(primary):
            mirror = mirror_prefix + image.removeprefix(primary)
            con.print(f"[yellow]Pull failed; retrying from public mirror: {mirror}[/yellow]")
            if subprocess.call([docker, "pull", mirror]) == 0:
                subprocess.call([docker, "tag", mirror, image])
                return
    con.print(f"[red]ERROR: docker pull failed for {image}.[/red]")
    sys.exit(1)


def inside_docker() -> bool:
    return Path("/.dockerenv").exists()


def exec_docker(docker: str, cmd: list[str], container_name: str) -> int:
    """Run a Docker container, stopping it on exit/signal to prevent orphans. Returns the exit code."""
    import atexit
    import signal
    import subprocess
    import threading

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

    # Library callers (vla_eval.api) may have their own SIGTERM handling; restore it after.
    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGHUP, signal.SIGTERM):
            previous[sig] = signal.signal(sig, _handle_signal)

    try:
        rc = proc.wait()
        atexit.unregister(_stop_container)
        return rc
    except KeyboardInterrupt:
        _stop_container()
        return 130
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_via_docker(
    config: dict[str, Any],
    *,
    auto_yes: bool = False,
    dev: bool = False,
    shard_id: int | None = None,
    num_shards: int | None = None,
    accept_license: list[str] | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
) -> int:
    """Execute the evaluation inside a Docker container. Returns the container's exit code."""
    import shutil

    docker = shutil.which("docker")
    if docker is None:
        _stderr_console().print(
            "[red]ERROR: 'docker' not found. Install Docker: https://docs.docker.com/get-docker/[/red]"
        )
        sys.exit(1)

    check_docker_daemon(docker)

    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if docker_cfg.image is None:
        _stderr_console().print("[red]ERROR: 'docker.image' must be set in config[/red]")
        sys.exit(1)

    ensure_image_local(docker, docker_cfg.image, auto_yes)

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

    # The watchdog runs inside the container; forward the host override so long
    # episodes (e.g. RoboDojo's 1900-step tasks) aren't killed as stalls.
    if os.environ.get("VLA_EVAL_WATCHDOG_TIMEOUT_S"):
        cmd.extend(["-e", f"VLA_EVAL_WATCHDOG_TIMEOUT_S={os.environ['VLA_EVAL_WATCHDOG_TIMEOUT_S']}"])

    # Forward stdin/TTY for in-container licence prompts.
    cmd.extend(tty_docker_flags())

    # Dev mode: mount host src/ into container (requires editable install in image).
    if dev:
        try:
            mount = dev_src_mount_flags()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        cmd.extend(mount)
        logger.info("Dev mode: mounting %s -> /workspace/src", mount[1].split(":", 1)[0])

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
        return exec_docker(docker, cmd, container_name)
    finally:
        Path(docker_config_path).unlink(missing_ok=True)
