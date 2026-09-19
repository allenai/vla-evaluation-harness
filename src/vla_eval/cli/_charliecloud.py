"""Charliecloud runtime: run the benchmark image with no daemon and no root (docs/runtimes.md)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from vla_eval import dirs
from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.config import BuildConfig, DockerConfig

logger = logging.getLogger(__name__)

TOOLS = ("ch-image", "ch-run", "ch-convert", "ch-fromhost")


def find_tools() -> dict[str, str]:
    """Paths of the ch-* tools, or exit with an install hint."""
    found = {t: shutil.which(t) for t in TOOLS}
    missing = [t for t, path in found.items() if path is None]
    if missing:
        _stderr_console().print(
            f"[red]ERROR: not on PATH: {', '.join(missing)}. Install: pixi global install charliecloud (>= 0.45.1)[/red]"
        )
        sys.exit(1)
    return {t: str(path) for t, path in found.items()}


def image_dir_for(image: str, root: Path | None = None, driver: str | None = None) -> Path:
    """Export directory; GPU exports are keyed by host driver version and never modified after publish."""
    root = root or dirs.home() / "charliecloud"
    name = image.replace("/", "%").replace(":", "+")
    return root / (f"{name}+nvidia-{driver}" if driver else name)


def ensure_image_dir(
    image: str,
    *,
    auto_yes: bool,
    gpu: bool,
    tools: dict[str, str] | None = None,
    build: BuildConfig | None = None,
    force_build: bool = False,
) -> Path:
    """Pull or build, export and (when *gpu*) inject the driver on first use, under a per-directory lock."""
    import fcntl

    tools = tools or find_tools()
    img_dir = image_dir_for(image, driver=host_driver_version() if gpu else None)
    img_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{img_dir}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rebuild = force_build and build is not None
        if (img_dir / "ch" / "metadata.json").is_file() and not rebuild:
            return img_dir
        con = _stderr_console()
        if build is None and not auto_yes:
            con.print(
                f"[yellow]Charliecloud export for '{image}' not found at {img_dir}; pulling (several GB).[/yellow]"
            )
            if not sys.stdin.isatty():
                con.print("[red]ERROR: cannot confirm in non-interactive mode; use --yes.[/red]")
                sys.exit(1)
            if input("Proceed with ch-image pull? [y/N] ").strip().lower() not in ("y", "yes"):
                sys.exit(0)
        if gpu and shutil.which("nvidia-container-cli") is None:
            con.print("[red]ERROR: GPU rendering needs nvidia-container-cli on the host, or use --render cpu.[/red]")
            sys.exit(1)
        if build is None:
            pulled = _pull_with_mirror(tools["ch-image"], image)
        elif rebuild or image not in _stored_images(tools["ch-image"]):
            con.print(f"Building {image} from {build.dockerfile_path} with ch-image ...", soft_wrap=True)
            cmd = [tools["ch-image"], "build", "-t", image, "-f", build.dockerfile_path, build.context]
            pulled = image if subprocess.call(cmd) == 0 else None
        else:
            pulled = image  # already in storage; only this export variant is missing
        if pulled is None:
            con.print(f"[red]ERROR: ch-image {'build' if build else 'pull'} failed for {image}.[/red]")
            sys.exit(1)
        if rebuild:  # other cached exports of this image are now stale
            base = image_dir_for(image, img_dir.parent).name
            for d in img_dir.parent.glob(f"{base}*"):
                if d != img_dir and (d.name == base or d.name.startswith(f"{base}+nvidia-")):
                    shutil.rmtree(d, ignore_errors=True)
        tmp_dir = Path(f"{img_dir}.tmp-{os.getpid()}")  # ch-run cannot use ch-image's storage directly
        shutil.rmtree(tmp_dir, ignore_errors=True)
        ok = subprocess.call([tools["ch-convert"], "-i", "ch-image", "-o", "dir", pulled, str(tmp_dir)]) == 0
        ok = ok and (not gpu or subprocess.call([tools["ch-fromhost"], "--nvidia", str(tmp_dir)]) == 0)
        if not ok:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            con.print(f"[red]ERROR: exporting {image} failed.[/red]")
            sys.exit(1)
        shutil.rmtree(img_dir, ignore_errors=True)
        os.rename(tmp_dir, img_dir)
    return img_dir


def _stored_images(ch_image: str) -> list[str]:
    return subprocess.run([ch_image, "list"], capture_output=True, text=True).stdout.split()


def _pull_with_mirror(ch_image: str, image: str) -> str | None:
    """``ch-image pull`` with the same public-mirror fallback as the Docker path; returns the ref pulled."""
    from vla_eval.cli._docker import _REGISTRY_MIRRORS

    if subprocess.call([ch_image, "pull", image]) == 0:
        return image
    for primary, mirror_prefix in _REGISTRY_MIRRORS.items():
        if image.startswith(primary):
            mirror = mirror_prefix + image.removeprefix(primary)
            _stderr_console().print(f"[yellow]Pull failed; retrying from public mirror: {mirror}[/yellow]")
            if subprocess.call([ch_image, "pull", mirror]) == 0:
                return mirror
    return None


def host_driver_version() -> str:
    try:
        return Path("/sys/module/nvidia/version").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def read_metadata(img_dir: Path) -> dict[str, Any]:
    """Entrypoint/cwd that ch-image saved from the OCI config."""
    with open(img_dir / "ch" / "metadata.json") as f:
        return json.load(f)


def _bind(spec: str) -> list[str]:
    """Docker ``-v host:container[:ro]`` to ``ch-run -b host:container`` (always rw)."""
    parts = spec.split(":")
    if len(parts) == 3 and parts[2] in ("ro", "rw"):
        spec = f"{parts[0]}:{parts[1]}"
    return ["-b", spec]


def build_ch_run_cmd(
    img_dir: Path,
    *,
    ch_run: str,
    results_dir: str,
    config_path: str,
    env: dict[str, str],
    volumes: list[str],
    dev_mount: list[str] | None,
    inner_args: list[str],
) -> list[str]:
    """Assemble the ``ch-run`` command line (pure, unit-tested without Charliecloud)."""
    from vla_eval.cli._docker import CONTAINER_CONFIG, CONTAINER_RESULTS

    meta = read_metadata(img_dir)
    # Writable overlay for mount points; start from the image's env (as Docker does), not the host's.
    cmd = [ch_run, "--write-fake", "--unset-env=*", "--set-env"]
    if (img_dir / "root").is_dir():
        cmd.append("--set-env=HOME=/root")  # tools configured at build time live under /root
    cmd.extend(f"--set-env={k}={v}" for k, v in env.items())
    cmd.extend(["--cd", meta.get("cwd") or "/workspace"])
    cmd.extend(_bind(f"{results_dir}:{CONTAINER_RESULTS}"))
    cmd.extend(_bind(f"{config_path}:{CONTAINER_CONFIG}"))
    if dev_mount:
        cmd.extend(_bind(dev_mount[1]))
    for vol in volumes:
        cmd.extend(_bind(vol))
    return [*cmd, str(img_dir), "--", *(meta.get("entrypoint") or []), *inner_args]


def _gpu_env(gpus: str | None, shard_id: int | None, num_shards: int | None) -> dict[str, str]:
    """Device visibility env. ``--unset-env=*`` drops the scheduler's mask, so re-apply it for unset/all."""
    from vla_eval.docker_resources import gpu_visibility_env, is_no_gpu_spec, parse_gpus

    env: dict[str, str] = {}
    if num_shards is not None:
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})  # as shard_docker_flags; no cpuset here
    if is_no_gpu_spec(gpus):
        return {**env, "CUDA_VISIBLE_DEVICES": ""}
    want_all = gpus is None or gpus.strip().lower() == "all"
    host_mask = next((os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES") if k in os.environ), None)
    if num_shards is None:
        return {**env, **gpu_visibility_env(host_mask if want_all else gpus)}
    assert shard_id is not None
    pool = [g for g in host_mask.split(",") if g.strip()] if want_all and host_mask is not None else parse_gpus(gpus)
    if not pool:
        return {**env, "CUDA_VISIBLE_DEVICES": ""}
    return {**env, **gpu_visibility_env(pool[shard_id % len(pool)].strip())}


def run_via_charliecloud(
    config: dict[str, Any],
    *,
    auto_yes: bool = False,
    dev: bool = False,
    shard_id: int | None = None,
    num_shards: int | None = None,
    accept_license: list[str] | None = None,
    eval_id: str | None = None,
    no_save: bool = False,
    force_build: bool = False,
) -> int:
    """Execute the evaluation under ``ch-run``. Returns the exit code."""
    from vla_eval.cli._docker import dev_src_mount_flags, exec_child, inner_run_args, prepare_container_config
    from vla_eval.docker_resources import is_no_gpu_spec

    tools = find_tools()
    docker_cfg = DockerConfig.from_dict(config.get("docker"))
    if docker_cfg.image is None:
        _stderr_console().print("[red]ERROR: 'docker.image' must be set in config[/red]")
        sys.exit(1)
    if docker_cfg.user or docker_cfg.cpus:
        logger.info("docker.user / docker.cpus are ignored under Charliecloud (runs as the caller, no cpuset)")

    gpu = not is_no_gpu_spec(docker_cfg.gpus) and str(config.get("render", "gpu")).lower() != "cpu"
    img_dir = ensure_image_dir(
        docker_cfg.image, auto_yes=auto_yes, gpu=gpu, tools=tools, build=docker_cfg.build, force_build=force_build
    )

    dev_mount: list[str] | None = None
    if dev:
        try:
            dev_mount = dev_src_mount_flags()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    results_dir, config_path = prepare_container_config(config)
    env = {"VLA_EVAL_HOST_OUTPUT_DIR": results_dir}
    if os.environ.get("VLA_EVAL_WATCHDOG_TIMEOUT_S"):
        env["VLA_EVAL_WATCHDOG_TIMEOUT_S"] = os.environ["VLA_EVAL_WATCHDOG_TIMEOUT_S"]
    for env_str in docker_cfg.env:
        key, has_value, value = env_str.partition("=")
        env[key] = value if has_value else os.environ.get(key, "")
    if accept_license:
        env["VLA_EVAL_ACCEPTED_LICENSES"] = ",".join(accept_license)
    env.update(_gpu_env(docker_cfg.gpus, shard_id, num_shards))

    cmd = build_ch_run_cmd(
        img_dir,
        ch_run=tools["ch-run"],
        results_dir=results_dir,
        config_path=config_path,
        env=env,
        volumes=docker_cfg.volumes,
        dev_mount=dev_mount,
        inner_args=inner_run_args(shard_id=shard_id, num_shards=num_shards, eval_id=eval_id, no_save=no_save),
    )
    logger.info("Running via Charliecloud: %s", " ".join(cmd))

    def _stop(proc: Any) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    try:
        return exec_child(cmd, _stop)
    finally:
        Path(config_path).unlink(missing_ok=True)
