"""Charliecloud runtime: run the benchmark image with no daemon and no root (docs/runtimes.md)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vla_eval import dirs
from vla_eval.cli._console import stderr_console as _stderr_console
from vla_eval.config import BuildConfig, CharliecloudConfig, DockerConfig

logger = logging.getLogger(__name__)

TOOLS = ("ch-image", "ch-run", "ch-convert", "ch-fromhost")
# Mount externally because some ch-run builds lack SquashFUSE support.
SQUASHFUSE_TOOLS = ("squashfuse_ll", "squashfuse")
FUSERMOUNT_TOOLS = ("fusermount3", "fusermount")
IMAGE_FORMAT_ENV = "VLA_EVAL_CH_IMAGE_FORMAT"
SQFS_SUFFIX = ".sqfs"


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


def image_dir_for(image: str, root: Path | None = None, driver: str | None = None, image_format: str = "dir") -> Path:
    """Export path, keyed by image, host driver and format."""
    root = root or dirs.home() / "charliecloud"
    name = image.replace("/", "%").replace(":", "+")
    name = f"{name}+nvidia-{driver}" if driver else name
    return root / ".squashfs" / (name + SQFS_SUFFIX) if image_format == "squashfs" else root / name


def resolve_image_format(docker_cfg: DockerConfig | None = None) -> str:
    """``$VLA_EVAL_CH_IMAGE_FORMAT`` > config > ``"auto"``."""
    fmt = os.environ.get(IMAGE_FORMAT_ENV) or (docker_cfg.charliecloud.image_format if docker_cfg else None)
    return CharliecloudConfig.from_value({"image_format": fmt}).image_format


def _fuse_available() -> bool:
    try:
        fd = os.open("/dev/fuse", os.O_RDWR | os.O_CLOEXEC)
    except OSError:
        return False
    os.close(fd)
    return True


def select_image_format(
    requested: str, image: str, *, gpu: bool, rebuild: bool = False
) -> tuple[str, tuple[str, str] | None]:
    """Choose a format and discover its mount tools once, before preparing an image."""
    if requested == "dir":
        return "dir", None
    mount = next((p for t in SQUASHFUSE_TOOLS if (p := shutil.which(t))), None)
    unmount = next((p for t in FUSERMOUNT_TOOLS if (p := shutil.which(t))), None)
    issues = [f"{name} is unavailable" for name, path in (("squashfuse", mount), ("fusermount", unmount)) if not path]
    if requested == "auto":
        sqfs = image_dir_for(image, driver=host_driver_version() if gpu else None, image_format="squashfs")
        if (rebuild or not sqfs.is_file()) and shutil.which("mksquashfs") is None:
            issues.append("mksquashfs is unavailable")
        if not _fuse_available():
            issues.append("/dev/fuse is unavailable")
    if issues:
        reason = "; ".join(issues)
        if requested == "squashfs":
            raise ValueError(f"{reason}; install squashfuse and fuse3, or use image_format dir")
        logger.info("Charliecloud image format auto: using dir (%s)", reason)
        return "dir", None
    assert mount is not None and unmount is not None
    return "squashfs", (mount, unmount)


def _remove(path: Path) -> None:
    """Remove an export, whichever shape it has."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _storage_path(ch_image: str) -> Path:
    storage = os.environ.get("CH_IMAGE_STORAGE")
    if not storage:
        result = subprocess.run([ch_image, "gestalt", "storage-path"], capture_output=True, text=True)
        storage = result.stdout.strip()
    if not storage:
        raise RuntimeError("ch-image did not report its storage path")
    return Path(os.path.realpath(storage))


def _open_storage_lock(storage: Path) -> Any:
    """Acquire the dedicated sidecar lock without changing its identity on failure."""
    import fcntl

    path = Path(f"{storage}.vla-eval-lock")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    lock = os.fdopen(fd, "r+")
    try:
        if os.fstat(fd).st_uid != os.getuid():
            raise PermissionError(f"lock belongs to another user: {path}")
        fcntl.flock(lock, fcntl.LOCK_EX)
    except BaseException:
        lock.close()
        raise
    return lock


def ensure_image_dir(
    image: str,
    *,
    auto_yes: bool,
    gpu: bool,
    tools: dict[str, str] | None = None,
    build: BuildConfig | None = None,
    force_build: bool = False,
    image_format: str = "dir",
) -> Path:
    """Pull or build, export and (when *gpu*) inject the driver on first use, under a per-directory lock."""
    import fcntl

    tools = tools or find_tools()
    squash = image_format == "squashfs"
    img_dir = image_dir_for(image, driver=host_driver_version() if gpu else None, image_format=image_format)
    img_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{img_dir}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rebuild = force_build and build is not None
        if (img_dir.is_file() if squash else (img_dir / "ch" / "metadata.json").is_file()) and not rebuild:
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
        if squash and shutil.which("mksquashfs") is None:
            con.print("[red]ERROR: image_format squashfs needs mksquashfs (squashfs-tools) on the host.[/red]")
            sys.exit(1)
        tmp_dir = Path(f"{img_dir}.tmp-{os.getpid()}")  # ch-run cannot use ch-image's storage directly
        try:
            storage_lock = _open_storage_lock(_storage_path(tools["ch-image"]))
        except (OSError, RuntimeError) as exc:
            con.print(f"[red]ERROR: cannot lock ch-image storage: {exc}[/red]")
            sys.exit(1)
        with storage_lock:
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
            if rebuild:  # other cached exports of this image, in either format, are now stale
                for fmt in ("dir", "squashfs"):
                    base_path = image_dir_for(image, image_format=fmt)
                    base = base_path.stem if fmt == "squashfs" else base_path.name
                    for d in base_path.parent.glob(f"{base}*"):
                        suffix = d.name[len(base) :]
                        if suffix.endswith(".lock") or ".tmp-" in suffix:
                            continue
                        stem = d.stem if fmt == "squashfs" and d.suffix == SQFS_SUFFIX else d.name
                        if d != img_dir and (stem == base or stem.startswith(f"{base}+nvidia-")):
                            _remove(d)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            ok = subprocess.call([tools["ch-convert"], "-i", "ch-image", "-o", "dir", pulled, str(tmp_dir)]) == 0
        # Inject the driver before packing the read-only SquashFS image.
        ok = ok and (not gpu or subprocess.call([tools["ch-fromhost"], "--nvidia", str(tmp_dir)]) == 0)
        tmp_out = tmp_dir
        if ok and squash:
            tmp_out = Path(f"{tmp_dir}{SQFS_SUFFIX}")
            tmp_out.unlink(missing_ok=True)
            ok = subprocess.call([tools["ch-convert"], "-i", "dir", "-o", "squash", str(tmp_dir), str(tmp_out)]) == 0
            shutil.rmtree(tmp_dir, ignore_errors=True)  # the tree is what the .sqfs is there to avoid
        if not ok:
            _remove(tmp_dir)
            _remove(tmp_out)
            con.print(f"[red]ERROR: exporting {image} failed.[/red]")
            sys.exit(1)
        _remove(img_dir)
        os.rename(tmp_out, img_dir)
    return img_dir


@contextlib.contextmanager
def mount_image(img: Path, squashfuse: tuple[str, str] | None) -> Iterator[Path]:
    """Mount a SquashFS export for this run and release it on every exit path."""
    if img.is_dir():
        yield img
        return
    assert squashfuse is not None
    mount_tool, fusermount = squashfuse
    mnt = Path(tempfile.mkdtemp(prefix="vla-eval-ch-"))
    mounted = False
    try:
        options = f"ro,uid={os.getuid()},gid={os.getgid()}"
        mounted = subprocess.call([mount_tool, "-o", options, str(img), str(mnt)]) == 0
        if not mounted:
            raise OSError(f"{mount_tool} could not mount {img}; check /dev/fuse access")
        if not (mnt / "ch" / "metadata.json").is_file():
            raise ValueError(f"{img} is not a Charliecloud image (no ch/metadata.json)")
        yield mnt
    finally:
        if mounted:
            for extra in ((), ("-z",)):  # Retry a busy mount with lazy unmount.
                if subprocess.call([fusermount, "-u", *extra, str(mnt)], stderr=subprocess.DEVNULL) == 0:
                    break
            else:
                logger.warning("could not unmount %s; run %s -u when free", mnt, fusermount)
        with contextlib.suppress(OSError):
            mnt.rmdir()


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
    try:
        requested_format = resolve_image_format(docker_cfg)
        image_format, squashfuse = select_image_format(
            requested_format, docker_cfg.image, gpu=gpu, rebuild=force_build and docker_cfg.build is not None
        )
    except ValueError as exc:
        _stderr_console().print(f"[red]ERROR: {exc}[/red]")
        sys.exit(1)
    img_dir = ensure_image_dir(
        docker_cfg.image,
        auto_yes=auto_yes,
        gpu=gpu,
        tools=tools,
        build=docker_cfg.build,
        force_build=force_build,
        image_format=image_format,
    )
    dev_mount: list[str] | None = None
    if dev:
        try:
            dev_mount = dev_src_mount_flags()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    with contextlib.ExitStack() as cleanup:
        try:
            img_dir = cleanup.enter_context(mount_image(img_dir, squashfuse))
        except OSError as exc:
            if requested_format != "auto" or image_format != "squashfs":
                _stderr_console().print(f"[red]ERROR: {exc}[/red]")
                sys.exit(1)
            logger.warning("SquashFS mount failed; using directory export: %s", exc)
            img_dir = ensure_image_dir(
                docker_cfg.image,
                auto_yes=auto_yes,
                gpu=gpu,
                tools=tools,
                build=docker_cfg.build,
                image_format="dir",
            )
        except ValueError as exc:
            _stderr_console().print(f"[red]ERROR: {exc}[/red]")
            sys.exit(1)
        results_dir, config_path = prepare_container_config(config)
        cleanup.callback(Path(config_path).unlink, missing_ok=True)
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

        return exec_child(cmd, _stop)
