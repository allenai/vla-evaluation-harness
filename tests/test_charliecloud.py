"""Charliecloud runtime: command assembly and runtime resolution, without ch-run installed."""

from __future__ import annotations

import json
import shutil
import os
import time
from pathlib import Path

import pytest

from vla_eval.cli import _charliecloud as ch
from vla_eval.cli._docker import CONTAINER_CONFIG, CONTAINER_RESULTS, inner_run_args, resolve_runtime


@pytest.fixture(autouse=True)
def _isolated_ch_image_storage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CH_IMAGE_STORAGE", str(tmp_path / "ch-image-storage"))


def _fake_image(tmp_path: Path, *, with_root: bool = True) -> Path:
    img = tmp_path / "img"
    (img / "ch").mkdir(parents=True)
    (img / "ch" / "metadata.json").write_text(
        json.dumps({"entrypoint": ["conda", "run", "-n", "libero", "vla-eval"], "cmd": ["run"], "cwd": "/workspace"})
    )
    if with_root:
        (img / "root").mkdir()
    return img


def test_resolve_runtime_precedence(monkeypatch) -> None:
    monkeypatch.delenv("VLA_EVAL_RUNTIME", raising=False)
    assert resolve_runtime({}) == "docker"
    assert resolve_runtime({"docker": {"runtime": "charliecloud"}}) == "charliecloud"
    monkeypatch.setenv("VLA_EVAL_RUNTIME", "docker")
    assert resolve_runtime({"docker": {"runtime": "charliecloud"}}) == "docker"
    assert resolve_runtime({"docker": {"runtime": "docker"}}, "charliecloud") == "charliecloud"
    with pytest.raises(ValueError, match="unknown container runtime"):
        resolve_runtime({}, "podman")


def test_image_dir_mangles_like_ch_image(tmp_path: Path) -> None:
    d = ch.image_dir_for("ghcr.io/allenai/vla-evaluation-harness/libero:latest", root=tmp_path)
    assert d == tmp_path / "ghcr.io%allenai%vla-evaluation-harness%libero+latest"
    assert ch.image_dir_for("a/b:c", root=tmp_path, driver="580.95") == tmp_path / "a%b+c+nvidia-580.95"


def test_build_ch_run_cmd(tmp_path: Path) -> None:
    img = _fake_image(tmp_path)
    cmd = ch.build_ch_run_cmd(
        img,
        ch_run="/opt/ch-run",
        results_dir="/host/results",
        config_path="/host/cfg.yaml",
        env={"VLA_EVAL_HOST_OUTPUT_DIR": "/host/results", "CUDA_VISIBLE_DEVICES": "1"},
        volumes=["/data:/data:ro", "/x:/y"],
        dev_mount=["-v", "/src:/workspace/src"],
        inner_args=inner_run_args(shard_id=None, num_shards=None, eval_id="e1", no_save=False),
    )
    assert cmd[:4] == ["/opt/ch-run", "--write-fake", "--unset-env=*", "--set-env"]
    assert "--set-env=HOME=/root" in cmd
    assert "--set-env=CUDA_VISIBLE_DEVICES=1" in cmd
    assert cmd[cmd.index("--cd") + 1] == "/workspace"
    binds = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-b"]
    assert binds == [
        f"/host/results:{CONTAINER_RESULTS}",
        f"/host/cfg.yaml:{CONTAINER_CONFIG}",
        "/src:/workspace/src",
        "/data:/data",
        "/x:/y",
    ]
    sep = cmd.index("--")
    assert cmd[sep - 1] == str(img)
    assert cmd[sep + 1 :] == [
        "conda",
        "run",
        "-n",
        "libero",
        "vla-eval",
        "run",
        "--no-docker",
        "--config",
        CONTAINER_CONFIG,
        "--eval-id",
        "e1",
    ]


def test_build_ch_run_cmd_without_root_dir(tmp_path: Path) -> None:
    img = _fake_image(tmp_path, with_root=False)
    cmd = ch.build_ch_run_cmd(
        img,
        ch_run="ch-run",
        results_dir="/r",
        config_path="/c",
        env={},
        volumes=[],
        dev_mount=None,
        inner_args=["run"],
    )
    assert "--set-env=HOME=/root" not in cmd


def test_container_config_avoids_charliecloud_shared_tmp() -> None:
    assert not CONTAINER_CONFIG.startswith("/tmp/")


def test_gpu_env_none_all_and_shards(monkeypatch) -> None:
    monkeypatch.setattr("vla_eval.docker_resources._detect_runtime", lambda: "cuda")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    assert ch._gpu_env("none", None, None) == {"CUDA_VISIBLE_DEVICES": ""}
    assert ch._gpu_env("all", None, None) == {}
    assert ch._gpu_env("2,3", None, None) == {"CUDA_VISIBLE_DEVICES": "2,3"}
    env = ch._gpu_env("2,3", 1, 2)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["OMP_NUM_THREADS"] == "1"


def test_ensure_image_dir_pulls_converts_and_injects_per_driver(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ch.dirs, "home", lambda: tmp_path)
    calls: list[list[str]] = []

    def fake_call(cmd, *a, **kw):
        calls.append(list(cmd))
        if cmd[0] == "ch-convert":  # pretend the export produced a directory
            _fake_image(Path(cmd[-1]).parent).rename(Path(cmd[-1]))
        return 0

    monkeypatch.setattr(ch.subprocess, "call", fake_call)
    monkeypatch.setattr(ch.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(ch, "host_driver_version", lambda: "580.95")
    tools = {t: t for t in ch.TOOLS}
    img = ch.ensure_image_dir("reg/img:tag", auto_yes=True, gpu=True, tools=tools)
    assert img.name.endswith("+nvidia-580.95")
    assert [c[0] for c in calls] == ["ch-image", "ch-convert", "ch-fromhost"]
    assert calls[0][1:] == ["pull", "reg/img:tag"]
    assert calls[2][-1] == calls[1][-1] != str(img)  # injected into the temp export, then published
    calls.clear()
    assert ch.ensure_image_dir("reg/img:tag", auto_yes=False, gpu=True, tools=tools) == img
    assert calls == []  # cached: nothing re-run, nothing rewritten
    monkeypatch.setattr(ch, "host_driver_version", lambda: "590.10")  # another node / upgraded driver
    other = ch.ensure_image_dir("reg/img:tag", auto_yes=True, gpu=True, tools=tools)
    assert other != img and other.name.endswith("+nvidia-590.10")
    assert [c[0] for c in calls] == ["ch-image", "ch-convert", "ch-fromhost"]
    cpu = ch.ensure_image_dir("reg/img:tag", auto_yes=True, gpu=False, tools=tools)
    assert "+nvidia-" not in cpu.name


def test_ensure_image_dir_requires_confirmation_non_interactive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ch.dirs, "home", lambda: tmp_path)
    monkeypatch.setattr(ch.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit):
        ch.ensure_image_dir("reg/img:tag", auto_yes=False, gpu=False, tools={t: t for t in ch.TOOLS})


def test_run_via_charliecloud_env_and_cleanup(tmp_path: Path, monkeypatch) -> None:
    img_root = tmp_path / "home"
    monkeypatch.setattr(ch.dirs, "home", lambda: img_root)
    img = ch.image_dir_for("reg/img:tag")
    (img / "ch").mkdir(parents=True)
    (img / "ch" / "metadata.json").write_text(json.dumps({"entrypoint": ["vla-eval"], "cwd": "/workspace"}))
    monkeypatch.setattr(ch, "find_tools", lambda: {t: t for t in ch.TOOLS})
    monkeypatch.setattr("vla_eval.docker_resources._detect_runtime", lambda: "cuda")
    seen: dict[str, list[str]] = {}

    def fake_exec(cmd, stop):
        seen["cmd"] = cmd
        return 0

    monkeypatch.setattr("vla_eval.cli._docker.exec_child", fake_exec)
    config = {
        "output_dir": str(tmp_path / "out"),
        "render": "cpu",
        "docker": {"image": "reg/img:tag", "gpus": "none", "env": ["FOO=bar"], "volumes": ["/a:/b:ro"]},
        "benchmarks": [{"benchmark": "x:Y"}],
    }
    rc = ch.run_via_charliecloud(config, accept_license=["lic"], eval_id="e", no_save=True)
    assert rc == 0
    cmd = seen["cmd"]
    assert "--set-env=FOO=bar" in cmd
    assert "--set-env=VLA_EVAL_ACCEPTED_LICENSES=lic" in cmd
    assert "--set-env=CUDA_VISIBLE_DEVICES=" in cmd
    assert f"--set-env=VLA_EVAL_HOST_OUTPUT_DIR={(tmp_path / 'out').resolve()}" in cmd
    assert "/a:/b" in cmd and "--no-save" in cmd
    cfg_bind = next(b for b in cmd if b.endswith(f":{CONTAINER_CONFIG}"))
    assert not os.path.exists(cfg_bind.split(":")[0])  # temp config removed after the run


def test_gpu_env_inherits_scheduler_mask(monkeypatch) -> None:
    monkeypatch.setattr("vla_eval.docker_resources._detect_runtime", lambda: "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    assert ch._gpu_env(None, None, None) == {"CUDA_VISIBLE_DEVICES": "2,5"}
    assert ch._gpu_env("all", None, None) == {"CUDA_VISIBLE_DEVICES": "2,5"}
    assert ch._gpu_env("7", None, None) == {"CUDA_VISIBLE_DEVICES": "7"}  # explicit spec wins
    assert ch._gpu_env(None, 1, 2)["CUDA_VISIBLE_DEVICES"] == "5"  # shards round-robin inside the mask
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert ch._gpu_env(None, 0, 2)["CUDA_VISIBLE_DEVICES"] == ""  # empty mask: no device, limits still set


def test_ensure_image_dir_publishes_atomically_and_locks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ch.dirs, "home", lambda: tmp_path)
    targets: list[str] = []

    def fake_call(cmd, *a, **kw):
        if cmd[0] == "ch-convert":
            targets.append(cmd[-1])
            _fake_image(Path(cmd[-1]).parent).rename(Path(cmd[-1]))
        return 0

    monkeypatch.setattr(ch.subprocess, "call", fake_call)
    img = ch.ensure_image_dir("reg/img:tag", auto_yes=True, gpu=False, tools={t: t for t in ch.TOOLS})
    assert targets == [f"{img}.tmp-{os.getpid()}"]  # exported beside the final path, then renamed
    assert not Path(targets[0]).exists() and (img / "ch" / "metadata.json").is_file()
    assert Path(f"{img}.lock").exists()


def test_pull_falls_back_to_public_mirror(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_call(cmd, *a, **kw):
        attempts.append(cmd[-1])
        return 1 if cmd[-1].startswith("ghcr.io/allenai/") else 0

    monkeypatch.setattr(ch.subprocess, "call", fake_call)
    ref = ch._pull_with_mirror("ch-image", "ghcr.io/allenai/vla-evaluation-harness/libero:latest")
    assert ref == "ghcr.io/worv-ai/vla-evaluation-harness-public/libero:latest"
    assert attempts == ["ghcr.io/allenai/vla-evaluation-harness/libero:latest", ref]
    monkeypatch.setattr(ch.subprocess, "call", lambda cmd, *a, **kw: 1)
    assert ch._pull_with_mirror("ch-image", "example.org/x:y") is None


def test_shards_get_thread_limits_even_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr("vla_eval.docker_resources._detect_runtime", lambda: "cuda")
    env = ch._gpu_env("none", 0, 4)
    assert env == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    assert "OMP_NUM_THREADS" not in ch._gpu_env("none", None, None)


def test_ensure_image_dir_builds_from_dockerfile(tmp_path: Path, monkeypatch) -> None:
    from vla_eval.config import BuildConfig

    monkeypatch.setattr(ch.dirs, "home", lambda: tmp_path)
    stored: list[str] = []
    monkeypatch.setattr(ch, "_stored_images", lambda ch_image: stored)
    calls: list[list[str]] = []

    def fake_call(cmd, *a, **kw):
        calls.append(list(cmd))
        if cmd[0] == "ch-image":
            stored.append(cmd[3])
        if cmd[0] == "ch-convert":
            _fake_image(Path(cmd[-1]).parent).rename(Path(cmd[-1]))
        return 0

    monkeypatch.setattr(ch.subprocess, "call", fake_call)
    build = BuildConfig(context="/ctx")
    tools = {t: t for t in ch.TOOLS}
    ch.ensure_image_dir("x:local", auto_yes=False, gpu=False, tools=tools, build=build)
    assert calls[0] == ["ch-image", "build", "-t", "x:local", "-f", "/ctx/Dockerfile", "/ctx"]
    assert calls[1][0] == "ch-convert"

    calls.clear()  # in storage, this variant missing: export only
    shutil.rmtree(ch.image_dir_for("x:local"))
    ch.ensure_image_dir("x:local", auto_yes=False, gpu=False, tools=tools, build=build)
    assert [c[0] for c in calls] == ["ch-convert"]

    calls.clear()  # forced: rebuild, and other variants are dropped
    stale_gpu = ch.image_dir_for("x:local", driver="999.1")
    _fake_image(stale_gpu.parent).rename(stale_gpu)
    ch.ensure_image_dir("x:local", auto_yes=False, gpu=False, tools=tools, build=build, force_build=True)
    assert [c[0] for c in calls] == ["ch-image", "ch-convert"]
    assert not stale_gpu.exists()


def test_storage_lock_serialises_ch_image_across_images(tmp_path: Path, monkeypatch) -> None:
    """Different images still serialize on their shared ch-image storage."""
    import threading

    monkeypatch.setattr(ch.dirs, "home", lambda: tmp_path)
    monkeypatch.setattr(ch, "_stored_images", lambda ch_image: [])
    live = 0
    peak = 0
    guard = threading.Lock()

    def fake_call(cmd, *a, **kw):
        nonlocal live, peak
        if cmd[0] in ("ch-image", "ch-convert"):
            with guard:
                live += 1
                peak = max(peak, live)
            time.sleep(0.15)
            with guard:
                live -= 1
        if cmd[0] == "ch-convert":
            _fake_image(Path(cmd[-1]).parent).rename(Path(cmd[-1]))
        return 0

    monkeypatch.setattr(ch.subprocess, "call", fake_call)
    from vla_eval.config import BuildConfig

    tools = {t: t for t in ch.TOOLS}
    errors: list[BaseException] = []

    def build(name: str) -> None:
        try:
            ch.ensure_image_dir(name, auto_yes=True, gpu=False, tools=tools, build=BuildConfig(context="/ctx"))
        except BaseException as exc:  # surfaced after join
            errors.append(exc)

    threads = [threading.Thread(target=build, args=(f"img{i}:tag",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads) and not errors
    assert peak == 1, f"{peak} concurrent users of the ch-image storage; build and export must serialise"


def test_storage_lock_refuses_symlink(tmp_path: Path) -> None:
    victim = tmp_path / "bashrc"
    victim.write_text("keep")
    storage = tmp_path / "storage"
    planted = Path(f"{storage}.vla-eval-lock")
    planted.symlink_to(victim)
    with pytest.raises(OSError):
        ch._open_storage_lock(storage)
    assert victim.read_text() == "keep"


def test_storage_lock_stays_on_sidecar_after_initialization(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    first = ch._open_storage_lock(storage)
    first_inode = os.fstat(first.fileno()).st_ino
    first.close()
    storage.mkdir()
    (storage / "version").write_text("7\n")
    second = ch._open_storage_lock(storage)
    sidecar = Path(f"{storage}.vla-eval-lock")
    assert first_inode == os.fstat(second.fileno()).st_ino == sidecar.stat().st_ino
    second.close()


def test_storage_lock_does_not_fall_back_inside_existing_storage(tmp_path: Path, monkeypatch) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    version = storage / "version"
    version.write_text("7\n")

    def deny_sidecar(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr(ch.os, "open", deny_sidecar)
    with pytest.raises(PermissionError):
        ch._open_storage_lock(storage)


def test_storage_lock_failure_stops_preparation(tmp_path: Path, monkeypatch, capsys) -> None:
    import fcntl

    monkeypatch.setattr(ch.dirs, "home", lambda: tmp_path)
    original_flock = fcntl.flock
    failed_locks = []

    def fail_storage_lock(lock, operation):
        if isinstance(lock.name, int):
            failed_locks.append(lock)
            raise OSError("storage locking unavailable")
        return original_flock(lock, operation)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("image preparation started without the storage lock")

    monkeypatch.setattr(fcntl, "flock", fail_storage_lock)
    monkeypatch.setattr(ch.subprocess, "call", unexpected_call)
    with pytest.raises(SystemExit, match="1"):
        ch.ensure_image_dir("reg/img:tag", auto_yes=True, gpu=False, tools={t: t for t in ch.TOOLS})
    assert "cannot lock ch-image storage" in capsys.readouterr().err
    assert len(failed_locks) == 1 and failed_locks[0].closed
