"""Docker image identity: ``docker image inspect`` parsing, env plumbing and the mutable-tag warning."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from vla_eval.cli import _docker
from vla_eval.cli._docker import build_docker_command, docker_image_identity

DIGEST = "sha256:" + "ab" * 32
OTHER = "sha256:" + "cd" * 32
IMAGE_ID = "sha256:" + "ef" * 32


def _inspect(monkeypatch, info: dict[str, Any] | None) -> None:
    def fake_run(cmd, **kw):
        assert cmd[1:3] == ["image", "inspect"]
        if info is None:
            return subprocess.CompletedProcess(cmd, 1, stdout="[]", stderr="No such image")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps([info]), stderr="")

    monkeypatch.setattr(_docker.subprocess, "run", fake_run)


@pytest.mark.parametrize(
    ("image", "repo_digests", "digest"),
    [
        ("ghcr.io/org/img:latest", [f"mirror.io/x/img@{OTHER}", f"ghcr.io/org/img@{DIGEST}"], DIGEST),  # own repo
        ("ghcr.io/org/img:latest", [f"mirror.io/x/img@{OTHER}"], OTHER),  # retagged from a mirror
        (f"ghcr.io/org/img@{DIGEST}", [f"mirror.io/x/img@{DIGEST}"], DIGEST),  # the pin is authoritative
        ("local:vla-eval", [], None),  # a local build has no registry digest
    ],
)
def test_identity_from_inspect(monkeypatch, image, repo_digests, digest) -> None:
    _inspect(monkeypatch, {"Id": IMAGE_ID, "RepoDigests": repo_digests})
    identity = docker_image_identity("docker", image)
    assert identity["image"] == image and identity["runtime"] == "docker" and identity["id"] == IMAGE_ID
    assert identity.get("digest") == digest


def test_identity_never_raises(monkeypatch, caplog) -> None:
    _inspect(monkeypatch, None)
    with caplog.at_level(logging.WARNING):
        assert docker_image_identity("docker", "x:y") == {"image": "x:y", "runtime": "docker"}
    assert "could not resolve" in caplog.text


def test_identity_goes_in_as_env_and_the_resolved_ref_runs() -> None:
    config = {"docker": {"image": f"ghcr.io/org/img@{DIGEST}", "env": ["HF_TOKEN"]}}
    identity = {"image": f"ghcr.io/org/img@{DIGEST}", "digest": DIGEST, "id": IMAGE_ID, "runtime": "docker"}
    cmd = build_docker_command(
        "docker", config, "/out", "/cfg.yaml", "c", interactive=False, image_ref="mirror/img", image_identity=identity
    )
    envs = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
    assert f"VLA_EVAL_IMAGE={identity['image']}" in envs and f"VLA_EVAL_IMAGE_DIGEST={DIGEST}" in envs
    assert f"VLA_EVAL_IMAGE_ID={IMAGE_ID}" in envs and "VLA_EVAL_IMAGE_RUNTIME=docker" in envs
    assert "HF_TOKEN" in envs  # docker.env passes through by name; its value is never recorded
    assert cmd[cmd.index("mirror/img") + 1] == "run" and config["docker"]["image"] not in cmd
    plain = build_docker_command("docker", config, "/out", "/cfg.yaml", "c", interactive=False)
    assert not any(tok.startswith("VLA_EVAL_IMAGE") for tok in plain) and config["docker"]["image"] in plain


def test_dev_mount_is_recorded(monkeypatch) -> None:
    """``--dev`` runs host src/ over the image's: the digest alone no longer describes the code that ran."""
    monkeypatch.setattr(_docker, "dev_src_mount_flags", lambda: ["-v", "/src:/workspace/src"])
    config = {"docker": {"image": "ghcr.io/org/img:1"}}
    identity = {"image": "ghcr.io/org/img:1", "digest": DIGEST, "runtime": "docker"}
    cmd = build_docker_command("docker", config, "/out", "/cfg.yaml", "c", interactive=False, image_identity=identity)
    assert "VLA_EVAL_IMAGE_DEV=1" not in cmd
    cmd = build_docker_command(
        "docker", config, "/out", "/cfg.yaml", "c", interactive=False, dev=True, image_identity=identity
    )
    assert "VLA_EVAL_IMAGE_DEV=1" in cmd and f"VLA_EVAL_IMAGE_DIGEST={DIGEST}" in cmd


def _run_via_docker(monkeypatch, tmp_path: Path, image: str, **docker: Any) -> list[str]:
    monkeypatch.setattr("shutil.which", lambda name: name)
    monkeypatch.setattr(_docker, "check_docker_daemon", lambda d: None)
    monkeypatch.setattr(_docker, "ensure_image_local", lambda *a, **kw: image)
    _inspect(monkeypatch, {"Id": IMAGE_ID, "RepoDigests": [f"ghcr.io/org/img@{DIGEST}"]})
    seen: list[list[str]] = []
    monkeypatch.setattr(_docker, "exec_docker", lambda d, cmd, name: seen.append(cmd) or 0)
    config = {"output_dir": str(tmp_path), "docker": {"image": image, "gpus": "none", **docker}, "benchmarks": []}
    assert _docker.run_via_docker(config) == 0
    return seen[0]


def test_run_via_docker_records_the_digest_and_warns_on_a_mutable_tag(monkeypatch, tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        cmd = _run_via_docker(monkeypatch, tmp_path, "ghcr.io/org/img:latest")
    assert f"VLA_EVAL_IMAGE_DIGEST={DIGEST}" in cmd and "VLA_EVAL_IMAGE=ghcr.io/org/img:latest" in cmd
    (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert f"docker.image: ghcr.io/org/img@{DIGEST}" in warning


@pytest.mark.parametrize(
    ("image", "build"),
    [(f"ghcr.io/org/img@{DIGEST}", None), ("ghcr.io/org/img:0.4.0", None), ("ghcr.io/org/img", ".")],
)
def test_pinned_tagged_or_built_images_do_not_warn(monkeypatch, tmp_path, caplog, image, build) -> None:
    with caplog.at_level(logging.WARNING):
        cmd = _run_via_docker(monkeypatch, tmp_path, image, build=build)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert f"VLA_EVAL_IMAGE={image}" in cmd
