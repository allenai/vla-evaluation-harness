"""Smoke uses production container options and cleans up on timeout."""

import json
from pathlib import Path
import subprocess

import pytest
import yaml

from vla_eval.cli import smoke


@pytest.mark.parametrize("timeout", [False, True])
def test_smoke_container_paths_user_and_cleanup(monkeypatch, tmp_path, timeout):
    config = {
        "docker": {"image": "test-image", "user": "123:456", "gpus": "none"},
        "output_dir": str(tmp_path / "custom-output"),
        "render": "cpu",
        "benchmarks": [{"benchmark": "test:Benchmark", "episodes_per_task": 10}],
    }
    monkeypatch.setattr(smoke, "_load_yaml", lambda _: config)
    monkeypatch.setattr(smoke.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(smoke, "check_docker", lambda: (True, ""))
    monkeypatch.setattr(smoke, "check_docker_image", lambda _: (True, ""))
    paths = []
    removed = []

    def run(cmd, **kwargs):
        if cmd[1:3] == ["rm", "-f"]:
            removed.append(cmd[-1])
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[cmd.index("--user") + 1] == "123:456"
        mounts = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-v"]
        result_dir = Path(next(m.split(":")[0] for m in mounts if m.endswith(":/workspace/results")))
        config_path = Path(next(m.split(":")[0] for m in mounts if m.endswith(":ro")))
        paths.extend([result_dir, config_path])
        inner = yaml.safe_load(config_path.read_text())
        assert inner["output_dir"] == "/workspace/results"
        assert inner["benchmarks"][0]["episodes_per_task"] == 1
        if timeout:
            raise subprocess.TimeoutExpired(cmd, 1)
        (result_dir / "test_aggregate.json").write_text(json.dumps({"tasks": [], "mean_success": 1}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", run)
    test = smoke.SmokeTest("benchmark", "test", tmp_path / "config.yaml", "test")
    result = smoke.run_benchmark_test(test, timeout=1)
    assert result.status == ("fail" if timeout else "pass"), result.message
    assert len(removed) == int(timeout)
    assert paths and all(not p.exists() for p in paths)
    assert config["benchmarks"][0]["episodes_per_task"] == 10
