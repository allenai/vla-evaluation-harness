"""``vla-eval test -c ... --validate`` validates the given configs and runs nothing else."""

from __future__ import annotations

from pathlib import Path

import pytest

from vla_eval.cli import main as cli
from vla_eval.cli import smoke

REPO = Path(__file__).resolve().parents[1]
BENCH = str(REPO / "configs/benchmarks/libero/spatial.yaml")
SERVER = str(REPO / "configs/model_servers/openvla/libero_10.yaml")


def _never(*args, **kwargs):
    raise AssertionError("a smoke test ran")


def test_validate_with_config_only_validates(monkeypatch, capsys) -> None:
    for name in ("check_docker", "check_uv", "run_benchmark_test", "run_server_test"):
        monkeypatch.setattr(smoke, name, _never)
    monkeypatch.setattr("sys.argv", ["vla-eval", "test", "-c", BENCH, "--validate"])
    cli.main()
    assert "1/1 configs valid" in capsys.readouterr().out


def test_validate_rejects_a_server_config(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["vla-eval", "test", "-c", SERVER, "--validate"])
    with pytest.raises(SystemExit):
        cli.main()
