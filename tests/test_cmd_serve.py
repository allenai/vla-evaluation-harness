"""``cmd_serve`` stringifies the yaml's args block onto the inner argv with
``--<server_key>=v`` for port/host and ``--args.<key>=v`` for class kwargs."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

from vla_eval.cli.main import cmd_serve


def _yaml(args_block: dict, script: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    import os

    os.close(fd)
    with open(path, "w") as f:
        yaml.safe_dump({"script": script, "args": args_block}, f)
    return Path(path)


def _capture_cmd(monkeypatch, ns: argparse.Namespace) -> list[str]:
    captured: list[str] = []

    def _stub_exec(cmd):
        captured.extend(cmd)
        raise SystemExit(0)

    monkeypatch.setattr("vla_eval.cli.main._exec_subprocess", _stub_exec)
    try:
        cmd_serve(ns)
    except SystemExit:
        pass
    return captured


def test_cmd_serve_emits_argv_with_args_prefix(monkeypatch, tmp_path):
    """class kwargs get ``--args.foo=bar``; server-level keys go to root."""
    script = tmp_path / "server.py"
    script.write_text("")
    yaml_path = _yaml({"model_path": "openvla/openvla-7b", "port": 8000}, script=str(script))
    try:
        ns = argparse.Namespace(config=str(yaml_path), address=None, arg=None)
        cmd = _capture_cmd(monkeypatch, ns)
        assert "--port=8000" in cmd
        assert "--args.model_path=openvla/openvla-7b" in cmd
        # no tempfile path emitted
        assert "--config" not in cmd
    finally:
        yaml_path.unlink()


def test_cmd_serve_host_also_routed(monkeypatch, tmp_path):
    script = tmp_path / "server.py"
    script.write_text("")
    yaml_path = _yaml({"model_path": "x", "host": "0.0.0.0", "port": 9001}, script=str(script))
    try:
        ns = argparse.Namespace(config=str(yaml_path), address=None, arg=None)
        cmd = _capture_cmd(monkeypatch, ns)
        assert "--host=0.0.0.0" in cmd
        assert "--port=9001" in cmd
        assert "--args.model_path=x" in cmd
        assert "--args.host=0.0.0.0" not in cmd
        assert "--args.port=9001" not in cmd
    finally:
        yaml_path.unlink()


def test_cmd_serve_arg_override_routes_by_key(monkeypatch, tmp_path):
    """server-level override → root flag; class-init override → ``--args.*``."""
    script = tmp_path / "server.py"
    script.write_text("")
    yaml_path = _yaml({"cache_len": 4096}, script=str(script))
    try:
        ns = argparse.Namespace(config=str(yaml_path), address=None, arg=["cache_len=2048", "port=9002"])
        cmd = _capture_cmd(monkeypatch, ns)
        assert "--args.cache_len=2048" in cmd
        assert "--port=9002" in cmd
    finally:
        yaml_path.unlink()


def test_cmd_serve_address_override(monkeypatch, tmp_path):
    script = tmp_path / "server.py"
    script.write_text("")
    yaml_path = _yaml({"model_path": "x"}, script=str(script))
    try:
        ns = argparse.Namespace(config=str(yaml_path), address="0.0.0.0:9001", arg=None)
        cmd = _capture_cmd(monkeypatch, ns)
        i = cmd.index("--address")
        assert cmd[i + 1] == "0.0.0.0:9001"
    finally:
        yaml_path.unlink()


def test_cmd_serve_handles_yaml_with_null_args(monkeypatch, tmp_path):
    """yaml ``args:`` with no body parses to None — must not crash on dict(None)."""
    script = tmp_path / "server.py"
    script.write_text("")
    fd, p = tempfile.mkstemp(suffix=".yaml")
    import os

    os.close(fd)
    Path(p).write_text(f"script: {script}\nargs:\n")
    try:
        ns = argparse.Namespace(config=p, address=None, arg=None)
        cmd = _capture_cmd(monkeypatch, ns)
        assert not any(c.startswith("--args.") for c in cmd)
    finally:
        Path(p).unlink()


def test_cmd_serve_list_and_dict_round_trip_as_json(monkeypatch, tmp_path):
    """yaml list/dict values stringify as JSON literals so jsonargparse parses them back."""
    import json

    script = tmp_path / "server.py"
    script.write_text("")
    yaml_path = _yaml({"camera_keys": ["agentview", "wrist"], "thresholds": {"x": 0.1}}, script=str(script))
    try:
        ns = argparse.Namespace(config=str(yaml_path), address=None, arg=None)
        cmd = _capture_cmd(monkeypatch, ns)
        token = next(c for c in cmd if c.startswith("--args.camera_keys="))
        assert json.loads(token.split("=", 1)[1]) == ["agentview", "wrist"]
        token = next(c for c in cmd if c.startswith("--args.thresholds="))
        assert json.loads(token.split("=", 1)[1]) == {"x": 0.1}
    finally:
        yaml_path.unlink()


def test_smoke_reuses_serve_helpers():
    """``cli/smoke.py`` and ``cmd_serve`` share the cmd builder so paths can't drift."""
    import vla_eval.cli.smoke as smoke

    src = Path(smoke.__file__).read_text()
    assert "_build_serve_cmd" in src, (
        "smoke.py must import the shared cmd builder from cli.main; otherwise the "
        "jsonargparse routing in production won't apply to `vla-eval test --server`."
    )
