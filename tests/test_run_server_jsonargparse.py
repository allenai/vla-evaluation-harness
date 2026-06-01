"""``run_server`` builds argparse from class signature via jsonargparse, plus
the ``ActionConfigFile`` yaml-direct path that ``cmd_serve`` exercises."""

from __future__ import annotations

import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from vla_eval.model_servers.base import ModelServer
from vla_eval.model_servers.serve import run_server


class _Mode(str, Enum):
    FAST = "fast"
    SLOW = "slow"


def _drive(server_cls: type[ModelServer], argv: list[str], monkeypatch) -> Any:
    """Drive ``run_server`` against argv with ``serve`` stubbed to capture+exit
    (keeps the real ``__init__`` signature visible to jsonargparse)."""
    captured: dict[str, Any] = {}

    def _capture_serve(server, host, port):
        captured["server"] = server
        sys.exit(0)

    monkeypatch.setattr("vla_eval.model_servers.serve.serve", _capture_serve)
    saved_argv = sys.argv
    sys.argv = ["serve.py", *argv]
    try:
        with pytest.raises(SystemExit) as exc:
            run_server(server_cls)
        assert exc.value.code == 0, f"run_server exited with {exc.value.code}"
    finally:
        sys.argv = saved_argv
    return captured["server"]


def _yaml_config(args_block: dict, script: str = "/unused/script.py") -> Path:
    """Write a temp yaml in the cmd_serve format (script + args)."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    import os

    os.close(fd)
    with open(path, "w") as f:
        yaml.safe_dump({"script": script, "args": args_block}, f)
    return Path(path)


class _BaseModelServer(ModelServer):
    """ModelServer subclass that records its ``__init__`` kwargs on self."""

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_action_spec(self):
        return {}

    def get_observation_spec(self):
        return {}

    async def on_observation(self, *a, **k):
        pass


class TestNullableArg:
    """yaml-null at the field level routes to Python ``None`` for ``T | None``."""

    def test_int_or_none_with_null_literal(self, monkeypatch):
        class S(_BaseModelServer):
            def __init__(self, cache_len: int | None = 4096):
                super().__init__(cache_len=cache_len)

        s = _drive(S, ["--args.cache_len", "null"], monkeypatch)
        assert s.cache_len is None

    def test_int_or_none_with_int_value(self, monkeypatch):
        class S(_BaseModelServer):
            def __init__(self, cache_len: int | None = 4096):
                super().__init__(cache_len=cache_len)

        s = _drive(S, ["--args.cache_len", "2048"], monkeypatch)
        assert s.cache_len == 2048

    def test_omitting_flag_uses_default(self, monkeypatch):
        class S(_BaseModelServer):
            def __init__(self, cache_len: int | None = 4096):
                super().__init__(cache_len=cache_len)

        s = _drive(S, [], monkeypatch)
        assert s.cache_len == 4096


class TestLiteralArg:
    def test_literal_value_accepted(self, monkeypatch):
        class S(_BaseModelServer):
            def __init__(self, mode: Literal["max-autotune", "reduce-overhead"] = "max-autotune"):
                super().__init__(mode=mode)

        s = _drive(S, ["--args.mode", "reduce-overhead"], monkeypatch)
        assert s.mode == "reduce-overhead"

    def test_literal_invalid_rejected(self, monkeypatch):
        class S(_BaseModelServer):
            def __init__(self, mode: Literal["max-autotune", "reduce-overhead"] = "max-autotune"):
                super().__init__(mode=mode)

        monkeypatch.setattr("vla_eval.model_servers.serve.serve", lambda *a, **k: sys.exit(0))
        saved_argv = sys.argv
        sys.argv = ["serve.py", "--args.mode", "turbo"]
        try:
            with pytest.raises(SystemExit) as exc:
                run_server(S)
            assert exc.value.code != 0
        finally:
            sys.argv = saved_argv


class TestEnumArg:
    def test_enum_by_name(self, monkeypatch):
        class S(_BaseModelServer):
            def __init__(self, mode: _Mode = _Mode.FAST):
                super().__init__(mode=mode)

        s = _drive(S, ["--args.mode", "SLOW"], monkeypatch)
        assert s.mode is _Mode.SLOW


class TestPathArg:
    def test_path_typed(self, monkeypatch, tmp_path):
        target = tmp_path / "ckpt"

        class S(_BaseModelServer):
            def __init__(self, ckpt: Path = Path("/")):
                super().__init__(ckpt=ckpt)

        s = _drive(S, ["--args.ckpt", str(target)], monkeypatch)
        assert isinstance(s.ckpt, Path)
        assert s.ckpt == target


class TestActionConfigFile:
    """End-to-end yaml load path that ``cmd_serve`` will use."""

    def test_yaml_load_types_through(self, monkeypatch):
        """Yaml null / int / Literal all round-trip without stringify pain."""

        class S(_BaseModelServer):
            def __init__(
                self,
                model_path: str = "default",
                cache_len: int | None = 4096,
                mode: Literal["max-autotune", "reduce-overhead"] = "max-autotune",
            ):
                super().__init__(model_path=model_path, cache_len=cache_len, mode=mode)

        yaml_path = _yaml_config({"model_path": "openvla/openvla-7b", "cache_len": None, "mode": "reduce-overhead"})
        try:
            s = _drive(S, ["--config", str(yaml_path)], monkeypatch)
            assert s.model_path == "openvla/openvla-7b"
            assert s.cache_len is None
            assert s.mode == "reduce-overhead"
        finally:
            yaml_path.unlink()

    def test_cli_override_beats_yaml(self, monkeypatch):
        """``--args.foo=value`` beats the yaml's ``args.foo``."""

        class S(_BaseModelServer):
            def __init__(self, cache_len: int | None = 4096):
                super().__init__(cache_len=cache_len)

        yaml_path = _yaml_config({"cache_len": 1024})
        try:
            s = _drive(S, ["--config", str(yaml_path), "--args.cache_len", "8192"], monkeypatch)
            assert s.cache_len == 8192
        finally:
            yaml_path.unlink()

    def test_yaml_script_field_ignored(self, monkeypatch):
        """Inner parser accepts and ignores yaml's host-side ``script:`` field."""

        class S(_BaseModelServer):
            def __init__(self, cache_len: int = 4096):
                super().__init__(cache_len=cache_len)

        yaml_path = _yaml_config({"cache_len": 2048}, script="/some/path.py")
        try:
            s = _drive(S, ["--config", str(yaml_path)], monkeypatch)
            assert s.cache_len == 2048
        finally:
            yaml_path.unlink()
