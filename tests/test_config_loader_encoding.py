"""Config files are UTF-8, independent of the host's default text encoding."""

import builtins
from pathlib import Path

import pytest

from vla_eval.cli import config_loader


@pytest.fixture
def cp936_default(monkeypatch):
    """Simulate a Windows CP936 locale without changing process-global state."""
    real_open = builtins.open

    def locale_open(file, mode="r", *args, encoding=None, **kwargs):
        return real_open(file, mode, *args, encoding=encoding or "cp936", **kwargs)

    monkeypatch.setattr(config_loader, "open", locale_open, raising=False)


def test_utf8_standalone_config(cp936_default, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("task: '抓取 π₀.5'\n", encoding="utf-8")

    assert config_loader.load_config(str(path))["task"] == "抓取 π₀.5"


@pytest.mark.parametrize("unicode_location", ["base", "child"])
def test_utf8_inherited_config(cp936_default, tmp_path, monkeypatch, unicode_location):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    text = "抓取 π₀.5"
    base_text = text if unicode_location == "base" else "base"
    child_text = text if unicode_location == "child" else "child"
    base.write_text(f"base_task: '{base_text}'\nargs:\n  retained: true\n  value: 1\n", encoding="utf-8")
    child.write_text(
        f"extends: base.yaml\nchild_task: '{child_text}'\nargs:\n  value: 2\n  root: ${{oc.env:VLA_ENCODING_TEST}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLA_ENCODING_TEST", "配置目录")

    config = config_loader.load_config(str(child))

    assert config["base_task"] == base_text
    assert config["child_task"] == child_text
    assert config["args"] == {"retained": True, "value": 2, "root": "配置目录"}


def test_packaged_pi05_config_under_cp936(cp936_default):
    path = Path(__file__).resolve().parents[1] / "configs/model_servers/lerobot/pi05_libero.yaml"

    config = config_loader.load_config(str(path))

    assert config["args"]["policy_type"] == "pi05"
