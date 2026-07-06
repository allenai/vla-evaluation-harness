"""CLI-level recording override behavior."""

from __future__ import annotations

import argparse

import pytest

from vla_eval.cli import main as cli


def test_record_video_override_creates_recording_blocks() -> None:
    config = {
        "benchmarks": [
            {"name": "implicit"},
            {"name": "explicit", "recording": {"record_step": False, "record_video": False}},
            {"name": "null", "recording": None},
        ]
    }

    cli._apply_record_video_override(config, enabled=True, create_missing=True)

    assert config["benchmarks"][0]["recording"] == {"record_video": True}
    assert config["benchmarks"][1]["recording"] == {"record_step": False, "record_video": True}
    assert config["benchmarks"][2]["recording"] == {"record_video": True}


def test_no_record_video_override_does_not_create_recording_blocks() -> None:
    config = {
        "benchmarks": [
            {"name": "implicit"},
            {"name": "explicit", "recording": {"record_step": True, "record_video": True}},
            {"name": "null", "recording": None},
        ]
    }

    cli._apply_record_video_override(config, enabled=False, create_missing=False)

    assert "recording" not in config["benchmarks"][0]
    assert config["benchmarks"][1]["recording"] == {"record_step": True, "record_video": False}
    assert config["benchmarks"][2]["recording"] is None


def test_record_video_override_rejects_invalid_recording_block() -> None:
    config = {"benchmarks": [{"name": "bad", "recording": "yes"}]}

    with pytest.raises(ValueError, match="recording must be a mapping"):
        cli._apply_record_video_override(config, enabled=True, create_missing=True)


def test_cmd_run_rejects_record_video_with_no_save(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_load_config", lambda _path: {"benchmarks": []})
    args = argparse.Namespace(
        config="unused.yaml",
        server_url=None,
        output_dir=None,
        param=None,
        shard_id=None,
        num_shards=None,
        eval_id=None,
        no_save=True,
        record_video=True,
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_run(args)

    assert exc.value.code == 1
