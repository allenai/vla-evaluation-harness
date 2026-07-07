"""CLI-level recording override behavior."""

from __future__ import annotations

import argparse

import pytest

from vla_eval.cli import main as cli


@pytest.mark.parametrize("enabled", [True, False])
def test_record_video_override_sets_flag_on_all_benchmarks(enabled: bool) -> None:
    config = {
        "benchmarks": [
            {"name": "implicit"},
            {"name": "explicit", "recording": {"record_step": False, "record_video": not enabled}},
            {"name": "null", "recording": None},
        ]
    }

    cli._apply_record_video_override(config, enabled=enabled)

    assert config["benchmarks"][0]["recording"] == {"record_video": enabled}
    assert config["benchmarks"][1]["recording"] == {"record_step": False, "record_video": enabled}
    assert config["benchmarks"][2]["recording"] == {"record_video": enabled}


def test_record_video_override_rejects_invalid_recording_block() -> None:
    config = {"benchmarks": [{"name": "bad", "recording": "yes"}]}

    with pytest.raises(ValueError, match="recording must be a mapping"):
        cli._apply_record_video_override(config, enabled=True)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], None),
        (["--record-video"], True),
        (["--no-record-video"], False),
    ],
)
def test_record_video_action_parses_boolean_optional_flag(argv: list[str], expected: bool | None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-video", "--no-record-video", action=cli._RecordVideoAction, default=None)

    assert parser.parse_args(argv).record_video is expected


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
