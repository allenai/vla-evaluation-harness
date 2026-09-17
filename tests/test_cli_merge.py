"""How `vla-eval merge` picks the recording DBs to materialize."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from vla_eval.cli import main as cli


def _args(**kwargs: object) -> argparse.Namespace:
    base: dict[str, object] = {"db": None, "config": None, "output_dir": None, "eval_id": None, "verbose": False}
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.fixture
def merged(monkeypatch) -> list[tuple[Path, Path]]:
    """Capture (db_path, output_dir) per merge_db call instead of touching SQLite."""
    calls: list[tuple[Path, Path]] = []

    def fake_merge_db(db_path: Path, output_dir: Path) -> list[dict[str, object]]:
        calls.append((db_path, output_dir))
        return []

    monkeypatch.setattr("vla_eval.results.merge.merge_db", fake_merge_db)
    return calls


def test_output_dir_with_eval_id_needs_no_config(tmp_path, merged) -> None:
    db = tmp_path / "recording-abc.sqlite"
    db.touch()

    cli.cmd_merge(_args(output_dir=str(tmp_path), eval_id="abc"))

    assert merged == [(db, tmp_path.resolve())]


def test_output_dir_without_eval_id_merges_every_db(tmp_path, merged) -> None:
    (tmp_path / "recording-a.sqlite").touch()
    (tmp_path / "recording-b.sqlite").touch()
    (tmp_path / "unrelated.sqlite").touch()

    cli.cmd_merge(_args(output_dir=str(tmp_path)))

    assert [db.name for db, _ in merged] == ["recording-a.sqlite", "recording-b.sqlite"]


def test_output_dir_overrides_config_output_dir(tmp_path, monkeypatch, merged) -> None:
    override = tmp_path / "override"
    override.mkdir()
    (override / "recording-abc.sqlite").touch()
    monkeypatch.setattr(cli, "_load_config", lambda _path: {"output_dir": str(tmp_path / "from-config")})

    cli.cmd_merge(_args(config="eval.yaml", output_dir=str(override), eval_id="abc"))

    assert merged == [(override / "recording-abc.sqlite", override.resolve())]


def test_config_with_eval_id_still_resolves_db_from_config_output_dir(tmp_path, monkeypatch, merged) -> None:
    (tmp_path / "recording-abc.sqlite").touch()
    monkeypatch.setattr(cli, "_load_config", lambda _path: {"output_dir": str(tmp_path)})

    cli.cmd_merge(_args(config="eval.yaml", eval_id="abc"))

    assert merged == [(tmp_path / "recording-abc.sqlite", tmp_path.resolve())]


def test_db_path_still_wins_and_defaults_output_dir_to_its_parent(tmp_path, merged) -> None:
    db = tmp_path / "recording-abc.sqlite"
    db.touch()

    cli.cmd_merge(_args(db=str(db)))

    assert merged == [(db, tmp_path.resolve())]


def test_no_source_arguments_exits_with_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.cmd_merge(_args())

    assert exc.value.code == 1
    assert "--output-dir" in capsys.readouterr().err


def test_output_dir_without_any_recording_db_exits(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.cmd_merge(_args(output_dir=str(tmp_path)))

    assert exc.value.code == 1
    assert "no recording-*.sqlite" in capsys.readouterr().err
