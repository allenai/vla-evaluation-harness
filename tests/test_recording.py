"""Unit tests for ``vla_eval.benchmarks.recording.EpisodeVideoRecorder``.

Exercises the full lifecycle (start → record → save / discard), filename
templating (str + callable), required-context auto-derivation,
required-context validation, error paths (writer-open failure, encode
failure, missing context key at save), state isolation between episodes,
and collision detection at save time.

Frames are 4×4 RGB ``uint8`` stubs — small enough that ``imageio``'s
ffmpeg writer copes without external codec setup, but real enough that
the produced mp4 has a verifiable framecount.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from vla_eval.benchmarks.recording import EpisodeVideoRecorder, _fields_from_template


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _count_frames(path: Path) -> int:
    # imageio's `_BaseReaderWriter` type stub omits __iter__; the concrete
    # Reader returned for mp4 is iterable in practice.
    import imageio

    with imageio.get_reader(str(path)) as r:
        return sum(1 for _ in r)  # ty: ignore[not-iterable]


def _rec(tmp_path: Path, **overrides: Any) -> EpisodeVideoRecorder:
    """Construct a recorder with stable test-suite defaults.

    The recorder itself has no defaults for ``filename`` (every benchmark
    spells it out).  Tests don't need to repeat that boilerplate, so
    this helper picks a ``task_name`` + ``episode_idx`` template and lets
    auto-derivation discover the required keys.
    """
    kwargs: dict[str, Any] = {
        "output_dir": tmp_path,
        "filename": "{task_name}_ep{episode_idx:04d}_{status}.mp4",
    }
    kwargs.update(overrides)
    return EpisodeVideoRecorder(**kwargs)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_save_writes_mp4_with_correct_framecount(tmp_path: Path) -> None:
    rec = _rec(tmp_path, fps=10)
    rec.start({"task_name": "PickCube", "episode_idx": 0})
    for _ in range(5):
        rec.record(_frame())
    final = rec.save(status="success")

    assert final is not None
    assert final == tmp_path / "PickCube_ep0000_success.mp4"
    assert final.exists()
    assert _count_frames(final) == 5


def test_save_uses_status_in_filename(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.start({"task_name": "T", "episode_idx": 7})
    rec.record(_frame())
    final = rec.save(status="fail")
    assert final is not None
    assert final == tmp_path / "T_ep0007_fail.mp4"
    assert final.exists()


def test_active_flag_tracks_lifecycle(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    assert rec.active is False
    rec.start({"task_name": "T", "episode_idx": 0})
    assert rec.active is True
    rec.save()
    assert rec.active is False


def test_consecutive_episodes_each_produce_their_own_file(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    for ep in range(3):
        rec.start({"task_name": "T", "episode_idx": ep})
        rec.record(_frame())
        rec.record(_frame())
        final = rec.save(status="success")
        assert final is not None
        assert final == tmp_path / f"T_ep{ep:04d}_success.mp4"
        assert final.exists()
    assert sorted(p.name for p in tmp_path.glob("T_ep*.mp4")) == [
        "T_ep0000_success.mp4",
        "T_ep0001_success.mp4",
        "T_ep0002_success.mp4",
    ]


# ---------------------------------------------------------------------------
# Filename templating
# ---------------------------------------------------------------------------


def test_filename_template_with_subdirectories(tmp_path: Path) -> None:
    rec = EpisodeVideoRecorder(
        output_dir=tmp_path,
        filename="{suite}/{task_name}_ep{episode_idx:04d}_{status}.mp4",
    )
    rec.start({"suite": "Counting", "task_name": "PickX", "episode_idx": 12})
    rec.record(_frame())
    final = rec.save(status="success")
    assert final is not None
    assert final == tmp_path / "Counting" / "PickX_ep0012_success.mp4"
    assert final.exists()


def test_filename_callable_requires_explicit_required_context(tmp_path: Path) -> None:
    def naming(ctx: Mapping[str, Any]) -> str:
        return f"{ctx['run_id']}-{ctx['episode_idx']}-{ctx['status']}.mp4"

    # Callable + required_context=None → ValueError at construction.
    with pytest.raises(ValueError, match="required_context must be specified when filename is a callable"):
        EpisodeVideoRecorder(output_dir=tmp_path, filename=naming)

    # Explicit required_context unblocks it.
    rec = EpisodeVideoRecorder(
        output_dir=tmp_path,
        filename=naming,
        required_context=("run_id", "episode_idx"),
    )
    rec.start({"run_id": "abc", "episode_idx": 3})
    rec.record(_frame())
    final = rec.save(status="ok")
    assert final == tmp_path / "abc-3-ok.mp4"


def test_save_with_template_key_not_in_required_context_is_handled(tmp_path: Path) -> None:
    # An explicit required_context is allowed to be a SUBSET of template keys —
    # callers may want some keys to be optional (e.g. `seed`).  Auto-derivation
    # would have included `seed`, but here the caller deliberately overrides
    # to make `seed` optional, so start() doesn't enforce it.  When a template
    # key is missing at save() time, resolution should fail gracefully rather
    # than raise.
    rec = EpisodeVideoRecorder(
        output_dir=tmp_path,
        filename="{task_name}_{seed}_{status}.mp4",
        required_context=("task_name",),  # explicit subset
    )
    rec.start({"task_name": "T"})  # `seed` missing
    rec.record(_frame())
    final = rec.save(status="success")
    assert final is None
    assert list(tmp_path.glob(".recorder-*.mp4")) == []


# ---------------------------------------------------------------------------
# required_context auto-derivation
# ---------------------------------------------------------------------------


def test_required_context_auto_derived_from_template(tmp_path: Path) -> None:
    rec = EpisodeVideoRecorder(
        output_dir=tmp_path,
        filename="{a}_{b:04d}_{status}.mp4",
    )
    # Auto-derivation strips `:format_spec` and excludes `status`.
    assert rec._required_context == ("a", "b")  # type: ignore[attr-defined]
    # And the validation actually applies:
    with pytest.raises(ValueError, match="missing required context keys"):
        rec.start({"a": "x"})  # `b` missing


def test_fields_from_template_helper() -> None:
    # Direct unit test of the helper since it's the engine for auto-derivation.
    assert _fields_from_template("{a}_{b}_{status}.mp4") == ("a", "b")
    assert _fields_from_template("{episode_idx:04d}_{status}.mp4") == ("episode_idx",)
    assert _fields_from_template("{suite}/{task}_ep{i:04d}.mp4") == ("suite", "task", "i")
    assert _fields_from_template("plain.mp4") == ()
    # Duplicates collapsed, order preserved.
    assert _fields_from_template("{a}_{b}_{a}_{status}.mp4") == ("a", "b")
    # Attribute/indexing access — bare name only.
    assert _fields_from_template("{ctx.task}_{arr[0]}.mp4") == ("ctx", "arr")


# ---------------------------------------------------------------------------
# Validation / error paths
# ---------------------------------------------------------------------------


def test_start_missing_required_context_raises(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    with pytest.raises(ValueError, match="missing required context keys"):
        rec.start({"task_name": "T"})  # episode_idx missing
    assert rec.active is False


def test_record_before_start_is_noop(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.record(_frame())  # must not raise
    assert rec.active is False


def test_save_before_start_returns_none(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    assert rec.save() is None


def test_discard_before_start_is_noop(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.discard()  # must not raise
    assert rec.active is False


def test_writer_open_failure_leaves_recorder_inactive(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    with patch("imageio.get_writer", side_effect=RuntimeError("nope")):
        rec.start({"task_name": "T", "episode_idx": 0})
    assert rec.active is False
    # No leftover working file.
    assert list(tmp_path.glob(".recorder-*.mp4")) == []
    # Subsequent record/save are no-ops.
    rec.record(_frame())
    assert rec.save() is None


# ---------------------------------------------------------------------------
# Collision detection at save
# ---------------------------------------------------------------------------


def test_save_raises_on_collision_by_default(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    rec.save(status="success")  # writes T_ep0000_success.mp4

    # Programmer error: same context, same status, second time around.
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    with pytest.raises(FileExistsError, match="already exists"):
        rec.save(status="success")
    # Original file is untouched, working file remains for manual recovery.
    assert (tmp_path / "T_ep0000_success.mp4").exists()
    assert len(list(tmp_path.glob(".recorder-*.mp4"))) == 1


def test_save_overwrites_when_overwrite_true(tmp_path: Path) -> None:
    rec = _rec(tmp_path, overwrite=True)
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    rec.record(_frame())
    first = rec.save(status="success")
    assert first is not None
    assert _count_frames(first) == 2

    rec.start({"task_name": "T", "episode_idx": 0})
    for _ in range(5):
        rec.record(_frame())
    second = rec.save(status="success")
    assert second is not None
    assert second == first  # same final path
    assert _count_frames(second) == 5  # but new content


# ---------------------------------------------------------------------------
# Mid-episode interruption / cleanup
# ---------------------------------------------------------------------------


def test_start_again_without_save_discards_prior_episode(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    # Simulate orchestrator skipping save() / discard() and starting next ep:
    rec.start({"task_name": "T", "episode_idx": 1})
    rec.record(_frame())
    final = rec.save(status="success")
    assert final is not None
    assert final == tmp_path / "T_ep0001_success.mp4"
    # Only ep1 mp4 should exist; ep0's working file was cleaned up.
    mp4s = sorted(p.name for p in tmp_path.glob("*.mp4"))
    assert mp4s == ["T_ep0001_success.mp4"]
    assert list(tmp_path.glob(".recorder-*.mp4")) == []


def test_discard_cleans_up_working_file(tmp_path: Path) -> None:
    rec = _rec(tmp_path)
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    rec.discard()
    assert rec.active is False
    assert list(tmp_path.glob(".recorder-*.mp4")) == []
    assert list(tmp_path.glob("*.mp4")) == []


# ---------------------------------------------------------------------------
# Output dir created if missing
# ---------------------------------------------------------------------------


def test_output_dir_created_lazily(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "videos"
    assert not target.exists()
    rec = _rec(target)
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    final = rec.save()
    assert final is not None
    assert final.exists()
    assert target.is_dir()


def test_str_path_accepted(tmp_path: Path) -> None:
    rec = _rec(tmp_path, output_dir=str(tmp_path))
    rec.start({"task_name": "T", "episode_idx": 0})
    rec.record(_frame())
    final = rec.save()
    assert final is not None
    assert final.exists()
    assert os.fspath(final).startswith(str(tmp_path))
