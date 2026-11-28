"""Unit tests for per-view episode video on ``EpisodeRecorder`` (``recording.record_video``).

Covers the harness-side half of multi-camera recording: ``record_video(frame, view=...)`` fans one
episode out to one mp4 per view, named ``<filename_stem>_<view>_<status>.mp4``. Everything here uses
the public recorder API only — no benchmark adapter is imported, so the cases run on the host without
a simulator (see ``skills/add-benchmark/SKILL.md``).

Frames are 4×4 RGB ``uint8`` stubs: small enough for imageio's ffmpeg writer, real enough that the
produced mp4 has a verifiable frame count.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vla_eval.recording import (
    EpisodeRecorder,
    EpisodeStatus,
    NullEpisodeRecorder,
    RecordingStore,
    normalize_video_views,
    view_filename_stem,
)


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _recorder(tmp_path: Path, **overrides: Any) -> EpisodeRecorder:
    """Build an ``EpisodeRecorder`` with stable test defaults.

    Args:
        tmp_path: Output directory for the mp4s and the SQLite recording.
        **overrides: ``EpisodeRecorder`` keyword arguments to replace the defaults.

    Returns:
        A recorder with video on, one step field, and a ``{status}``-bearing stem.
    """
    store = RecordingStore(tmp_path / "recording.sqlite")
    kwargs: dict[str, Any] = {
        "store": store,
        "sid": "s",
        "eid": "e",
        "eval_id": "ev",
        "output_dir": tmp_path,
        "filename_stem": "task{task_idx:04d}_ep{episode_id:04d}_{status}",
        "context": {"task_idx": 7, "episode_id": 3},
        "record_video": True,
    }
    kwargs.update(overrides)
    return EpisodeRecorder(**kwargs)


def _close(rec: EpisodeRecorder, status: EpisodeStatus = "success") -> None:
    """Close a recorder the way the orchestrator does, then release its SQLite store."""
    rec.close(status=status, metrics={"success": status == "success"})
    rec._store.close()  # noqa: SLF001 — test helper, no public close on the recorder


def _mp4s(tmp_path: Path) -> list[str]:
    """Return the sorted names of the mp4s in ``tmp_path`` (working files excluded)."""
    return sorted(p.name for p in tmp_path.glob("*.mp4"))


def _count_frames(path: Path) -> int:
    """Count the frames of an mp4 via imageio's reader."""
    import imageio

    with imageio.get_reader(str(path)) as reader:
        return sum(1 for _ in reader)  # ty: ignore[not-iterable]


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def test_view_filename_stem_inserts_before_status() -> None:
    stem = "{benchmark_safe_name}/task{task_idx:04d}_ep{episode_id:04d}_{status}"
    assert view_filename_stem(stem, "cam_head") == (
        "{benchmark_safe_name}/task{task_idx:04d}_ep{episode_id:04d}_cam_head_{status}"
    )


def test_view_filename_stem_without_status_appends_before_extension() -> None:
    assert view_filename_stem("ep{episode_id:04d}.mp4", "left") == "ep{episode_id:04d}_left.mp4"
    assert view_filename_stem("ep{episode_id:04d}", "left") == "ep{episode_id:04d}_left"


def test_view_filename_stem_rejects_unsafe_label() -> None:
    for label in ("../head", "a b", "head/../../x", ""):
        with pytest.raises(ValueError):
            view_filename_stem("ep_{status}", label)


def test_normalize_video_views_accepts_lists_and_mappings() -> None:
    assert normalize_video_views(None) == {}
    assert normalize_video_views([]) == {}
    assert normalize_video_views(["cam_head"]) == {"cam_head": "cam_head"}
    assert normalize_video_views({"cam_head": "head", "cam_left_wrist": "left"}) == {
        "cam_head": "head",
        "cam_left_wrist": "left",
    }


def test_normalize_video_views_rejects_duplicates_and_bad_labels() -> None:
    with pytest.raises(ValueError):
        normalize_video_views(["cam_head", "cam_head"])
    with pytest.raises(ValueError):
        normalize_video_views({"cam_head": "head", "cam_left_wrist": "head"})
    with pytest.raises(ValueError):
        normalize_video_views({"cam_head": "../head"})


# ---------------------------------------------------------------------------
# record_video(view=...)
# ---------------------------------------------------------------------------


def test_one_file_per_view_with_the_episode_status(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    for _ in range(3):
        rec.record_video(_frame(), view="cam_head")
        rec.record_video(_frame(), view="cam_left_wrist")
    _close(rec, status="fail")

    assert _mp4s(tmp_path) == ["task0007_ep0003_cam_head_fail.mp4", "task0007_ep0003_cam_left_wrist_fail.mp4"]
    assert _count_frames(tmp_path / "task0007_ep0003_cam_head_fail.mp4") == 3


def test_no_view_keeps_the_single_stream(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    rec.record_video(_frame())
    _close(rec)

    assert _mp4s(tmp_path) == ["task0007_ep0003_success.mp4"]


def test_missing_camera_warns_once_and_writes_no_substitute(tmp_path: Path, caplog: Any) -> None:
    """A view the observation never carries gets no file; the others are unaffected."""
    rec = _recorder(tmp_path)
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            rec.record_video(_frame(), view="cam_head")  # cam_left_wrist never arrives
    _close(rec)

    assert _mp4s(tmp_path) == ["task0007_ep0003_cam_head_success.mp4"]
    assert not any("cam_left_wrist" in message for message in caplog.messages)


def test_zero_frame_episode_writes_nothing(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    _close(rec)

    assert _mp4s(tmp_path) == []


def test_recording_off_is_a_noop(tmp_path: Path) -> None:
    rec = _recorder(tmp_path, record_video=False)
    rec.record_video(_frame())
    rec.record_video(_frame(), view="cam_head")
    _close(rec)

    assert _mp4s(tmp_path) == []


def test_null_recorder_accepts_a_view(tmp_path: Path) -> None:
    rec = NullEpisodeRecorder()
    rec.record_video(_frame(), view="cam_head")
    rec.record_video(_frame())
    rec.close(status="success", metrics={})

    assert _mp4s(tmp_path) == []


def test_duplicate_target_is_logged_and_other_views_still_save(tmp_path: Path) -> None:
    """A view whose file already exists must not cost the episode's other views."""
    (tmp_path / "task0007_ep0003_cam_head_success.mp4").write_bytes(b"existing")
    rec = _recorder(tmp_path)
    rec.record_video(_frame(), view="cam_head")
    rec.record_video(_frame(), view="cam_left_wrist")
    _close(rec)

    assert (tmp_path / "task0007_ep0003_cam_head_success.mp4").read_bytes() == b"existing"
    assert _count_frames(tmp_path / "task0007_ep0003_cam_left_wrist_success.mp4") == 1


def test_unusable_label_is_warned_and_skipped(tmp_path: Path, caplog: Any) -> None:
    """A label that cannot reach the filesystem is rejected by the recorder, not the benchmark."""
    rec = _recorder(tmp_path)
    with caplog.at_level(logging.WARNING):
        rec.record_video(_frame(), view="../escape")
        rec.record_video(_frame(), view="cam_head")
    _close(rec)

    assert _mp4s(tmp_path) == ["task0007_ep0003_cam_head_success.mp4"]
    assert any("unusable" in message for message in caplog.messages)


def test_views_are_independent_across_episodes(tmp_path: Path) -> None:
    """Each episode's views are saved and the next episode opens fresh writers."""
    for episode_id in (3, 4):
        rec = _recorder(tmp_path, eid=f"e{episode_id}", context={"task_idx": 7, "episode_id": episode_id})
        rec.record_video(_frame(), view="cam_head")
        _close(rec)

    assert _mp4s(tmp_path) == ["task0007_ep0003_cam_head_success.mp4", "task0007_ep0004_cam_head_success.mp4"]
