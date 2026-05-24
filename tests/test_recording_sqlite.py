"""Tests for the SQLite recording path: store, EpisodeRecorder, StepRecorder,
and ``vla-eval merge``.

The most important test here is :func:`test_multi_writer_field_union` — it
exercises the contract that two processes (orchestrator + model server, or
two shards) can independently insert step rows for the same
``(sid, eid, step_id)`` with disjoint or overlapping field sets and have
the daemon-less store merge them atomically via ``json_patch`` UPSERT.
That contract is the entire reason we chose SQLite over a per-file
filesystem layout.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from vla_eval.recording import (
    EpisodeRecorder,
    NullEpisodeRecorder,
    RecordingStore,
    StepRecorder,
    db_path_for_eval,
)
from vla_eval.results.merge import merge_db, merge_eval


# ---------------------------------------------------------------------------
# Schema / store
# ---------------------------------------------------------------------------


def test_store_schema_idempotent_across_processes(tmp_path: Path) -> None:
    """Two writers open the same DB; the second's CREATE TABLE IF NOT EXISTS is a no-op."""
    db = tmp_path / "recording.sqlite"
    s1 = RecordingStore(db)
    s2 = RecordingStore(db)
    try:
        s1.upsert_eval_metadata("ev1", "demo", {"benchmark": "demo"})
        s2.upsert_eval_metadata("ev1", "demo", {"benchmark": "demo-different"})  # ignored: first wins
    finally:
        s1.close()
        s2.close()

    conn = sqlite3.connect(str(db))
    rows = list(conn.execute("SELECT eval_id, safe_name, metadata FROM eval_metadata"))
    conn.close()
    assert len(rows) == 1
    assert json.loads(rows[0][2])["benchmark"] == "demo"


def test_store_step_upsert_field_union(tmp_path: Path) -> None:
    """``json_patch`` UPSERT must field-union, not overwrite the entire row."""
    db = tmp_path / "recording.sqlite"
    s = RecordingStore(db)
    try:
        s.upsert_step_rows(
            "s",
            "e",
            {
                0: {"reward": 0.5, "task": "pick"},
                1: {"reward": 0.6},
            },
        )
        # Second writer adds different fields for step 0; merges.
        s.upsert_step_rows(
            "s",
            "e",
            {
                0: {"inference_ms": 12.3, "task": "pick_overridden"},
                2: {"reward": 0.9},
            },
        )
    finally:
        s.close()

    conn = sqlite3.connect(str(db))
    rows = dict(conn.execute("SELECT step_id, fields FROM step_rows WHERE sid='s' AND eid='e'"))
    conn.close()

    step0 = json.loads(rows[0])
    # Disjoint fields preserved, overlapping field overwritten (last-writer wins per-key).
    assert step0 == {"reward": 0.5, "task": "pick_overridden", "inference_ms": 12.3}
    assert json.loads(rows[1]) == {"reward": 0.6}
    assert json.loads(rows[2]) == {"reward": 0.9}


def test_step_rows_handle_numpy(tmp_path: Path) -> None:
    """numpy arrays/scalars must round-trip as JSON arrays/numbers, not as strings.

    Regression: an earlier draft used ``json.dumps(..., default=str)`` which
    encoded ``np.array([1.5, 2.5])`` as the unparseable string ``"[1.5 2.5]"``.
    """
    db = tmp_path / "recording.sqlite"
    s = RecordingStore(db)
    try:
        s.upsert_step_rows(
            "s",
            "e",
            {
                0: {
                    "robot_state": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "reward": np.float32(0.75),
                    "step_count": np.int64(7),
                },
            },
        )
    finally:
        s.close()

    conn = sqlite3.connect(str(db))
    fields = json.loads(conn.execute("SELECT fields FROM step_rows WHERE sid='s' AND eid='e'").fetchone()[0])
    conn.close()
    assert fields["robot_state"] == pytest.approx([0.1, 0.2, 0.3])
    assert fields["reward"] == pytest.approx(0.75)
    assert fields["step_count"] == 7


def test_jsonl_path_is_relative_to_db_dir(tmp_path: Path) -> None:
    """Regression: ``jsonl_path`` must be stored relative to the DB dir so that
    ``vla-eval merge`` resolves it correctly when the run happened in Docker
    (output_dir=/workspace/results) but the merge happens on the host
    (output_dir=/mnt/host/...).
    """
    db_dir = tmp_path / "run"
    episodes_dir = db_dir / "episodes"
    db_dir.mkdir()
    episodes_dir.mkdir()
    store = RecordingStore(db_dir / "recording.sqlite")
    rec = EpisodeRecorder(
        store=store,
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=episodes_dir,
        filename_stem="task_{status}",
        context={},
        record_video=False,
    )
    rec.close(status="success", metrics={"success": True}, task_name="t", episode_id=0, steps=0)
    store.close()

    conn = sqlite3.connect(str(db_dir / "recording.sqlite"))
    (jsonl_path,) = conn.execute("SELECT jsonl_path FROM episode_results").fetchone()
    conn.close()
    assert jsonl_path == "episodes/task_success.jsonl"


# ---------------------------------------------------------------------------
# Multi-writer: orchestrator + model server -style fan-in
# ---------------------------------------------------------------------------


def test_multi_writer_field_union(tmp_path: Path) -> None:
    """Two *separate* RecordingStore instances on the same DB independently write
    step rows for the same (sid, eid, step_id) — they must field-union.

    Simulates the real production topology: the orchestrator's EpisodeRecorder
    in one Python process and the model server's StepRecorder in another.
    """
    db = tmp_path / "recording.sqlite"

    # Pretend writer A is the orchestrator (benchmark side).
    a = RecordingStore(db)
    a.upsert_step_rows(
        "s",
        "e",
        {
            0: {"reward": 0.1, "robot_state": [0.1, 0.2, 0.3]},
            1: {"reward": 0.2, "robot_state": [0.4, 0.5, 0.6]},
            2: {"reward": 0.3, "robot_state": [0.7, 0.8, 0.9]},
        },
    )

    # Pretend writer B is the model server (inference-trace side).
    b = RecordingStore(db)
    b.upsert_step_rows(
        "s",
        "e",
        {
            0: {"inference_ms": 11.1, "action_logits": [0.0, 1.0]},
            1: {"inference_ms": 12.2, "action_logits": [0.5, 0.5]},
            2: {"inference_ms": 13.3, "action_logits": [1.0, 0.0]},
        },
    )

    a.close()
    b.close()

    conn = sqlite3.connect(str(db))
    rows = {
        step_id: json.loads(fields)
        for step_id, fields in conn.execute(
            "SELECT step_id, fields FROM step_rows WHERE sid='s' AND eid='e' ORDER BY step_id"
        )
    }
    conn.close()

    assert rows[0] == {
        "reward": 0.1,
        "robot_state": [0.1, 0.2, 0.3],
        "inference_ms": 11.1,
        "action_logits": [0.0, 1.0],
    }
    assert rows[1]["reward"] == pytest.approx(0.2)
    assert rows[1]["inference_ms"] == pytest.approx(12.2)
    assert rows[2]["action_logits"] == [1.0, 0.0]


def test_step_recorder_external_caller(tmp_path: Path) -> None:
    """StepRecorder is the convenience API model-server code (e.g. reflex-train) uses.

    It opens its own RecordingStore against the DB path the harness forwards in
    EPISODE_START, buffers rows in memory, and flushes them in a single
    transaction on ``close()``.
    """
    db = tmp_path / "recording.sqlite"
    # Harness side: open the DB and put an eval row (so the schema exists).
    primary = RecordingStore(db)
    primary.upsert_eval_metadata("ev", "demo", {"benchmark": "demo", "metric_keys": {"success": "mean"}})
    primary.upsert_step_rows("s", "e", {0: {"reward": 0.5}, 1: {"reward": 0.7}})
    primary.close()

    # Model-server side opens a StepRecorder on the same path.
    with StepRecorder(db, sid="s", eid="e") as rec:
        rec.record({"step": 0, "inference_ms": 9.9})
        rec.record({"step": 1, "inference_ms": 10.5})

    conn = sqlite3.connect(str(db))
    rows = {
        step_id: json.loads(fields)
        for step_id, fields in conn.execute("SELECT step_id, fields FROM step_rows WHERE sid='s' AND eid='e'")
    }
    conn.close()
    assert rows[0] == {"reward": 0.5, "inference_ms": 9.9}
    assert rows[1] == {"reward": 0.7, "inference_ms": 10.5}


# ---------------------------------------------------------------------------
# EpisodeRecorder
# ---------------------------------------------------------------------------


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_null_recorder_is_strict_noop(tmp_path: Path) -> None:
    rec = NullEpisodeRecorder()
    rec.record_video(_frame())
    rec.record_step(reward=1.0)
    rec.close(status="success", metrics={"success": True})
    rec.close(status="success", metrics={})  # idempotent
    assert rec.is_active is False
    assert rec.sid == ""
    assert rec.eid == ""
    assert rec.db_path == ""


def test_episode_recorder_close_writes_steps_and_result(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "recording.sqlite")
    store.upsert_eval_metadata("ev", "demo", {"benchmark": "demo"})
    rec = EpisodeRecorder(
        store=store,
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="{env_id}_ep{episode_idx:04d}_{status}",
        context={"env_id": "demo", "episode_idx": 3},
        record_video=False,
    )
    rec.record_step(reward=0.1)
    rec.record_step(reward=0.2)
    rec.record_step(reward=0.3)
    rec.close(
        status="success",
        metrics={"success": True},
        task_name="demo_task",
        episode_id=3,
        steps=3,
        elapsed_sec=0.42,
    )
    store.close()

    conn = sqlite3.connect(str(tmp_path / "recording.sqlite"))
    er = conn.execute("SELECT task_name, status, jsonl_path FROM episode_results").fetchone()
    # jsonl_path is stored relative to the SQLite directory so vla-eval merge
    # works regardless of host-vs-container path differences.
    assert er == ("demo_task", "success", "demo_ep0003_success.jsonl")
    step_rows = [json.loads(f) for (_, f) in conn.execute("SELECT step_id, fields FROM step_rows ORDER BY step_id")]
    assert [r["reward"] for r in step_rows] == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]
    conn.close()


def test_episode_recorder_close_idempotent(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "recording.sqlite")
    rec = EpisodeRecorder(
        store=store,
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="ep_{status}",
        context={},
        record_video=False,
    )
    rec.close(status="success", metrics={})
    rec.close(status="success", metrics={})  # second call no-op
    store.close()
    conn = sqlite3.connect(str(tmp_path / "recording.sqlite"))
    count = conn.execute("SELECT COUNT(*) FROM episode_results").fetchone()[0]
    conn.close()
    assert count == 1


# ---------------------------------------------------------------------------
# vla-eval merge
# ---------------------------------------------------------------------------


def _write_sample_db(tmp_path: Path) -> tuple[Path, str]:
    """Populate a recording DB with 1 eval + 2 successful + 1 failed episode."""
    db = db_path_for_eval(tmp_path, "ev")
    store = RecordingStore(db)
    store.upsert_eval_metadata(
        "ev",
        "demo_bench",
        {
            "benchmark": "demo_bench",
            "mode": "sync",
            "config": {"params": {"seed": 7}},
            "metric_keys": {"success": "mean"},
            "harness_version": "test",
            "server_info": {"model_server": "EchoServer"},
        },
    )
    # Three episodes.
    for i, status in enumerate(["success", "success", "fail"]):
        sid = "shard-0"
        eid = f"ep-{i}"
        store.upsert_step_rows(sid, eid, {0: {"reward": float(i)}, 1: {"reward": float(i) + 0.1}})
        store.upsert_episode_result(
            sid=sid,
            eid=eid,
            eval_id="ev",
            task_name="taskA",
            episode_id=i,
            status=status,
            metrics={"success": status == "success"},
            steps=2,
            elapsed_sec=0.1,
            context={"env_id": "demo", "episode_idx": i},
            jsonl_path=str(tmp_path / f"demo_ep{i:04d}_{status}.jsonl"),
            failure_reason=None,
            failure_detail=None,
        )
    store.close()
    return db, "ev"


def test_merge_db_emits_per_episode_jsonl_and_aggregate(tmp_path: Path) -> None:
    db, eval_id = _write_sample_db(tmp_path)
    aggregates = merge_db(db, tmp_path)

    # Per-episode jsonls
    for i, status in enumerate(["success", "success", "fail"]):
        path = tmp_path / f"demo_ep{i:04d}_{status}.jsonl"
        assert path.exists(), f"missing {path}"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["step"] for r in rows] == [0, 1]
        assert rows[0]["reward"] == pytest.approx(float(i))

    # Aggregate
    assert len(aggregates) == 1
    body = aggregates[0]
    assert body["benchmark"] == "demo_bench"
    assert body["mode"] == "sync"
    assert body["seed"] == 7
    assert body["harness_version"] == "test"
    assert body["server_info"] == {"model_server": "EchoServer"}
    assert body["metric_keys"] == {"success": "mean"}
    assert body["mean_success"] == pytest.approx(2 / 3, abs=1e-4)
    assert [t["task"] for t in body["tasks"]] == ["taskA"]
    assert body["tasks"][0]["mean_success"] == pytest.approx(2 / 3, abs=1e-4)
    assert body["tasks"][0]["num_episodes"] == 3
    # Aggregate JSON also written
    agg_path = tmp_path / "demo_bench_aggregate.json"
    assert agg_path.exists()
    on_disk = json.loads(agg_path.read_text())
    assert on_disk["benchmark"] == "demo_bench"


def test_merge_eval_wrapper(tmp_path: Path) -> None:
    db, eval_id = _write_sample_db(tmp_path)
    aggregates = merge_eval(tmp_path, eval_id)
    assert len(aggregates) == 1
    assert aggregates[0]["eval_id"] == "ev"


def test_merge_handles_missing_jsonl_path(tmp_path: Path) -> None:
    """Episode without a step buffer still produces an aggregate row, no jsonl."""
    db = db_path_for_eval(tmp_path, "ev")
    store = RecordingStore(db)
    store.upsert_eval_metadata("ev", "demo", {"benchmark": "demo", "metric_keys": {"success": "mean"}})
    # No step rows at all (e.g. benchmark.reset raised before first step).
    store.upsert_episode_result(
        sid="s",
        eid="e",
        eval_id="ev",
        task_name="t",
        episode_id=0,
        status="error",
        metrics={"success": False},
        steps=0,
        elapsed_sec=0.0,
        context={"env_id": "x", "episode_idx": 0},
        jsonl_path=str(tmp_path / "x_ep0000_error.jsonl"),
        failure_reason="server_unreachable",
        failure_detail="boom",
    )
    store.close()

    aggregates = merge_db(db, tmp_path)
    assert aggregates[0]["mean_success"] == 0.0
    # No step rows → no per-episode jsonl was written.
    assert not (tmp_path / "x_ep0000_error.jsonl").exists()
    # But the aggregate captures the failure.
    body = aggregates[0]
    failed = body["tasks"][0]["episodes"][0]
    assert failed["failure_reason"] == "server_unreachable"


# --------------------------------------------------------------------------
# Host translation (VLA_EVAL_HOST_OUTPUT_DIR) for cross-container db_path
# --------------------------------------------------------------------------


def test_host_translate_no_env(monkeypatch, tmp_path):
    """Env unset → passthrough."""
    monkeypatch.delenv("VLA_EVAL_HOST_OUTPUT_DIR", raising=False)
    from vla_eval.recording import _host_translate

    p = tmp_path / "recording-x.sqlite"
    assert _host_translate(p) == p


def test_host_translate_rewrites_container_prefix(monkeypatch, tmp_path):
    """Env set + container-prefix path → host root rewrite."""
    monkeypatch.setenv("VLA_EVAL_HOST_OUTPUT_DIR", str(tmp_path))
    from vla_eval.recording import _host_translate

    container_path = Path("/workspace/results/recording-abc.sqlite")
    out = _host_translate(container_path)
    assert out == tmp_path / "recording-abc.sqlite"


def test_host_translate_leaves_unrelated_path_alone(monkeypatch, tmp_path):
    """Env set + path outside container prefix → passthrough."""
    monkeypatch.setenv("VLA_EVAL_HOST_OUTPUT_DIR", str(tmp_path))
    from vla_eval.recording import _host_translate

    p = Path("/some/other/place/recording-y.sqlite")
    assert _host_translate(p) == p


def test_recording_store_chmods_db_world_writable(tmp_path):
    """Main + WAL + SHM are mode 666 so external (different-uid) writers can upsert."""
    from vla_eval.recording import RecordingStore

    db = tmp_path / "rec.sqlite"
    store = RecordingStore(db)
    try:
        for suffix in ("", "-wal", "-shm"):
            f = tmp_path / ("rec.sqlite" + suffix)
            assert f.exists(), f"missing {f.name}"
            mode = f.stat().st_mode & 0o777
            assert mode == 0o666, f"{f.name}: expected 0o666, got {oct(mode)}"
    finally:
        store.close()
