"""Tests for ResultCollector."""

from __future__ import annotations

import json

import pytest

from vla_eval.results.collector import ResultCollector
from vla_eval.results.merge import load_shard_files, merge_shards


def test_result_collector():
    collector = ResultCollector("test_bench", mode="sync", metric_keys={"success": "mean"})
    collector.record("task_a", {"episode_id": 0, "metrics": {"success": True}, "steps": 10})
    collector.record("task_a", {"episode_id": 1, "metrics": {"success": False}, "steps": 20})
    collector.record("task_b", {"episode_id": 0, "metrics": {"success": True}, "steps": 5})

    result = collector.get_benchmark_result()
    assert result["benchmark"] == "test_bench"
    assert result.get("mean_success") == pytest.approx(2 / 3, abs=0.001)

    task_a = collector.get_task_result("task_a")
    assert task_a.get("mean_success") == pytest.approx(0.5)
    assert task_a["avg_steps"] == pytest.approx(15.0)


def test_empty_collector():
    collector = ResultCollector("empty_bench")
    result = collector.get_benchmark_result()
    assert result["benchmark"] == "empty_bench"
    assert result["tasks"] == []


def test_to_json_returns_valid_json():
    collector = ResultCollector("json_bench", mode="sync", metric_keys={"success": "mean"})
    collector.record("t1", {"episode_id": 0, "metrics": {"success": True}, "steps": 5})
    text = collector.to_json()
    parsed = json.loads(text)
    assert parsed["benchmark"] == "json_bench"
    assert isinstance(parsed["tasks"], list)


def test_errors_included_in_metrics():
    """All episodes count toward metrics. Errors are reported but not excluded."""
    collector = ResultCollector("bench", mode="sync", metric_keys={"success": "mean"})
    collector.record("task_a", {"episode_id": 0, "metrics": {"success": True}, "steps": 10})
    collector.record("task_a", {"episode_id": 1, "metrics": {"success": False}, "steps": 20})
    collector.record("task_a", {"episode_id": 2, "metrics": {"success": False}, "failure_reason": "timeout"})
    collector.record("task_a", {"episode_id": 3, "metrics": {"success": False}, "failure_reason": "exception"})

    task = collector.get_task_result("task_a")
    # Success rate = 1/4 (all episodes count)
    assert task.get("mean_success") == pytest.approx(0.25)
    assert task["num_episodes"] == 4
    assert task["num_errors"] == 2  # episodes with any failure_reason

    result = collector.get_benchmark_result()
    assert result.get("mean_success") == pytest.approx(0.25)


def test_num_errors_absent_when_no_errors():
    """When no errors occur, num_errors should be absent from task result."""
    collector = ResultCollector("bench", mode="sync", metric_keys={"success": "mean"})
    collector.record("task_a", {"episode_id": 0, "metrics": {"success": True}, "steps": 5})
    collector.record("task_a", {"episode_id": 1, "metrics": {"success": False}, "steps": 10})

    task = collector.get_task_result("task_a")
    assert "num_errors" not in task
    assert task.get("mean_success") == pytest.approx(0.5)


def test_custom_metric_aggregation():
    """Benchmark-specific metrics (e.g. completed_subtasks) are aggregated."""
    collector = ResultCollector(
        "calvin_bench", mode="sync", metric_keys={"success": "mean", "completed_subtasks": "mean"}
    )
    collector.record("seq_0", {"episode_id": 0, "metrics": {"success": True, "completed_subtasks": 5}})
    collector.record("seq_0", {"episode_id": 1, "metrics": {"success": False, "completed_subtasks": 3}})

    task = collector.get_task_result("seq_0")
    assert task.get("mean_success") == pytest.approx(0.5)
    assert task.get("mean_completed_subtasks") == pytest.approx(4.0)

    result = collector.get_benchmark_result()
    assert result.get("mean_completed_subtasks") == pytest.approx(4.0)
    assert result.get("metric_keys") == {"success": "mean", "completed_subtasks": "mean"}


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


def _make_shard(shard_id: int, total: int, tasks: list[dict], metric_keys: dict | None = None) -> dict:
    """Helper to build a shard result dict."""
    d: dict = {
        "benchmark": "test_bench",
        "mode": "sync",
        "harness_version": "0.1.0",
        "tasks": tasks,
        "config": {},
        "shard": {"id": shard_id, "total": total},
    }
    if metric_keys:
        d["metric_keys"] = metric_keys
    return d


def test_merge_two_shards():
    mk = {"success": "mean"}
    shard0 = _make_shard(
        0,
        2,
        [{"task": "A", "episodes": [{"episode_id": 0, "metrics": {"success": True}, "steps": 10}]}],
        metric_keys=mk,
    )
    shard1 = _make_shard(
        1,
        2,
        [{"task": "A", "episodes": [{"episode_id": 1, "metrics": {"success": False}, "steps": 20}]}],
        metric_keys=mk,
    )

    merged = merge_shards([shard0, shard1])
    assert merged["benchmark"] == "test_bench"
    assert merged["merge_info"]["num_shards"] == 2
    assert merged["merge_info"]["shards_missing"] == []
    assert "partial" not in merged

    # Should have 1 task with 2 episodes
    assert len(merged["tasks"]) == 1
    assert len(merged["tasks"][0]["episodes"]) == 2
    assert merged.get("mean_success") == pytest.approx(0.5)


def test_merge_detects_missing_shard():
    shard0 = _make_shard(
        0,
        3,
        [{"task": "A", "episodes": [{"episode_id": 0, "metrics": {"success": True}, "steps": 5}]}],
    )
    shard2 = _make_shard(
        2,
        3,
        [{"task": "A", "episodes": [{"episode_id": 2, "metrics": {"success": True}, "steps": 5}]}],
    )

    merged = merge_shards([shard0, shard2])
    assert merged["merge_info"]["shards_missing"] == [1]
    assert merged["partial"] is True


def test_merge_rejects_duplicate_shard_ids():
    """Duplicate shard IDs should raise ValueError."""
    shard0_v1 = _make_shard(
        0,
        1,
        [{"task": "A", "episodes": [{"episode_id": 0, "metrics": {"success": False}, "steps": 10}]}],
    )
    shard0_v2 = _make_shard(
        0,
        1,
        [{"task": "A", "episodes": [{"episode_id": 0, "metrics": {"success": True}, "steps": 5}]}],
    )

    with pytest.raises(ValueError, match="Duplicate shard IDs"):
        merge_shards([shard0_v1, shard0_v2])


def test_merge_rejects_benchmark_mismatch():
    shard0 = _make_shard(0, 2, [])
    shard1 = _make_shard(1, 2, [])
    shard1["benchmark"] = "different_bench"

    with pytest.raises(ValueError, match="Benchmark mismatch"):
        merge_shards([shard0, shard1])


def test_merge_empty_raises():
    with pytest.raises(ValueError, match="No shard files"):
        merge_shards([])


def test_merge_excludes_infra_errors():
    """Merged results should exclude infra-errored episodes from metrics."""
    mk = {"success": "mean"}
    shard0 = _make_shard(
        0,
        2,
        [
            {
                "task": "A",
                "episodes": [
                    {"episode_id": 0, "metrics": {"success": True}, "steps": 10},
                    {"episode_id": 2, "metrics": {"success": False}, "failure_reason": "timeout"},
                ],
            }
        ],
        metric_keys=mk,
    )
    shard1 = _make_shard(
        1,
        2,
        [
            {
                "task": "A",
                "episodes": [
                    {"episode_id": 1, "metrics": {"success": False}, "steps": 20},
                    {"episode_id": 3, "metrics": {"success": False}, "failure_reason": "connection_closed"},
                ],
            }
        ],
        metric_keys=mk,
    )

    merged = merge_shards([shard0, shard1])
    # 4 total episodes, all count → mean_success = 1/4
    assert merged.get("mean_success") == pytest.approx(0.25)
    task = merged["tasks"][0]
    assert task["num_episodes"] == 4
    assert task["num_errors"] == 2  # episodes with failure_reason


def test_load_shard_files_rejects_non_shard(tmp_path):
    path = tmp_path / "not_a_shard.json"
    path.write_text(json.dumps({"benchmark": "x", "tasks": []}))

    with pytest.raises(ValueError, match="not a shard result file"):
        load_shard_files([path])
