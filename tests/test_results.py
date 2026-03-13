"""Tests for ResultCollector."""

from __future__ import annotations

import json

import pytest

from vla_eval.results.collector import ResultCollector
from vla_eval.results.merge import load_shard_files, merge_shards


def test_result_collector():
    collector = ResultCollector("test_bench", mode="sync")
    collector.record("task_a", {"task": "task_a", "episode_id": "task_a_ep0", "success": True, "steps": 10})
    collector.record("task_a", {"task": "task_a", "episode_id": "task_a_ep1", "success": False, "steps": 20})
    collector.record("task_b", {"task": "task_b", "episode_id": "task_b_ep0", "success": True, "steps": 5})

    result = collector.get_benchmark_result()
    assert result["benchmark"] == "test_bench"
    assert result["overall_success_rate"] == pytest.approx(2 / 3)

    task_a = collector.get_task_result("task_a")
    assert task_a["success_rate"] == pytest.approx(0.5)
    assert task_a["avg_steps"] == pytest.approx(15.0)


def test_empty_collector():
    collector = ResultCollector("empty_bench")
    result = collector.get_benchmark_result()
    assert result["benchmark"] == "empty_bench"
    assert result["overall_success_rate"] == pytest.approx(0.0)
    assert result["tasks"] == []


def test_to_json_returns_valid_json():
    collector = ResultCollector("json_bench", mode="sync")
    collector.record("t1", {"task": "t1", "episode_id": "t1_ep0", "success": True, "steps": 5})
    text = collector.to_json()
    parsed = json.loads(text)
    assert parsed["benchmark"] == "json_bench"
    assert isinstance(parsed["tasks"], list)


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------


def _make_shard(shard_id: int, total: int, tasks: list[dict]) -> dict:
    """Helper to build a shard result dict."""
    return {
        "benchmark": "test_bench",
        "mode": "sync",
        "harness_version": "0.1.0",
        "tasks": tasks,
        "overall_success_rate": 0.5,
        "config": {},
        "shard": {"id": shard_id, "total": total},
    }


def test_merge_two_shards():
    shard0 = _make_shard(
        0,
        2,
        [
            {
                "task": "A",
                "episodes": [
                    {"episode_id": "A_ep0", "success": True, "steps": 10},
                ],
            },
        ],
    )
    shard1 = _make_shard(
        1,
        2,
        [
            {
                "task": "A",
                "episodes": [
                    {"episode_id": "A_ep1", "success": False, "steps": 20},
                ],
            },
        ],
    )

    merged = merge_shards([shard0, shard1])
    assert merged["benchmark"] == "test_bench"
    assert merged["merge_info"]["num_shards"] == 2
    assert merged["merge_info"]["shards_missing"] == []
    assert "partial" not in merged

    # Should have 1 task with 2 episodes
    assert len(merged["tasks"]) == 1
    assert len(merged["tasks"][0]["episodes"]) == 2
    assert merged["overall_success_rate"] == pytest.approx(0.5)


def test_merge_detects_missing_shard():
    shard0 = _make_shard(
        0,
        3,
        [
            {"task": "A", "episodes": [{"episode_id": "A_ep0", "success": True, "steps": 5}]},
        ],
    )
    shard2 = _make_shard(
        2,
        3,
        [
            {"task": "A", "episodes": [{"episode_id": "A_ep2", "success": True, "steps": 5}]},
        ],
    )

    merged = merge_shards([shard0, shard2])
    assert merged["merge_info"]["shards_missing"] == [1]
    assert merged["partial"] is True


def test_merge_rejects_duplicate_shard_ids():
    """Duplicate shard IDs should raise ValueError."""
    shard0_v1 = _make_shard(
        0,
        1,
        [
            {"task": "A", "episodes": [{"episode_id": "A_ep0", "success": False, "steps": 10}]},
        ],
    )
    shard0_v2 = _make_shard(
        0,
        1,
        [
            {"task": "A", "episodes": [{"episode_id": "A_ep0", "success": True, "steps": 5}]},
        ],
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


def test_load_shard_files_rejects_non_shard(tmp_path):
    path = tmp_path / "not_a_shard.json"
    path.write_text(json.dumps({"benchmark": "x", "tasks": []}))

    with pytest.raises(ValueError, match="not a shard result file"):
        load_shard_files([path])
