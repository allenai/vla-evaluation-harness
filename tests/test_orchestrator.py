"""Tests for Orchestrator: integration and error handling paths."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
import websockets.exceptions

from vla_eval.orchestrator import Orchestrator

from tests.conftest import StubBenchmark


@pytest.mark.anyio
async def test_orchestrator_runs_to_completion(echo_server, tmp_path):
    """Full integration: echo server + stub benchmark → results saved."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "stub_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 3, "num_tasks": 2},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ):
        orchestrator = Orchestrator(config)
        results = await orchestrator.run()

    assert len(results) == 1
    result = results[0]
    assert result["overall_success_rate"] == pytest.approx(1.0)
    assert len(result["tasks"]) == 2

    # Verify file was saved
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    saved = json.loads(json_files[0].read_text())
    assert "partial" not in saved


@pytest.mark.anyio
async def test_orchestrator_saves_partial_on_server_death(tmp_path):
    """When connection fails mid-run, partial results are saved."""

    call_count = 0

    class FlakyConnection:
        """Mock connection that raises ConnectionClosed after 9 act() calls."""

        def __init__(self, url, **kwargs):
            self.url = url
            self.server_info = {}

        async def connect(self, **kwargs):
            pass

        async def close(self):
            pass

        async def start_episode(self, cfg):
            pass

        async def end_episode(self, result):
            pass

        async def act(self, obs):
            nonlocal call_count
            call_count += 1
            # Fail on 10th call (after 3 complete tasks × 3 steps each)
            if call_count >= 10:
                raise websockets.exceptions.ConnectionClosed(None, None)
            return {"actions": np.ones(7, dtype=np.float32)}

        async def reconnect(self):
            raise ConnectionError("Server unreachable after retries")

    config = {
        "server": {"url": "ws://fake:9999"},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "partial_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 3, "num_tasks": 5},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ), patch(
        "vla_eval.orchestrator.Connection",
        FlakyConnection,
    ):
        orchestrator = Orchestrator(config)
        results = await orchestrator.run()

    # Orchestrator returns partial result instead of raising
    assert len(results) == 1
    assert results[0].get("partial") is True

    # Partial results should have been saved
    json_files = list(tmp_path.glob("*partial*.json"))
    assert len(json_files) >= 1
    saved = json.loads(json_files[0].read_text())
    assert saved["partial"] is True

    # 3 complete tasks + 1 failed episode = 4 episodes total
    total_episodes = sum(len(t["episodes"]) for t in saved["tasks"])
    assert total_episodes == 4


@pytest.mark.anyio
async def test_orchestrator_sharding_splits_work(echo_server, tmp_path):
    """Sharded runs produce disjoint subsets that cover all episodes."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "shard_test",
                "episodes_per_task": 3,
                "max_steps": 50,
                "params": {"done_at_step": 2, "num_tasks": 2},
            }
        ],
    }

    all_episode_keys: list[tuple[str, int]] = []
    for shard_id in range(3):
        with patch(
            "vla_eval.orchestrator.resolve_import_string",
            return_value=StubBenchmark,
        ):
            orch = Orchestrator(config, shard_id=shard_id, num_shards=3)
            results = await orch.run()

        for task_result in results[0]["tasks"]:
            for ep in task_result["episodes"]:
                all_episode_keys.append((task_result["task"], ep["episode_id"]))

    # 2 tasks × 3 episodes = 6 total, split across 3 shards
    assert len(all_episode_keys) == 6
    assert len(set(all_episode_keys)) == 6  # no duplicates


@pytest.mark.anyio
async def test_orchestrator_sharding_deterministic_filename(echo_server, tmp_path):
    """Sharded result uses deterministic filename (no timestamp)."""
    config = {
        "server": {"url": echo_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "tests.conftest:StubBenchmark",
                "name": "fname_test",
                "episodes_per_task": 1,
                "max_steps": 50,
                "params": {"done_at_step": 2, "num_tasks": 1},
            }
        ],
    }

    with patch(
        "vla_eval.orchestrator.resolve_import_string",
        return_value=StubBenchmark,
    ):
        orch = Orchestrator(config, shard_id=0, num_shards=2)
        await orch.run()

    expected = tmp_path / "fname_test_shard0of2.json"
    assert expected.exists()
    saved = json.loads(expected.read_text())
    assert saved["shard"] == {"id": 0, "total": 2}
