"""Push-T adapter against the real gym-pusht env (the ``pusht`` extra); skipped when absent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_eval.api import evaluate
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

pytest.importorskip("gym_pusht")

from vla_eval.benchmarks.pusht.benchmark import ACTION_DIM, PushTBenchmark  # noqa: E402


class _CenterPolicy(PredictModelServer):
    """Always drives the agent to the board centre; enough to exercise the loop."""

    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        assert obs["images"]["agentview"].shape == (96, 96, 3)
        assert obs["state"].shape == (ACTION_DIM,)
        return {"actions": np.array([256.0, 256.0], dtype=np.float32)}

    def get_action_spec(self):
        return PushTBenchmark().get_action_spec()

    def get_observation_spec(self):
        return PushTBenchmark().get_observation_spec()


def test_pusht_episode_end_to_end(tmp_path) -> None:
    config = {
        "benchmarks": [
            {
                "benchmark": "vla_eval.benchmarks.pusht.benchmark:PushTBenchmark",
                "episodes_per_task": 2,
                "max_steps": 20,
                "params": {"seed": 3},
            }
        ]
    }
    results = evaluate(_CenterPolicy(), config, docker=False, no_save=True, output_dir=tmp_path)
    task = results[0]["tasks"][0]
    assert task["num_episodes"] == 2
    assert task.get("num_errors", 0) == 0
    assert 0.0 <= results[0]["mean_coverage"] <= 1.0
    assert all(ep["steps"] == 20 for ep in task["episodes"])


def test_pusht_hold_action_repeats_last_target() -> None:
    bench = PushTBenchmark()
    first = bench.get_hold_action(None)["actions"]
    assert first.shape == (ACTION_DIM,)
    last = {"actions": np.array([1.0, 2.0], dtype=np.float32)}
    assert bench.get_hold_action(last) is last
