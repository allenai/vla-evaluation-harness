"""Push-T adapter against the real gym-pusht env (the ``pusht`` extra); skipped when absent."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.orchestrator import Orchestrator

from tests.conftest import start_server, stop_server

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


@pytest.fixture
async def center_server(free_port: int):
    task = await start_server(_CenterPolicy(), free_port)
    yield f"ws://127.0.0.1:{free_port}"
    await stop_server(task)


@pytest.mark.anyio
async def test_pusht_episode_end_to_end(center_server: str, tmp_path) -> None:
    config = {
        "server": {"url": center_server},
        "output_dir": str(tmp_path),
        "benchmarks": [
            {
                "benchmark": "vla_eval.benchmarks.pusht.benchmark:PushTBenchmark",
                "episodes_per_task": 2,
                "max_steps": 20,
                "params": {"seed": 3},
            }
        ],
    }
    results = await Orchestrator(config, no_save=True).run()
    task = results[0]["tasks"][0]
    assert task["num_episodes"] == 2
    assert task.get("num_errors", 0) == 0
    assert 0.0 <= results[0]["mean_coverage"] <= 1.0
    assert all(ep["steps"] == 20 for ep in task["episodes"])


def test_pusht_hold_action_repeats_last_target() -> None:
    bench = PushTBenchmark()
    assert bench.get_hold_action(None)["actions"].shape == (ACTION_DIM,)
    last = {"actions": np.array([1.0, 2.0], dtype=np.float32)}
    assert bench.get_hold_action(last) is last
