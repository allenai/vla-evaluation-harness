"""Kinetix episode success, against a scripted env (no JAX)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vla_eval.benchmarks.kinetix.benchmark import ACTION_DIM, KinetixBenchmark
from vla_eval.recording import NullEpisodeRecorder


def _run(steps: list[tuple[float, bool]]) -> bool:
    """Step through scripted (reward, GoalR) pairs and return the episode's success."""
    bench = KinetixBenchmark(action_noise_std=0.0)
    bench._jax = SimpleNamespace(random=SimpleNamespace(split=lambda key: (key, key)))
    bench._jnp = SimpleNamespace(array=np.asarray)
    bench._rng = 0
    bench._recorder = NullEpisodeRecorder()
    script = iter(steps)

    def jit_step(_rng, state, _action, _params):
        reward, goal = next(script)
        return np.zeros(1), state, np.float32(reward), np.bool_(False), {"GoalR": np.bool_(goal)}

    bench._jit_step = jit_step
    result = None
    for _ in steps:
        result = bench.step({"actions": np.zeros(ACTION_DIM)})
    assert result is not None
    return bench.get_step_result(result)["success"]


@pytest.mark.parametrize(
    ("steps", "expected"),
    [
        pytest.param([(0.03, False), (0.01, False)], False, id="shaping-only"),
        pytest.param([(0.03, False), (-0.995, False)], False, id="shaping-then-red-shape"),
        pytest.param([(0.03, False), (1.02, True)], True, id="goal"),
        pytest.param([(0.98, True), (-0.01, False)], True, id="goal-stays-true"),
    ],
)
def test_success_follows_goal_not_reward(steps: list[tuple[float, bool]], expected: bool) -> None:
    assert _run(steps) is expected
