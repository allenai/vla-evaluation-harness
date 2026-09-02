"""Tests for VSArena remote benchmark adapter."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.benchmarks.vsarena.benchmark import (
    VSArenaBenchmark,
    _action_to_vsarena,
    _decode_scene_image,
    _mock_state,
)


def test_mock_state_vla_has_image() -> None:
    state = _mock_state(0, "test-match", mode="vla")
    assert state["observation_mode"] == "vla"
    assert "images" in state
    img = _decode_scene_image(state)
    assert img is not None
    assert img.shape == (8, 8, 3)


def test_action_to_vsarena_libero_style() -> None:
    action = {"actions": np.array([0.01, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)}
    out = _action_to_vsarena(action)
    assert out["gripper_state"] == "closed"
    assert out["ee_delta"]["dx"] == pytest.approx(0.01)


def test_dry_run_episode_completes() -> None:
    bench = VSArenaBenchmark(dry_run=True, max_episode_steps=4)
    tasks = bench.get_tasks()
    assert len(tasks) == 1

    import asyncio

    async def _run() -> None:
        await bench.start_episode(tasks[0])
        obs = await bench.get_observation()
        assert "task_description" in obs
        while not await bench.is_done():
            await bench.apply_action({"actions": np.zeros(7, dtype=np.float32)})
        result = await bench.get_result()
        assert "success" in result

    asyncio.run(_run())
    bench.cleanup()
