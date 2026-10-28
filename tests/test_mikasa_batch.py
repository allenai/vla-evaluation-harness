"""Batched MIKASA: :class:`_BatchCore` against a fake ``num_envs``-wide ManiSkill env."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval.benchmarks.base import Benchmark
from vla_eval.benchmarks.mikasa.benchmark import MIKASABenchmark
from vla_eval.recording import NullEpisodeRecorder

_ACT_DIM = 8


class _FakeEnv:
    """All envs step together; env ``i`` ends after ``lengths[i]`` steps and succeeds when ``i`` is even."""

    def __init__(self, num_envs: int, lengths: list[int]) -> None:
        """``num_envs`` envs; env ``i`` ends after ``lengths[i]`` steps."""
        self.action_space = SimpleNamespace(shape=(num_envs, _ACT_DIM))
        self._n = num_envs
        self._lengths = lengths
        self._t = np.zeros(num_envs, dtype=np.int64)
        self.resets = 0
        self.actions: list[np.ndarray] = []

    def _obs(self) -> dict[str, Any]:
        """Batched observation whose first pixel tags each row with its env index."""
        rgb = np.zeros((self._n, 2, 2, 3), dtype=np.uint8)
        rgb[:, 0, 0, 0] = np.arange(self._n)  # tag each row with its env index
        return {"sensor_data": {"base_camera": {"rgb": rgb}}, "state": np.zeros((self._n, 4))}

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset every env."""
        self.resets += 1
        self._t[:] = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[Any, ...]:
        """Step every env with the ``(num_envs, act_dim)`` action."""
        assert action.shape == (self._n, _ACT_DIM)
        self.actions.append(action.copy())
        self._t += 1
        truncated = self._t >= np.asarray(self._lengths)
        success = truncated & (np.arange(self._n) % 2 == 0)
        return self._obs(), self._t.astype(np.float32), np.zeros(self._n, bool), truncated, {"success": success}

    def close(self) -> None:
        """Nothing to release."""
        pass


def _bench(monkeypatch: pytest.MonkeyPatch, num_envs: int, lengths: list[int]) -> tuple[MIKASABenchmark, _FakeEnv]:
    """A MIKASA benchmark over :class:`_FakeEnv`, with torch faked as a pass-through."""
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda a: a))
    bench = MIKASABenchmark(tasks=["RememberColor3-v0"], num_envs=num_envs)
    env = _FakeEnv(num_envs, lengths)

    def ensure_env(task_name: str) -> None:
        """Install the fake env as if ``task_name`` was built."""
        bench._env = env
        bench._task_desc = task_name

    monkeypatch.setattr(bench, "_ensure_env", ensure_env)
    return bench, env


async def _run(views: list[Benchmark], recorders: list[Any] | None = None) -> list[Any]:
    """Run one episode per view side by side; return each episode's result."""
    out: list[Any] = [None] * len(views)

    async def one(i: int) -> None:
        """Drive slot ``i``; it checks its own row and acts with ``i + 1``."""
        try:
            await views[i].start_episode({"name": "RememberColor3-v0"}, recorders[i] if recorders else None)
            first = await views[i].get_observation()
            assert first["images"]["base_camera"][0, 0, 0] == i
            while not await views[i].is_done():
                await views[i].apply_action({"actions": np.full(_ACT_DIM, i + 1, dtype=np.float32)})
            out[i] = await views[i].get_result()
        finally:
            views[i].cleanup()

    async with anyio.create_task_group() as tg:
        for i in range(len(views)):
            tg.start_soon(one, i)
    return out


@pytest.mark.anyio
async def test_slots_slice_their_env_and_finished_rows_get_zero_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each slot reads its own row, and rows of finished slots get zero actions."""
    bench, env = _bench(monkeypatch, num_envs=3, lengths=[1, 3, 2])
    out = await _run(bench.open_batch(3))
    assert out == [{"success": True}, {"success": False}, {"success": True}]
    assert env.resets == 1
    assert [a[:, 0].tolist() for a in env.actions] == [[1, 2, 3], [0, 2, 3], [0, 2, 0]]


@pytest.mark.anyio
async def test_tail_batch_steps_empty_rows_with_zero_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tail batch steps the rows of missing slots with zero actions."""
    bench, env = _bench(monkeypatch, num_envs=4, lengths=[2, 2, 2, 2])
    out = await _run(bench.open_batch(2))
    assert out == [{"success": True}, {"success": False}]
    assert [a[:, 0].tolist() for a in env.actions] == [[1, 2, 0, 0], [1, 2, 0, 0]]


@pytest.mark.anyio
async def test_each_slot_records_into_its_own_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every slot writes frames and steps into its own recorder only."""

    class _Rec(NullEpisodeRecorder):
        """Recorder that keeps the row tag of each frame and every step row."""

        def __init__(self) -> None:
            """Start with no frames and no steps."""
            super().__init__()
            self.frames: list[int] = []
            self.steps: list[dict[str, Any]] = []

        def record_video(self, frame: Any) -> None:
            """Keep the frame's row tag."""
            self.frames.append(int(frame[0, 0, 0]))

        def record_step(self, **row: Any) -> None:
            """Keep the step row."""
            self.steps.append(row)

    bench, _ = _bench(monkeypatch, num_envs=2, lengths=[1, 2])
    recs = [_Rec(), _Rec()]
    await _run(bench.open_batch(2), recs)
    assert [r.frames for r in recs] == [[0, 0], [1, 1, 1]]  # reset frame + one per step, own row only
    assert recs[1].steps == [
        {"reward": 1.0, "done": False, "success": False},
        {"reward": 2.0, "done": True, "success": False},
    ]


def test_num_envs_must_be_positive() -> None:
    """num_envs below 1 is rejected."""
    with pytest.raises(ValueError, match="num_envs must be >= 1"):
        MIKASABenchmark(num_envs=0)
