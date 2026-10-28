"""Batched MIKASA: ``num_envs`` episodes of one task in one GPU-parallel ManiSkill3 env.

A batch resets all ``num_envs`` envs together (the same unseeded ``reset()`` as a single
episode), then steps all of them on every call: ManiSkill cannot step a subset, so rows of
empty or finished slots take a zero action and are never read.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks import batching
from vla_eval.benchmarks.base import Benchmark, StepResult
from vla_eval.recording import EpisodeRecorder
from vla_eval.types import Action, EpisodeResult, Observation, Task

from vla_eval.benchmarks.mikasa.benchmark import MIKASABenchmark


def _numpy(x: Any) -> np.ndarray:
    """``x`` as a host numpy array (tensors are moved off the device first)."""
    return np.asarray(x.cpu().numpy() if hasattr(x, "cpu") else x)


def _frames(raw_obs: Any) -> dict[str, np.ndarray]:
    """``{camera: (num_envs, H, W, 3)}`` on the host; one device copy per camera."""
    if not isinstance(raw_obs, dict) or "sensor_data" not in raw_obs:
        return {}
    return {cam: _numpy(data["rgb"]) for cam, data in raw_obs["sensor_data"].items() if "rgb" in data}


def _row(frames: dict[str, np.ndarray], slot: int) -> dict[str, Any]:
    """The slot's raw obs in the shape ``MIKASABenchmark.make_obs`` reads."""
    return {"sensor_data": {cam: {"rgb": rgb[slot]} for cam, rgb in frames.items()}}


def _flags(value: Any, rows: int) -> np.ndarray:
    """A per-env bool vector from a batched tensor, a scalar or None."""
    if value is None:
        return np.zeros(rows, dtype=bool)
    return np.broadcast_to(_numpy(value).astype(bool).reshape(-1), (rows,))


class _BatchCore(batching.BatchCore):
    """Drives the parent benchmark's ``num_envs``-wide env; env ``i`` belongs to slot ``i``."""

    bench: MIKASABenchmark

    def __init__(self, bench: MIKASABenchmark, size: int) -> None:
        """Batch of ``size`` slots over the ``bench.batch_size()`` envs of the parent env."""
        super().__init__(bench, size)
        self._rows = bench.batch_size()
        self._recorders: dict[int, EpisodeRecorder] = {}
        self._live_rows: set[int] = set()

    def _reset_batch(self, payloads: dict[int, tuple[Task, EpisodeRecorder]]) -> tuple[dict[int, Any], set[int]]:
        """Reset every env of the shared task; slot ``i`` takes env ``i``."""
        bench = self.bench
        names = {str(task["name"]) for task, _ in payloads.values()}
        if len(names) != 1:
            raise RuntimeError(f"MIKASA: one batch must share one task, got {sorted(names)}")
        bench._ensure_env(names.pop())
        obs, _info = bench._env.reset()
        frames = _frames(obs)
        results: dict[int, Any] = {}
        for slot, (_, recorder) in payloads.items():
            self._recorders[slot] = recorder
            results[slot] = _row(frames, slot)
            recorder.record_video(bench._extract_frame(results[slot]))
        self._live_rows = set(payloads)
        return results, set()

    def _step_batch(self, payloads: dict[int, Action]) -> tuple[dict[int, StepResult], set[int]]:
        """Step every env; submitted slots get their action, the other envs a zero action."""
        import torch

        bench = self.bench
        if bench._env is None or not self._live_rows:
            raise RuntimeError("MIKASA: step before reset")
        act_dim = bench._env.action_space.shape[-1]
        actions = np.zeros((self._rows, act_dim), dtype=np.float32)
        for slot, action in payloads.items():
            actions[slot] = bench._action_array(action)
        obs, reward, terminated, truncated, info = bench._env.step(torch.from_numpy(actions))
        frames = _frames(obs)
        rewards = np.broadcast_to(_numpy(reward).astype(np.float32).reshape(-1), (self._rows,))
        done = _flags(terminated, self._rows) | _flags(truncated, self._rows)
        success = _flags(info.get("success"), self._rows)
        results: dict[int, StepResult] = {}
        finished: set[int] = set()
        for slot in payloads:
            done_val, success_val, rew = bool(done[slot]), bool(success[slot]), float(rewards[slot])
            raw_obs = _row(frames, slot)
            recorder = self._recorders[slot]
            recorder.record_video(bench._extract_frame(raw_obs))
            recorder.record_step(reward=rew, done=done_val, success=success_val)
            results[slot] = StepResult(obs=raw_obs, reward=rew, done=done_val, info={"success": success_val})
            if done_val:
                finished.add(slot)
        self._live_rows -= finished
        return results, finished

    def make_obs(self, slot: int, raw_obs: Any, task: Task) -> Observation:
        """The slot's observation, built exactly like the single-env one."""
        return self.bench.make_obs(raw_obs, task)

    def get_result(self, slot: int, last: StepResult) -> EpisodeResult:
        """Success as reported by the env on the slot's last step."""
        return {"success": bool(last.info.get("success", False))}


def open_batch(bench: MIKASABenchmark, size: int) -> list[Benchmark]:
    """One view per slot over a fresh MIKASA batch core."""
    return batching.open_views(_BatchCore(bench, size))
