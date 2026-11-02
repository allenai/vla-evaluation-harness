"""OCBench (arXiv:2610.07056) on MuJoCo Warp: ``num_envs`` worlds of one task stepped together.

Follows ``evaluate_mjwarp``: episode ``i`` uses world ``i``'s seed (``make_collection_seeds(seed, 0, 10, ·)[i]``),
ends on termination or the step limit; bowling, which never flags success, scores its return x 100.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from vla_eval.benchmarks.base import Batch, VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.specs import RAW, DimSpec
from vla_eval.types import EpisodeResult, Task

TASKS = [
    *(f"block-{v}-v0" for v in ("single-task1", "double-task1", "double-task2", "triple-task1", "triple-task2")),
    *(f"block-quadruple-task{i}-v0" for i in (1, 2, 3)),
    *(f"chamber-easy-task{i}-v0" for i in (1, 2, 3)),
    *(f"chamber-{d}-task{i}-v0" for d in ("medium", "hard") for i in (1, 2)),
    *(f"switch-{g}-task{i}-v0" for g in ("3x3", "4x4", "5x5") for i in (1, 2)),
    *(f"hanoi-{d}-task{i}-v0" for d in ("single", "double", "triple") for i in (1, 2)),
    "bowling-task1-v0",
]


def episode_seed(root: int, episode_idx: int) -> int:
    """``make_collection_seeds(root, 0, 10, n)[episode_idx]`` of OCBench's ``impls/envs/streaming_data.py``
    (not in the ocbench package), the reset seed of world ``episode_idx`` in ``evaluate_mjwarp``."""
    mask = int(np.random.SeedSequence([int(root)]).generate_state(1, dtype=np.uint32)[0])
    v = ((10 << 18) | episode_idx) ^ mask  # split 10, collection 0
    v ^= v >> 16
    v = (v * 0x7FEB352D) & 0xFFFFFFFF
    v ^= v >> 15
    v = (v * 0x846CA68B) & 0xFFFFFFFF
    return v ^ (v >> 16)


class OCBenchBenchmark(VectorStepBenchmark):
    """Params: ``num_envs`` (worlds), ``tasks`` (OCBench MJWarp env names; default the 28 tasks),
    ``seed`` (the evaluation's seed root)."""

    partial_reset: ClassVar[bool] = True

    def __init__(self, num_envs: int = 500, tasks: list[str] | None = None, seed: int = 0) -> None:
        super().__init__(num_envs)
        self.task_names = list(tasks or TASKS)
        self.seed = seed
        self._env: Any = None
        self._env_name = ""
        self._limit = 0
        self._steps = np.zeros(num_envs, np.int64)
        self._returns = np.zeros(num_envs)
        self._success = np.zeros(num_envs, bool)
        self._failure = np.zeros(num_envs, bool)
        self._recorders: list[EpisodeRecorder | None] = [None] * num_envs
        self._action = np.zeros((num_envs, 7))

    def get_tasks(self) -> list[Task]:
        return [{"name": name} for name in self.task_names]

    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> Batch:
        import ocbench

        name = str(tasks[0]["name"])  # the harness runs one task at a time
        seeds = np.array([episode_seed(self.seed, int(t["episode_idx"])) for t in tasks], np.uint32)
        if name != self._env_name:
            if self._env is not None:
                self._env.close()
            family, _, kwargs, self._limit = ocbench.parse_env_spec(name)
            from ocbench.mjwarp import make_env

            self._env, self._env_name = make_env(family, kwargs, self.num_envs), name
            full = np.zeros(self.num_envs, np.uint32)  # a first reset is of every world
            full[slots] = seeds
            obs, _ = self._env.reset(seeds=full)
        else:
            obs, _ = self._env.reset_worlds(np.asarray(slots, np.int32), seeds=seeds)
        # only the reset worlds' rows: reset_worlds recomputes the other worlds' observations, not as step() does
        self._steps[slots] = 0
        self._returns[slots] = 0.0
        self._success[slots] = self._failure[slots] = False
        for slot, recorder in zip(slots, recorders):
            self._recorders[slot] = recorder if recorder.records_steps else None
        return {"state": np.asarray(obs, np.float32)[slots], "task_description": [name] * len(slots)}

    def step(self, slots: list[int], actions: Batch) -> tuple[Batch, np.ndarray]:
        idx = np.asarray(slots, np.int64)
        self._action[:] = 0.0  # idle worlds hold still, as in evaluate_mjwarp
        self._action[idx] = np.asarray(actions["actions"], np.float64).reshape(len(slots), 7)
        obs, reward, terminated, _, info = self._env.step(self._action)
        reward = np.asarray(reward, np.float64)
        self._steps[idx] += 1
        self._returns[idx] += reward[idx]
        self._success[idx] = np.asarray(info["success"], bool)[idx]
        self._failure[idx] = np.asarray(info["failure"], bool)[idx] if "failure" in info else False
        done = (np.asarray(terminated, bool) | (self._steps >= self._limit))[idx]
        for i, slot in enumerate(slots):
            recorder = self._recorders[slot]
            if recorder is not None:
                recorder.record_step(reward=float(reward[slot]), done=bool(done[i]), success=bool(self._success[slot]))
        n = len(slots)
        return {"state": np.asarray(obs, np.float32)[idx], "task_description": [self._env_name] * n}, done

    def result(self, slot: int) -> EpisodeResult:
        out: EpisodeResult = {
            "success": bool(self._success[slot]),
            "failure": bool(self._failure[slot]),
            "return": float(self._returns[slot]),
        }
        if self._env_name.startswith("bowling"):
            out["score"] = 100.0 * out["return"]
        return out

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "failure": "mean", "return": "mean", "score": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": None}  # each task's own limit (1250-5000), applied in step()

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"state": RAW}

    def cleanup(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
