"""OCBenchBenchmark against a fake MJWarp env: seeds per episode, partial resets, per-task limits."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vla_eval.benchmarks.ocbench.benchmark import OCBenchBenchmark, episode_seed
from vla_eval.recording import NullEpisodeRecorder


class FakeEnv:
    """State = [seed, steps]; terminates (with success) when the state's seed is odd and 3 steps have passed."""

    def __init__(self, nworld: int) -> None:
        self.nworld = nworld
        self.state = np.zeros((nworld, 2))
        self.actions: list[np.ndarray] = []
        self.calls: list[tuple[str, list[int]]] = []

    def reset(self, seeds):
        self.calls.append(("reset", list(range(self.nworld))))
        self.state[:] = np.stack([seeds, np.zeros(self.nworld)], 1)
        return self.state.copy(), {}

    def reset_worlds(self, world_ids, seeds):
        self.calls.append(("reset_worlds", list(world_ids)))
        self.state[world_ids] = np.stack([seeds, np.zeros(len(world_ids))], 1)
        obs = self.state + 100.0  # the other worlds' rows are not step()'s observation
        obs[world_ids] = self.state[world_ids]
        return obs, {}

    def step(self, action):
        self.actions.append(action)
        self.state[:, 1] += 1
        success = (self.state[:, 0] % 2 == 1) & (self.state[:, 1] >= 3)
        return (
            self.state.copy(),
            success.astype(np.float32),
            success,
            np.zeros(self.nworld, bool),
            {"success": success},
        )

    def close(self):
        pass


@pytest.fixture
def envs(monkeypatch):
    made: list[FakeEnv] = []
    ocbench = types.ModuleType("ocbench")
    setattr(ocbench, "parse_env_spec", lambda name: (name.split("-")[0], "mjwarp", {}, 5))
    mjwarp = types.ModuleType("ocbench.mjwarp")

    def make_env(family, kwargs, nworld):
        made.append(FakeEnv(nworld))
        return made[-1]

    setattr(mjwarp, "make_env", make_env)
    monkeypatch.setitem(sys.modules, "ocbench", ocbench)
    monkeypatch.setitem(sys.modules, "ocbench.mjwarp", mjwarp)
    return made


def _reset(bench, slots, name, idxs):
    return bench.reset(slots, [{"name": name, "episode_idx": i} for i in idxs], [NullEpisodeRecorder()] * len(slots))


def _act(slots, value):
    return {"actions": np.full((len(slots), 7), value)}


def test_episodes_reset_with_their_own_seeds_and_end_at_the_limit(envs):
    bench = OCBenchBenchmark(num_envs=3, tasks=["block-single-task1-v0"], seed=7)
    seeds = np.asarray([episode_seed(7, i) for i in range(5)], np.float32)  # as the fake state holds them
    obs = _reset(bench, [0, 1], "block-single-task1-v0", [0, 1])
    assert obs["state"][:, 0].tolist() == seeds[:2].tolist() and envs[0].calls == [("reset", [0, 1, 2])]
    bench.step([0, 1], _act([0, 1], 1.0))
    assert np.all(envs[0].actions[-1][2] == 0) and np.all(envs[0].actions[-1][0] == 1)  # the idle world holds
    obs = _reset(bench, [2], "block-single-task1-v0", [4])  # a refill while slots 0 and 1 run
    assert obs["state"].tolist() == [[seeds[4], 0.0]] and envs[0].calls[-1] == ("reset_worlds", [2])
    for _ in range(4):
        obs, done = bench.step([1, 0], _act([1, 0], 0.0))
    assert done.tolist() == [True, True]  # the task's limit of 5 steps (or success)
    assert obs["state"][:, 1].tolist() == [5.0, 5.0] and obs["task_description"] == ["block-single-task1-v0"] * 2
    assert bench.result(0)["success"] == bool(envs[0].state[0, 0] % 2)


def test_seeds_are_ocbench_evaluate_mjwarp_s():
    # make_collection_seeds(7, 0, 10, n)[0] in OCBench's impls/envs/streaming_data.py
    assert episode_seed(7, 0) == 631529809 and episode_seed(0, 0) == 4197554498


def test_a_new_task_builds_its_env(envs):
    bench = OCBenchBenchmark(num_envs=2, tasks=["block-single-task1-v0", "bowling-task1-v0"])
    _reset(bench, [0, 1], "block-single-task1-v0", [0, 1])
    _reset(bench, [0], "bowling-task1-v0", [0])
    assert len(envs) == 2 and envs[1].calls == [("reset", [0, 1])]
    bench.step([0], _act([0], 0.0))
    assert "score" in bench.result(0) and "score" in bench.get_metric_keys()
