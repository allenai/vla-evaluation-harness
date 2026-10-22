"""RoboDojo layout selection: episode_idx picks the counted layout, spares are split by shard."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vla_eval.benchmarks.robodojo.benchmark import RoboDojoBenchmark


class _FakeEnv:
    def __init__(self, total: int, broken: set[int]) -> None:
        self.seed_manager = SimpleNamespace(seed_list=list(range(total)))
        self._broken = broken
        self.reset_calls: list[int] = []
        self._layout = -1

    def reset(self, seed: list[int]) -> None:
        self.reset_calls.append(seed[0])
        if seed[0] in self._broken:
            raise RuntimeError("scene unstable")
        self._layout = seed[0]

    def run_reward(self) -> None:
        pass

    def get_obs(self) -> dict[str, Any]:
        return {"vision": {}, "layout": self._layout}


def _bench(total: int = 10, eval_num: int = 6, broken: set[int] | None = None) -> tuple[RoboDojoBenchmark, _FakeEnv]:
    bench = RoboDojoBenchmark()
    env = _FakeEnv(total, broken or set())
    bench._env = env
    bench._current_task_name = "make_toast"
    bench._native_eval_num = eval_num
    return bench, env


def _run(bench: RoboDojoBenchmark, episodes: list[int], shard: dict[str, int] | None = None) -> list[int]:
    out = []
    for ep in episodes:
        bench.reset({"name": "make_toast", "episode_idx": ep, **(shard or {})})
        out.append(bench._current_layout_id)
    return out


def test_episode_idx_selects_layout() -> None:
    bench, _ = _bench()
    assert _run(bench, [3, 0, 5]) == [3, 0, 5]


def test_single_shard_spares_match_upstream_set() -> None:
    # Upstream consumes 0..9 in order skipping 2 -> counts {0,1,3,4,5,6}.
    bench, _ = _bench(broken={2})
    assert sorted(_run(bench, list(range(6)))) == [0, 1, 3, 4, 5, 6]


def test_shards_never_share_layouts() -> None:
    # Spares are 6..9: shard 0 owns 6/8, shard 1 owns 7/9.
    shard0, _ = _bench(broken={0, 3, 6})
    shard1, _ = _bench(broken={0, 3, 6})
    got0 = _run(shard0, [0, 2], {"shard_id": 0, "num_shards": 2})
    got1 = _run(shard1, [1, 3, 5], {"shard_id": 1, "num_shards": 2})
    assert got0 == [8, 2]
    assert got1 == [1, 7, 5]
    assert set(got0).isdisjoint(got1)


def test_spare_exhaustion_raises() -> None:
    bench, _ = _bench(total=8, broken={0, 6})
    with pytest.raises(RuntimeError, match="exhausted"):
        _run(bench, [0], {"shard_id": 0, "num_shards": 2})


def test_episode_idx_outside_counted_layouts_rejected() -> None:
    bench, _ = _bench(eval_num=6)
    with pytest.raises(ValueError, match="episode_idx 6"):
        _run(bench, [6])
