"""RoboDojo layout selection and env lifecycle over a stand-in EvalEnv (no Isaac)."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vla_eval.benchmarks.robodojo.benchmark import MAX_FAILED_RESETS, RoboDojoBenchmark
from vla_eval.runners.base import EnvUnusableError


class UnStableError(RuntimeError):
    """Named like RoboDojo's ``utils.cluttered_generator.UnStableError``."""


class _FakeEnv:
    """Single-env EvalEnv whose reset fails on the ``broken`` layouts; logs the thread of every call."""

    def __init__(self, total: int, broken: set[int], unstable: bool = False) -> None:
        self.seed_manager = SimpleNamespace(seed_list=list(range(total)))
        self.broken, self.unstable = broken, unstable
        self.calls: list[tuple[str, int]] = []
        self.threads: set[int] = set()
        self.layout = -1

    def _log(self, name: str, arg: int = -1) -> None:
        self.calls.append((name, arg))
        self.threads.add(threading.get_ident())

    def close(self) -> None:
        self._log("close")

    def reset(self, seed: list[int]) -> None:
        self._log("reset", seed[0])
        if seed[0] in self.broken:
            raise (UnStableError if self.unstable else RuntimeError)("scene failed")
        self.layout = seed[0]

    def run_reward(self) -> None:
        pass

    def get_obs(self) -> dict[str, Any]:
        return {"vision": {}, "layout": self.layout}


def _bench(
    total: int = 10, eval_num: int = 6, broken: set[int] | None = None, unstable: bool = False
) -> tuple[RoboDojoBenchmark, _FakeEnv]:
    bench = RoboDojoBenchmark()
    env = _FakeEnv(total, broken or set(), unstable)
    bench._env, bench._current_task_name, bench._native_eval_num = env, "t", eval_num
    return bench, env


async def _run(bench: RoboDojoBenchmark, episodes: list[int], **shard: int) -> list[int]:
    """Start each episode as the runner does; the layout each one landed on."""
    out = []
    for ep in episodes:
        await bench.start_episode({"name": "t", "episode_idx": ep, **shard})
        out.append(bench._current_layout_id)
    return out


@pytest.mark.anyio
async def test_every_reset_after_the_first_closes_the_scene_on_the_env_thread() -> None:
    bench, env = _bench(broken={1})
    assert await _run(bench, [0, 1]) == [0, 6]
    assert env.calls == [("reset", 0), ("close", -1), ("reset", 1), ("close", -1), ("reset", 6)]
    assert len(env.threads) == 1 and threading.get_ident() not in env.threads
    bench.cleanup()


@pytest.mark.anyio
async def test_shards_split_the_spares_and_a_resumed_shard_skips_counted_ones() -> None:
    bench, _ = _bench(total=12, broken={0, 1, 2})
    assert await _run(bench, [0, 1], shard_id=1, num_shards=2) == [7, 9]
    resumed, _ = _bench(total=12, broken={0, 1, 2})
    resumed.resume([{"layout_id": 7}, {"layout_id": 9}, {"success": False}])
    assert await _run(resumed, [2], shard_id=1, num_shards=2) == [11]


@pytest.mark.anyio
async def test_failed_resets_in_a_row_make_the_env_unusable() -> None:
    bench, env = _bench(total=20, eval_num=6, broken=set(range(20)))
    with pytest.raises(EnvUnusableError):
        await _run(bench, [0])
    assert [a for name, a in env.calls if name == "reset"] == [0, 6, 7, 8, 9][:MAX_FAILED_RESETS]


@pytest.mark.anyio
async def test_unstable_layouts_never_make_the_env_unusable() -> None:
    # Deterministic instability would fail the same way after a restart; only spare exhaustion ends the episode.
    bench, _ = _bench(total=20, eval_num=6, broken=set(range(20)), unstable=True)
    with pytest.raises(RuntimeError, match="no spare layout left") as raised:
        await _run(bench, [0])
    assert not isinstance(raised.value, EnvUnusableError)
