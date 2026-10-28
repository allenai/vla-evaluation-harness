"""Batched RoboDojo: the barrier of :class:`_BatchCore` against a fake batched EvalEnv."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval.benchmarks.base import Benchmark
from vla_eval.benchmarks import batching
from vla_eval.benchmarks.robodojo.benchmark import ACTION_DIM, RoboDojoBenchmark
from vla_eval.recording import NullEpisodeRecorder
from vla_eval.runners.base import RetryEpisode

_ACTION = {"actions": np.zeros(ACTION_DIM, dtype=np.float32)}


class _FakeBatchEnv:
    """Upstream EvalEnv's batch surface: per-env end_flag/success, None-padded seeds, idx-aligned obs."""

    def __init__(self, num_envs: int, total: int, lengths: dict[int, int], broken: set[int]) -> None:
        """``num_envs`` envs over ``total`` layouts; layout ``l`` ends after ``lengths[l]`` steps."""
        self.seed_manager = SimpleNamespace(seed_list=list(range(total)))
        self.reward_manager = SimpleNamespace(get_score=lambda: [50.0] * num_envs)
        self._n = num_envs
        self._lengths = lengths
        self._broken = broken
        self.layout: list[int | None] = [None] * num_envs
        self.end_flag = [True] * num_envs
        self.success = [False] * num_envs
        self.unstable_envs: list[int] = []
        self._steps = [0] * num_envs
        self.reset_calls: list[list[int | None]] = []
        self.step_calls: list[list[int]] = []
        self.raise_on_reset = False
        self.raise_on_step = False

    def reset(self, seed: list[int | None]) -> None:
        """Reset the envs with a seed; ``None`` leaves an env as it is."""
        assert len(seed) == self._n
        self.reset_calls.append(list(seed))
        if self.raise_on_reset:
            raise RuntimeError("PhysX blew up")
        self.unstable_envs = [i for i, s in enumerate(seed) if s in self._broken]
        for i, s in enumerate(seed):
            if s is not None:
                self.layout[i], self.end_flag[i], self.success[i], self._steps[i] = s, False, False, 0

    def run_reward(self) -> None:
        """No reward terms to update."""
        pass

    def get_score(self) -> list[float]:
        """No partial score."""
        return [0.0] * self._n

    def take_action_batch(self, actions: list[dict[str, Any]], env_idx_list: list[int]) -> None:
        """Step the envs in ``env_idx_list``; an env must not step after its episode ended."""
        assert len(actions) == len(env_idx_list)
        self.step_calls.append(list(env_idx_list))
        if self.raise_on_step:
            raise RuntimeError("sim crashed")
        for i in env_idx_list:
            assert not self.end_flag[i], f"env {i} stepped after its episode ended"
            self._steps[i] += 1

    def is_episode_end(self) -> None:
        """End the episodes that reached their length; even layouts succeed."""
        for i in range(self._n):
            layout = self.layout[i]
            if not self.end_flag[i] and layout is not None and self._steps[i] >= self._lengths.get(layout, 1):
                self.end_flag[i] = True
                self.success[i] = layout % 2 == 0  # even layouts succeed

    def get_obs_batch(self, idx: list[int]) -> list[dict[str, Any]]:
        """One observation per env in ``idx``, tagged with its layout."""
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return [{"vision": {"head": {"color": frame}}, "layout": self.layout[i]} for i in idx]


def _bench(
    num_envs: int = 3,
    total: int = 10,
    eval_num: int = 6,
    lengths: dict[int, int] | None = None,
    broken: set[int] | None = None,
) -> tuple[RoboDojoBenchmark, _FakeBatchEnv]:
    """A RoboDojo benchmark over :class:`_FakeBatchEnv` with ``eval_num`` scored layouts."""
    bench = RoboDojoBenchmark(num_envs=num_envs)
    env = _FakeBatchEnv(num_envs, total, lengths or {}, broken or set())
    bench._env = env
    bench._current_task_name = "make_toast"
    bench._native_eval_num = eval_num
    return bench, env


def _task(ep: int) -> dict[str, Any]:
    """Task dict of episode ``ep`` in a single-shard run."""
    return {"name": "make_toast", "episode_idx": ep, "shard_id": 0, "num_shards": 1}


async def _drive(view: Benchmark, ep: int, *, fail_after: int | None = None) -> Any:
    """A minimal runner: reset, step until done; returns the result or the exception it hit."""
    try:
        await view.start_episode(_task(ep))
        steps = 0
        while not await view.is_done():
            if fail_after is not None and steps >= fail_after:
                raise RuntimeError("model server error")
            await view.get_observation()
            await view.apply_action(_ACTION)
            steps += 1
        return await view.get_result()
    except Exception as exc:
        return exc
    finally:
        view.cleanup()


async def _run_batch(
    bench: RoboDojoBenchmark, episodes: list[int], fail_after: dict[int, int] | None = None
) -> list[Any]:
    """Run ``episodes`` side by side in one batch; ``fail_after`` makes a slot's runner fail."""
    views = bench.open_batch(len(episodes))
    out: list[Any] = [None] * len(episodes)

    async def one(i: int) -> None:
        """Drive slot ``i`` through its episode."""
        out[i] = await _drive(views[i], episodes[i], fail_after=(fail_after or {}).get(i))

    async with anyio.create_task_group() as tg:
        for i in range(len(episodes)):
            tg.start_soon(one, i)
    return out


@pytest.mark.anyio
async def test_slots_step_together_and_drop_out_as_they_finish() -> None:
    """All slots step together, and each one stops stepping once its episode ends."""
    bench, env = _bench(num_envs=3, lengths={0: 1, 1: 3, 2: 2})
    out = await _run_batch(bench, [0, 1, 2])
    assert env.reset_calls == [[0, 1, 2]]
    assert env.step_calls == [[0, 1, 2], [1, 2], [1]]
    assert out == [
        {"success": True, "score": 1.0, "layout_id": 0},
        {"success": False, "score": 0.5, "layout_id": 1},  # failed: partial score from the reward manager
        {"success": True, "score": 1.0, "layout_id": 2},
    ]


@pytest.mark.anyio
async def test_tail_batch_pads_seeds_with_none() -> None:
    """A tail batch pads the seeds of missing slots with ``None`` and steps only the live envs."""
    bench, env = _bench(num_envs=4)
    out = await _run_batch(bench, [3, 4])
    assert env.reset_calls == [[3, 4, None, None]]
    assert env.step_calls == [[0, 1]]
    assert [r["layout_id"] for r in out] == [3, 4]


def test_open_batch_rejects_more_views_than_envs() -> None:
    """A batch larger than num_envs is rejected."""
    bench, _ = _bench(num_envs=2)
    with pytest.raises(ValueError, match="num_envs=2"):
        bench.open_batch(3)


@pytest.mark.anyio
async def test_unstable_slot_retries_on_a_spare_layout() -> None:
    """An unstable layout retries on a spare, and a later batch runs it there."""
    bench, env = _bench(num_envs=2, broken={1})
    out = await _run_batch(bench, [0, 1])
    assert isinstance(out[1], RetryEpisode)
    assert out[0]["success"] is True
    assert env.step_calls == [[0]]  # the unstable env never steps
    assert bench._episode_layouts == {1: 6}  # first spare of the single shard
    # The orchestrator re-runs the episode in a later batch; it lands on the spare.
    again = await _run_batch(bench, [1])
    assert env.reset_calls[-1] == [6, None]
    assert again == [{"success": True, "score": 1.0, "layout_id": 6}]


@pytest.mark.anyio
async def test_out_of_range_episode_fails_only_its_slot() -> None:
    """An episode beyond the scored layouts fails its own slot only."""
    bench, env = _bench(num_envs=2, eval_num=6)
    out = await _run_batch(bench, [0, 6])
    assert isinstance(out[1], ValueError) and "episode_idx 6" in str(out[1])
    assert out[0]["success"] is True
    assert env.reset_calls == [[0, None]]


@pytest.mark.anyio
async def test_model_error_releases_only_that_slot() -> None:
    """A model error in one slot's runner leaves the other slot running."""
    bench, env = _bench(num_envs=2, lengths={0: 3, 1: 3})
    out = await _run_batch(bench, [0, 1], fail_after={1: 1})
    assert isinstance(out[1], RuntimeError) and "model server" in str(out[1])
    assert out[0]["success"] is True
    assert env.step_calls == [[0, 1], [0], [0]]


@pytest.mark.anyio
async def test_env_step_error_fails_every_slot() -> None:
    """An exception from the env step fails every slot."""
    bench, env = _bench(num_envs=2, lengths={0: 3, 1: 3})
    env.raise_on_step = True
    out = await _run_batch(bench, [0, 1])
    assert all(isinstance(o, RuntimeError) and "sim crashed" in str(o) for o in out), out
    assert env.step_calls == [[0, 1]]


@pytest.mark.anyio
async def test_barrier_timeout_fails_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slot that never submits fails the whole batch once the barrier times out."""
    monkeypatch.setattr(batching, "BARRIER_TIMEOUT_S", 0.05)
    bench, env = _bench(num_envs=2, lengths={0: 3, 1: 3})
    views = bench.open_batch(2)
    out: list[Any] = [None, None]

    async def stuck() -> None:  # resets, then hangs (e.g. on inference) without leaving the batch
        """Reset, then hang past the barrier timeout without leaving the batch."""
        try:
            await views[1].start_episode(_task(1))
            await anyio.sleep(0.3)
            await views[1].apply_action(_ACTION)
        except Exception as exc:
            out[1] = exc
        finally:
            views[1].cleanup()

    async def healthy() -> None:
        """Run episode 0 normally."""
        out[0] = await _drive(views[0], 0)

    async with anyio.create_task_group() as tg:
        tg.start_soon(healthy)
        tg.start_soon(stuck)
    assert isinstance(out[0], RuntimeError) and "barrier timed out" in str(out[0]) and "[1]" in str(out[0])
    assert isinstance(out[1], RuntimeError) and "batch failed" in str(out[1])
    assert env.step_calls == []


@pytest.mark.anyio
async def test_failed_batch_reset_moves_every_episode_to_a_spare() -> None:
    """A failed batch reset moves every episode to a spare, up to the limit of retries in a row."""
    bench, env = _bench(num_envs=2, total=40, eval_num=6)
    env.raise_on_reset = True
    for round_ in range(1, 5):
        out = await _run_batch(bench, [0, 1])
        assert all(isinstance(o, RetryEpisode) for o in out), out
        assert bench._failed_batch_resets == round_
    assert bench._episode_layouts == {0: 12, 1: 13}  # four rounds of spares: 6/7, 8/9, 10/11, 12/13
    out = await _run_batch(bench, [0, 1])
    assert all(isinstance(o, RuntimeError) and "5 batch resets in a row" in str(o) for o in out), out
    env.raise_on_reset = False
    bench._failed_batch_resets = 0
    bench._episode_failures.clear()
    out = await _run_batch(bench, [0, 1])
    assert [r["layout_id"] for r in out] == [12, 13]  # the stored spare stays the episode's layout


@pytest.mark.anyio
async def test_each_slot_records_into_its_own_recorder() -> None:
    """Every slot writes frames and steps into its own recorder only."""

    class _Rec(NullEpisodeRecorder):
        """Recorder that counts frames and keeps every step row."""

        def __init__(self) -> None:
            """Start with no frames and no steps."""
            super().__init__()
            self.frames = 0
            self.steps: list[dict[str, Any]] = []

        def record_video(self, frame: Any) -> None:
            """Count the frame."""
            self.frames += frame is not None

        def record_step(self, **row: Any) -> None:
            """Keep the step row."""
            self.steps.append(row)

    bench, _ = _bench(num_envs=2, lengths={0: 1, 1: 2})
    views = bench.open_batch(2)
    recs = [_Rec(), _Rec()]

    async def one(i: int) -> None:
        """Drive slot ``i`` through its episode into its own recorder."""
        try:
            await views[i].start_episode(_task(i), recs[i])
            while not await views[i].is_done():
                await views[i].apply_action(_ACTION)
        finally:
            views[i].cleanup()

    async with anyio.create_task_group() as tg:
        for i in range(2):
            tg.start_soon(one, i)
    assert [r.frames for r in recs] == [2, 3]  # reset frame + one per step
    assert [len(r.steps) for r in recs] == [1, 2]
    assert recs[1].steps[-1] == {"reward": 0.0, "done": True, "success": False, "score": 0.5}
