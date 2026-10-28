"""Batched Kinetix: every slot must replay the single-env episode bit for bit (fake functional env)."""

from __future__ import annotations

from typing import Any

import anyio
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vla_eval.benchmarks.base import Benchmark  # noqa: E402
from vla_eval.benchmarks.kinetix.benchmark import ACTION_DIM, KinetixBenchmark  # noqa: E402


class _FakeEnv:
    """Gymnax-style env: a noisy 2-D point; reward once x > 1, done after ``length`` steps."""

    def __init__(self, length: int) -> None:
        """Episodes end after ``length`` steps."""
        self.length = length

    def reset(self, rng: Any, params: Any, override_reset_state: Any = None) -> tuple[Any, Any]:
        """Start near the level's stored position."""
        state = {"pos": override_reset_state["pos"] + jax.random.normal(rng, (2,)) * 0.1, "t": jnp.int32(0)}
        return state["pos"], state

    def step(self, rng: Any, state: Any, action: Any, params: Any) -> tuple[Any, ...]:
        """Move by the action plus noise; reward once x > 1."""
        pos = state["pos"] + action[:2] * 0.3 + jax.random.normal(rng, (2,)) * 0.05
        state = {"pos": pos, "t": state["t"] + 1}
        reward = (pos[0] > 1.0).astype(jnp.float32)
        return pos, state, reward, state["t"] >= self.length, {}


def _bench(
    monkeypatch: pytest.MonkeyPatch, num_envs: int, length: int = 50, max_steps: int = 256, noise: float = 0.1
) -> KinetixBenchmark:
    """A symbolic-obs Kinetix benchmark over :class:`_FakeEnv` instead of a level file."""
    bench = KinetixBenchmark(
        num_envs=num_envs, max_episode_steps=max_steps, observation_type="symbolic", action_noise_std=noise, seed=7
    )

    def make_env(level: str) -> None:  # KinetixBenchmark._make_env without loading a level file
        """Install the fake env and its jitted reset / step as if ``level`` was loaded."""
        bench._init_jax()
        env = _FakeEnv(length)
        level_state = {"pos": jnp.zeros(2)}
        bench._env, bench._env_params, bench._level_state, bench._current_level = env, {}, level_state, level
        bench._jit_step = jax.jit(env.step)
        bench._jit_reset = jax.jit(lambda rng, params: env.reset(rng, params, override_reset_state=level_state))
        bench._jit_batch_reset = bench._jit_batch_step = None

    monkeypatch.setattr(bench, "_make_env", make_env)
    return bench


def _policy(obs: dict[str, Any], t: int) -> dict[str, Any]:
    """Deterministic in (obs, t), so diverging trajectories show up as different actions."""
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    a[0] = 0.5 + 0.1 * np.sin(t) - 0.2 * float(obs["state"][1])
    a[1] = -0.3 * float(obs["state"][1])
    return {"actions": a}


async def _episode(view: Benchmark, ep: int) -> tuple[list[np.ndarray], Any]:
    """Run one episode on ``view``; return its observation trace and result."""
    try:
        await view.start_episode({"name": "Fake", "level": "fake", "episode_idx": ep})
        trace = []
        t = 0
        while True:
            obs = await view.get_observation()
            trace.append(np.asarray(obs["state"]))
            if await view.is_done():
                return trace, await view.get_result()
            await view.apply_action(_policy(obs, t))
            t += 1
    finally:
        view.cleanup()


async def _batch(bench: KinetixBenchmark, episodes: list[int]) -> list[Any]:
    """Run ``episodes`` side by side in one batch; return each episode's trace and result."""
    views = bench.open_batch(len(episodes))
    out: list[Any] = [None] * len(episodes)

    async def one(i: int) -> None:
        """Drive slot ``i`` through its episode."""
        out[i] = await _episode(views[i], episodes[i])

    async with anyio.create_task_group() as tg:
        for i in range(len(episodes)):
            tg.start_soon(one, i)
    return out


def _assert_same(single: tuple[list[np.ndarray], Any], batched: tuple[list[np.ndarray], Any]) -> None:
    """Batched and single-env episodes have the same result and the same observations."""
    assert batched[1] == single[1]
    assert len(batched[0]) == len(single[0])
    for a, b in zip(single[0], batched[0]):
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6)


@pytest.mark.anyio
@pytest.mark.parametrize("noise", [0.0, 0.1])
async def test_batched_slots_match_single_env_episodes(monkeypatch: pytest.MonkeyPatch, noise: float) -> None:
    """Every slot reproduces its single-env episode, with and without action noise."""
    episodes = [0, 3, 5]
    single = _bench(monkeypatch, 1, noise=noise)
    expected = [await _episode(single, ep) for ep in episodes]
    batched = await _batch(_bench(monkeypatch, 4, noise=noise), episodes)  # 3 of 4 rows: also covers a tail batch
    for e, b in zip(expected, batched):
        _assert_same(e, b)
    assert len({len(trace) for trace, _ in expected}) == 1


@pytest.mark.anyio
async def test_slots_finish_on_max_steps_and_on_env_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slots end on max steps exactly like the single env, before the env's own done."""
    single = _bench(monkeypatch, 1, length=6, max_steps=4)
    expected = [await _episode(single, ep) for ep in (0, 1)]
    batched = await _batch(_bench(monkeypatch, 2, length=6, max_steps=4), [0, 1])
    for e, b in zip(expected, batched):
        _assert_same(e, b)
        assert len(b[0]) == 5  # reset obs + 4 steps


@pytest.mark.anyio
async def test_later_batch_on_reused_rows_matches_single_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch on rows used by an earlier batch still matches the single env."""
    bench = _bench(monkeypatch, 2)
    await _batch(bench, [0, 1])
    single = _bench(monkeypatch, 1)
    expected = await _episode(single, 9)
    (batched,) = await _batch(bench, [9])
    _assert_same(expected, batched)


@pytest.mark.anyio
async def test_mixed_levels_fail_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch whose slots ask for different levels fails every slot."""
    bench = _bench(monkeypatch, 2)
    views = bench.open_batch(2)
    out: list[Any] = [None, None]

    async def one(i: int) -> None:
        """Reset slot ``i`` on its own level and keep the error it raises."""
        try:
            await views[i].start_episode({"name": "Fake", "level": f"lvl{i}", "episode_idx": i})
        except Exception as exc:
            out[i] = exc
        finally:
            views[i].cleanup()

    async with anyio.create_task_group() as tg:
        for i in range(2):
            tg.start_soon(one, i)
    assert all(isinstance(o, RuntimeError) and "share one level" in str(o) for o in out), out


def test_num_envs_must_be_positive() -> None:
    """num_envs below 1 is rejected."""
    with pytest.raises(ValueError, match="num_envs must be >= 1"):
        KinetixBenchmark(num_envs=0)
