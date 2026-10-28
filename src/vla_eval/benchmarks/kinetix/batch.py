"""Batched Kinetix: ``num_envs`` episodes as one ``jax.vmap`` over the functional env.

Every slot runs the exact single-env recipe (``KinetixBenchmark.reset`` / ``step``): its
own ``PRNGKey(seed + episode_idx)`` chain, the same key splits and the same action noise.
The batched state always has ``num_envs`` rows, so a tail batch keeps the compiled shape:
rows of empty or finished slots are reset / stepped with dummy inputs and never read.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval.benchmarks import batching
from vla_eval.benchmarks.base import Benchmark, StepResult
from vla_eval.recording import EpisodeRecorder
from vla_eval.types import Action, EpisodeResult, Observation, Task

from vla_eval.benchmarks.kinetix.benchmark import ACTION_DIM, KinetixBenchmark


def _batch_fns(bench: KinetixBenchmark) -> tuple[Any, Any]:
    """Jitted ``(reset, step)`` over ``num_envs`` rows for the current level."""
    if bench._jit_batch_reset is not None:
        return bench._jit_batch_reset, bench._jit_batch_step
    jax, jnp = bench._jax, bench._jnp
    assert jax is not None and jnp is not None
    env, params, level_state, noise_std = bench._env, bench._env_params, bench._level_state, bench._action_noise_std
    assert env is not None

    def reset_one(key: Any) -> Any:
        """Single-env reset: split the episode key, reset to the level's stored state."""
        rng, reset_rng = jax.random.split(key)
        obs, state = env.reset(reset_rng, params, override_reset_state=level_state)
        return rng, obs, state

    def step_one(rng: Any, state: Any, action: Any) -> Any:
        """Single-env step: split the key, add the optional action noise, step the env."""
        rng, step_rng = jax.random.split(rng)
        if noise_std > 0:
            noise_rng, step_rng = jax.random.split(step_rng)
            action = action + jax.random.normal(noise_rng, action.shape) * noise_std
        obs, state, reward, done, _info = env.step(step_rng, state, action, params)
        return rng, obs, state, reward, done

    bench._jit_batch_reset = jax.jit(jax.vmap(reset_one))
    bench._jit_batch_step = jax.jit(jax.vmap(step_one))
    return bench._jit_batch_reset, bench._jit_batch_step


class _BatchCore(batching.BatchCore):
    """Holds the batched JAX state of one batch; row ``i`` belongs to slot ``i``."""

    bench: KinetixBenchmark

    def __init__(self, bench: KinetixBenchmark, size: int) -> None:
        """Batch of ``size`` slots over the ``bench.batch_size()`` rows of the batched state."""
        super().__init__(bench, size)
        self._rows = bench.batch_size()
        self._rngs: Any = None
        self._states: Any = None
        self._recorders: dict[int, EpisodeRecorder] = {}
        self._steps: dict[int, int] = {}
        self._success: dict[int, bool] = {}

    def _row(self, tree: Any, slot: int) -> Any:
        """Row ``slot`` of every leaf in ``tree``."""
        jax = self.bench._jax
        assert jax is not None
        return jax.tree.map(lambda x: x[slot], tree)

    def _reset_batch(self, payloads: dict[int, tuple[Task, EpisodeRecorder]]) -> tuple[dict[int, Any], set[int]]:
        """Reset all rows (one shared level); each submitted slot seeds its row from its episode."""
        bench = self.bench
        bench._init_jax()
        jax, jnp = bench._jax, bench._jnp
        assert jax is not None and jnp is not None
        levels = {str(task["level"]) for task, _ in payloads.values()}
        if len(levels) != 1:
            raise RuntimeError(f"Kinetix: one batch must share one level, got {sorted(levels)}")
        level = levels.pop()
        if bench._env is None or bench._current_level != level:
            bench._make_env(level)
        reset, _ = _batch_fns(bench)
        seeds = np.zeros(self._rows, dtype=np.int64)
        for slot, (task, _) in payloads.items():
            seeds[slot] = bench._seed + int(task.get("episode_idx", 0))
        keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
        self._rngs, obs, self._states = reset(keys)
        obs = jax.device_get(obs)
        results: dict[int, Any] = {}
        for slot, (_, recorder) in payloads.items():
            self._recorders[slot] = recorder
            self._steps[slot] = 0
            self._success[slot] = False
            results[slot] = self._row(obs, slot)
            recorder.record_video(bench._extract_frame(results[slot]))
        return results, set()

    def _step_batch(self, payloads: dict[int, Action]) -> tuple[dict[int, StepResult], set[int]]:
        """Step all rows; submitted slots get their action, the other rows a zero action."""
        bench = self.bench
        jax, jnp = bench._jax, bench._jnp
        if jax is None or jnp is None or self._states is None:
            raise RuntimeError("Kinetix: step before reset")
        _, step = _batch_fns(bench)
        actions = np.zeros((self._rows, ACTION_DIM), dtype=np.float32)
        for slot, action in payloads.items():
            actions[slot] = bench._action_array(action)
        self._rngs, obs, self._states, reward, done = step(self._rngs, self._states, jnp.asarray(actions))
        obs, reward, done = jax.device_get((obs, reward, done))
        results: dict[int, StepResult] = {}
        finished: set[int] = set()
        for slot in payloads:
            self._steps[slot] += 1
            reward_val = float(reward[slot])
            if reward_val > 0:  # success if green touched blue at any step
                self._success[slot] = True
            done_val = bool(done[slot])
            if done_val or self._steps[slot] >= bench._max_episode_steps:
                finished.add(slot)
            raw_obs = self._row(obs, slot)
            recorder = self._recorders[slot]
            recorder.record_video(bench._extract_frame(raw_obs))
            recorder.record_step(reward=reward_val, done=done_val, success=self._success[slot])
            results[slot] = StepResult(
                obs=raw_obs, reward=reward_val, done=slot in finished, info={"success": self._success[slot]}
            )
        return results, finished

    def make_obs(self, slot: int, raw_obs: Any, task: Task) -> Observation:
        """The slot's observation, built exactly like the single-env one."""
        return self.bench.make_obs(raw_obs, task)

    def get_result(self, slot: int, last: StepResult) -> EpisodeResult:
        """Success if the slot's reward was ever positive."""
        return {"success": bool(last.info.get("success", False))}


def open_batch(bench: KinetixBenchmark, size: int) -> list[Benchmark]:
    """One view per slot over a fresh Kinetix batch core."""
    return batching.open_views(_BatchCore(bench, size))
