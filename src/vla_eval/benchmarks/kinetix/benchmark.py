"""Kinetix benchmark implementation for RTC evaluation.

Kinetix is a JAX-based 2D physics engine with dynamic manipulation tasks
(throwing, catching, balancing, locomotion). This benchmark wraps the 12
tasks used in the RTC paper (arXiv:2506.07339) for evaluation under both
sync and sim2live conditions.

The environment uses a gymnax-style functional API where state is passed
explicitly on every call. This adapter stores JAX state as instance
variables and bridges to the StepBenchmark async interface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

# Action space: 4 motor bindings + 2 thruster bindings = 6 dims
ACTION_DIM = 6

# Default max timesteps per episode (RTC paper)
MAX_TIMESTEPS = 256

# The 12 tasks from the RTC paper (arXiv:2506.07339, Table 1).
# Each level file lives under the "l/" (large) size directory.
RTC_12_TASKS = [
    {"name": "Grasp Easy", "level": "grasp_easy"},
    {"name": "Catapult", "level": "catapult"},
    {"name": "Cartpole Thrust", "level": "cartpole_thrust"},
    {"name": "Hard Lunar Lander", "level": "hard_lunar_lander"},
    {"name": "Half Cheetah", "level": "mjc_half_cheetah"},
    {"name": "Swimmer", "level": "mjc_swimmer"},
    {"name": "Walker", "level": "mjc_walker"},
    {"name": "Unicycle", "level": "h17_unicycle"},
    {"name": "Chain Lander", "level": "chain_lander"},
    {"name": "Catcher", "level": "catcher_v3"},
    {"name": "Trampoline", "level": "trampoline"},
    {"name": "Car Launch", "level": "car_launch"},
]


def _resolve_level_path(level: str, rtc_worlds_dir: str | None) -> str:
    """Resolve a level name to a loadable path.

    Search order:
    1. ``rtc_worlds_dir/l/<level>.json`` (RTC repo custom levels)
    2. ``l/<level>.json`` (kinetix built-in levels, resolved by kinetix itself)
    """
    if rtc_worlds_dir is not None:
        candidate = Path(rtc_worlds_dir) / "l" / f"{level}.json"
        if candidate.is_file():
            return str(candidate)

    # Fall back to kinetix's built-in level path format
    return f"l/{level}.json"


class KinetixBenchmark(StepBenchmark):
    """Kinetix 2D physics benchmark (12 RTC tasks).

    Non-obvious behaviors:
        - **JAX functional API**: Kinetix uses gymnax-style functional state.
          ``env.step(rng, state, action, params)`` returns new state — no
          mutation. JAX RNG keys are split on every step.
        - **Pixel observations**: Rendered from the 2D physics state. Default
          resolution is 125×125 (screen_dim=500, downscale=4). Resized to
          ``image_size`` via PIL if different.
        - **Symbolic state**: Also included in observations under ``"state"``
          for models (like RTC) that use symbolic input.
        - **Env recreation per task**: Each task loads a different level file,
          so the env (with its reset_fn) is recreated when the task changes.

    Args:
        tasks: Subset of task names to evaluate. None = all 12 RTC tasks.
        max_episode_steps: Max steps per episode. Default 256 (RTC paper).
        seed: Base random seed.
        rtc_worlds_dir: Path to the RTC repo ``worlds/`` directory for custom
            levels. Falls back to kinetix built-in levels if None.
        observation_type: ``"pixels"`` (default) for rendered images, or
            ``"symbolic"`` for the flat symbolic state vector used by RTC.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        max_episode_steps: int = MAX_TIMESTEPS,
        seed: int = 0,
        rtc_worlds_dir: str | None = None,
        observation_type: str = "pixels",
        action_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self._task_names = tasks
        self._max_episode_steps = max_episode_steps
        self._seed = seed
        self._rtc_worlds_dir = rtc_worlds_dir
        if observation_type not in ("pixels", "symbolic"):
            raise ValueError(f"observation_type must be 'pixels' or 'symbolic', got {observation_type!r}")
        self._observation_type = observation_type
        self._action_noise_std = action_noise_std

        # Lazy-initialized JAX state
        self._env = None
        self._env_state = None
        self._env_params = None
        self._static_env_params = None
        self._rng = None
        self._current_level: str | None = None
        self._level_state = None
        self._step_count = 0
        self._episode_success = False
        self._jax = None
        self._jnp = None

    def _init_jax(self) -> None:
        """Lazily import JAX (heavy import)."""
        if self._jax is not None:
            return
        import jax
        import jax.numpy as jnp

        self._jax = jax
        self._jnp = jnp
        logger.info("JAX initialized, devices: %s", jax.devices())

    def _make_env(self, level: str) -> None:
        """Create a Kinetix environment for the given level."""
        self._init_jax()
        assert self._jax is not None
        jax = self._jax

        from kinetix.environment import env as kenv
        from kinetix.util.saving import load_from_json_file

        level_path = _resolve_level_path(level, self._rtc_worlds_dir)
        level_state, level_static_params, level_env_params = load_from_json_file(level_path)

        # Use level's own static params (they encode the correct physics config)
        static_env_params = level_static_params
        env_params = level_env_params.replace(max_timesteps=self._max_episode_steps)

        env_name = (
            "Kinetix-Symbolic-Continuous-v1"
            if self._observation_type == "symbolic"
            else "Kinetix-Pixels-Continuous-v1"
        )
        env = kenv.make_kinetix_env_from_name(env_name, static_env_params=static_env_params)

        self._env = env
        self._env_params = env_params
        self._static_env_params = static_env_params
        self._level_state = level_state
        self._current_level = level

        # JIT compile step_env and reset_env_to_level (raw env, no wrappers)
        self._jit_step = jax.jit(env.step_env)
        self._jit_reset = jax.jit(env.reset_env_to_level)

        logger.info("Kinetix env created for level: %s (path: %s)", level, level_path)

    def cleanup(self) -> None:
        self._env = None
        self._env_state = None
        self._level_state = None
        self._rng = None
        self._current_level = None

    def get_tasks(self) -> list[Task]:
        if self._task_names is not None:
            name_set = set(self._task_names)
            return [t for t in RTC_12_TASKS if t["name"] in name_set]
        return list(RTC_12_TASKS)

    def reset(self, task: Task) -> Any:
        self._init_jax()
        assert self._jax is not None
        jax = self._jax

        level = task["level"]
        episode_idx = task.get("episode_idx", 0)

        # Recreate env when the level changes
        if self._env is None or self._current_level != level:
            self._make_env(level)

        # Deterministic seed per episode
        rng = jax.random.PRNGKey(self._seed + episode_idx)
        rng, reset_rng = jax.random.split(rng)

        # reset_to_level: load the stored level state (matches RTC's eval pattern)
        obs, env_state = self._jit_reset(reset_rng, self._level_state, self._env_params)
        self._env_state = env_state
        self._rng = rng
        self._step_count = 0
        self._episode_success = False

        return obs

    def step(self, action: Action) -> StepResult:
        assert self._jax is not None and self._jnp is not None
        jax = self._jax
        jnp = self._jnp

        raw_action = action.get("actions", action.get("action"))
        raw_action = np.atleast_1d(np.asarray(raw_action, dtype=np.float32))
        assert raw_action.shape[-1] == ACTION_DIM, (
            f"Action dimension mismatch: got {raw_action.shape[-1]}, expected {ACTION_DIM}"
        )

        # Pad or truncate to ACTION_DIM
        if raw_action.shape[-1] < ACTION_DIM:
            raw_action = np.pad(raw_action, (0, ACTION_DIM - raw_action.shape[-1]))
        elif raw_action.shape[-1] > ACTION_DIM:
            raw_action = raw_action[..., :ACTION_DIM]

        jax_action = jnp.array(raw_action)

        assert self._rng is not None
        self._rng, step_rng = jax.random.split(self._rng)

        # NoisyActionWrapper: add Gaussian noise to actions before env step.
        # Mirrors RTC's NoisyActionWrapper(std=0.1) exactly: split the step key,
        # use key1 for noise, pass key2 to env.step.
        if self._action_noise_std > 0:
            noise_rng, step_rng = jax.random.split(step_rng)
            jax_action = jax_action + jax.random.normal(noise_rng, jax_action.shape) * self._action_noise_std

        obs, env_state, reward, done, info = self._jit_step(step_rng, self._env_state, jax_action, self._env_params)
        self._env_state = env_state
        self._step_count += 1

        # Convert JAX scalars to Python
        reward_val = float(reward)
        done_val = bool(done)

        # Track success across all steps (GoalR > 0 means green touched blue)
        if reward_val > 0:
            self._episode_success = True

        return StepResult(obs=obs, reward=reward_val, done=done_val, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        if self._observation_type == "symbolic":
            # raw_obs is a flat JAX array (symbolic state vector, ~679-dim)
            state = np.asarray(raw_obs, dtype=np.float32)
            return {"state": state, "task_description": task["name"]}

        # raw_obs is a PixelsObservation with .image (H×W×3 float32 in [0,1])
        img = np.asarray(raw_obs.image)

        # Convert float32 [0,1] to uint8 [0,255]
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

        obs_dict: dict[str, Any] = {
            "images": {"viewport": img},
            "task_description": task["name"],
        }
        return obs_dict

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done or self._step_count >= self._max_episode_steps

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        # Success if reward > 0 at any step during the episode
        return {"success": self._episode_success}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_episode_steps,
            "action_dim": ACTION_DIM,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"action": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "viewport": IMAGE_RGB,
            "language": LANGUAGE,
        }
