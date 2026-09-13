"""Push-T (gym-pusht) as a vla-eval benchmark: the adapter a project writes for its own env."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult, repeat_last_hold
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

ACTION_DIM = 2
MAX_EPISODE_STEPS = 300  # gym-pusht registration default
TASK_DESCRIPTION = "Push the T-shaped block onto the T-shaped target."

ACTION_XY_ABSOLUTE = DimSpec("position", ACTION_DIM, "absolute_xy", (0.0, 512.0))
STATE_AGENT_POS = DimSpec("state", ACTION_DIM, "agent_xy")


class PushTBenchmark(StepBenchmark):
    """Push-T (gym-pusht ``PushT-v0``).

    Observation: ``images["agentview"]`` (96x96x3 uint8), ``state`` (agent xy,
    float32), ``task_description``. Action: absolute agent target xy in
    ``[0, 512]``. Success: block/target coverage above 0.95 (``is_success``).

    Args:
        seed: Base seed; episode seed is ``seed + episode_idx``.
        max_episode_steps: Truncation horizon (default 300).
        image_size: Rendered observation side length in pixels.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "coverage", "success"})

    # pygame software rendering; no GL context involved.
    render_backends = frozenset({"gpu", "cpu"})

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        return {}

    def __init__(self, seed: int = 0, max_episode_steps: int = MAX_EPISODE_STEPS, image_size: int = 96) -> None:
        super().__init__()
        self._seed = seed
        self._max_episode_steps = max_episode_steps
        self._image_size = image_size
        self._env: Any = None
        self._step_count = 0
        self._last_info: dict[str, Any] = {}

    def _make_env(self) -> None:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import gym_pusht  # noqa: F401  (registers PushT-v0)
        import gymnasium as gym

        self._env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=self._image_size,
            observation_height=self._image_size,
            max_episode_steps=self._max_episode_steps,
        )

    def get_tasks(self) -> list[Task]:
        return [{"name": "push_t", "suite": "pusht", "task_description": TASK_DESCRIPTION}]

    def reset(self, task: Task) -> Any:
        if self._env is None:
            self._make_env()
        episode_seed = self._seed + int(task.get("episode_idx", 0))
        raw_obs, info = self._env.reset(seed=episode_seed)
        self._step_count = 0
        self._last_info = info
        return raw_obs

    def step(self, action: Action) -> StepResult:
        act = np.asarray(action["actions"], dtype=np.float32).reshape(-1)[:ACTION_DIM]
        act = np.clip(act, 0.0, 512.0)
        raw_obs, reward, terminated, truncated, info = self._env.step(act)
        self._step_count += 1
        self._last_info = info
        self._recorder.record_step(
            reward=float(reward), coverage=float(info.get("coverage", 0.0)), success=bool(info.get("is_success"))
        )
        return StepResult(obs=raw_obs, reward=float(reward), done=bool(terminated or truncated), info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        return {
            "images": {"agentview": np.ascontiguousarray(raw_obs["pixels"], dtype=np.uint8)},
            "state": np.asarray(raw_obs["agent_pos"], dtype=np.float32),
            "task_description": task.get("task_description", TASK_DESCRIPTION),
        }

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        info = step_result.info or self._last_info
        return {"success": bool(info.get("is_success", False)), "coverage": float(info.get("coverage", 0.0))}

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "coverage": "mean"}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": ACTION_XY_ABSOLUTE}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": STATE_AGENT_POS, "language": LANGUAGE}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_episode_steps, "action_dim": ACTION_DIM}

    def get_hold_action(self, last_action: Action | None) -> Action:
        # Absolute target control: repeating the last target holds the agent still.
        return repeat_last_hold(last_action, ACTION_DIM)

    def render(self) -> np.ndarray | None:
        return None if self._env is None else self._env.render()

    def cleanup(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
