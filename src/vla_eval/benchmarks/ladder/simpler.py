"""LADDER ``SimplerEnv-Long`` adapter (SAPIEN 3 / ManiSkill 3, WidowX)."""

from __future__ import annotations

from typing import Any, ClassVar

from vla_eval.benchmarks.ladder.base import EnvStep, WidowXBenchmark


class LadderSimplerBenchmark(WidowXBenchmark):
    """LADDER ``SimplerEnv-Long``: WidowX on SAPIEN 3 / ManiSkill 3, 54 tasks x 24 episodes."""

    SUITE_NAME: ClassVar[str] = "ladder_simpler"
    TASK_FILE: ClassVar[str] = "simplerenv_long.txt"

    def __init__(self, num_envs: int = 24, **kwargs: Any) -> None:
        super().__init__(num_envs=num_envs, **kwargs)
        self._env: Any = None
        self._env_key: tuple[str, int] | None = None

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[EnvStep]:
        from ladder.components.inference.robots.simpler.env_builder import ABS_CONTROL_MODE
        from ladder.components.inference.robots.simpler.simpler_env import SimplerEnv, SimplerEnvConfig

        key = (task_name, len(episode_ids))
        if self._env_key != key:
            if self._env is not None:
                self._env.close()
            self._env = SimplerEnv(
                SimplerEnvConfig(
                    env_name=f"{task_name}-v0",
                    robot_name="widowx",
                    scene_name="bridge_table_1_v1",
                    rgb_overlay_path="",
                    sim_freq=500,
                    control_mode=ABS_CONTROL_MODE,
                    control_freq=5,
                    max_episode_steps=0,  # the task's own budget
                    raytracing=False,
                    n_envs=len(episode_ids),
                    episode_layouts=max(24, max(episode_ids) + 1),
                )
            )
            self._env_key = key
        steps = self._to_steps(self._env.reset(episode_ids))
        self._actions = self._hold_actions()
        return steps

    def _step_wave(self, actions: list[Any]) -> list[EnvStep]:
        return self._to_steps(self._env.step(actions))

    def cleanup(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
