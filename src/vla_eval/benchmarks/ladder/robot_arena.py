"""LADDER ``RobotArena-∞-Long`` adapter (Genesis, WidowX)."""

from __future__ import annotations

import os
from typing import Any, ClassVar

from vla_eval.benchmarks.ladder.base import LADDER_ROOT, EnvStep, WidowXBenchmark

_K8_CONFIG = LADDER_ROOT / "ladder/components/inference/robots/robot_arena/configs/widowx_composition_k8.yaml"


class LadderRobotArenaBenchmark(WidowXBenchmark):
    """LADDER ``RobotArena-∞-Long``: WidowX on Genesis, 47 tasks x 8 episodes.

    Episode ``k`` of a task shifts the scene's objects by the ``k``-th offset of the published 8-episode grid.
    """

    SUITE_NAME: ClassVar[str] = "ladder_robot_arena"
    TASK_FILE: ClassVar[str] = "robotarena_long.txt"

    def __init__(self, num_envs: int = 8, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(num_envs=num_envs, **kwargs)
        from ladder.core.databib.config.yaml_util import load_config

        self._offsets = [tuple(o) for o in load_config(_K8_CONFIG)["test_config"]["object_variation_offsets_m"]]
        self._seed = seed
        self._env: Any = None
        self._padded = False

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[EnvStep]:
        from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
        from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnv, RobotArenaEnvConfig
        from ladder.components.inference.robots.robot_arena.scene_builder import set_genesis_seed
        from ladder.core.compute.seeder import SEEDER

        SEEDER.seed_everything(self._seed)
        set_genesis_seed(self._seed)
        offsets = [self._offsets[ep % len(self._offsets)] for ep in episode_ids]
        # The published protocol runs RobotArenaEnv's batched (K >= 2) path; a 1-episode wave is padded to 2.
        self._padded = len(offsets) == 1
        if self._padded:
            offsets.append(self._offsets[0])
        task = COMPOSITION_TASKS[task_name]
        cfg = RobotArenaEnvConfig(
            data_root=os.environ["LADDER_ROBOT_ARENA_ASSETS"],
            scene_name=task.scene,
            robot_name="widowx",
            max_episode_steps=task.max_episode_steps,
            control_freq=5,
            sim_freq=200,
            sim_substeps=1,
            rigid_constraint_timeconst=0.02,
            camera_width=640,
            camera_height=480,
            per_env_object_offsets=offsets,
            composition_task=task_name,
        )
        if self._env is not None and self._env.can_reuse_for(cfg):
            self._env.reconfigure(cfg)
        else:
            if self._env is not None:
                self._env.close()
            self._env = RobotArenaEnv(config=cfg)
        steps = self._to_steps(self._env.reset()[: len(episode_ids)])
        self._actions = self._hold_actions()
        return steps

    def _step_wave(self, actions: list[Any]) -> list[EnvStep]:
        obs = self._env.step(actions + actions[:1] if self._padded else actions)
        return self._to_steps(obs[: len(actions)])

    def cleanup(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
