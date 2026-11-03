"""LADDER ``RobotArena-∞-Long`` benchmark adapter (Genesis, WidowX)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from vla_eval.benchmarks.ladder.base import _LadderVectorBenchmark, _ensure_ladder_on_path
from vla_eval.benchmarks.ladder.tasks import ROBOT_ARENA_K8_OFFSETS, ROBOT_ARENA_TASKS
from vla_eval.dirs import assets_cache
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec


class LadderRobotArenaBenchmark(_LadderVectorBenchmark):
    """LADDER ``RobotArena-∞-Long`` benchmark (WidowX on Genesis, 47 tasks × 8 episodes)."""

    SUITE_NAME: ClassVar[str] = "ladder_robot_arena"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = ROBOT_ARENA_TASKS
    DEFAULT_MAX_STEPS: ClassVar[int] = 1000
    step_dt = 0.2

    def __init__(self, num_envs: int = 8, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(num_envs=num_envs, **kwargs)
        self._seed = int(seed)
        self._env: Any = None
        self._padded_single = False

    def _resolve_data_root(self) -> str:
        candidates = [
            self._assets_dir,
            Path(os.environ["LADDER_ROBOT_ARENA_ASSETS"]) if "LADDER_ROBOT_ARENA_ASSETS" in os.environ else None,
            self._root
            / "ladder"
            / "components"
            / "inference"
            / "robots"
            / "robot_arena"
            / "robot_arena_data"
            / "examples"
            / "data",
            assets_cache("ladder_robot_arena") / "examples" / "data",
            assets_cache("ladder_robot_arena"),
        ]
        for cand in candidates:
            if cand is not None and ((cand / "bridge").is_dir() or (cand / "scenes").is_dir()):
                return str(cand)
        raise FileNotFoundError(
            "RobotArena assets not found (expected a directory containing 'bridge/'). "
            "Set LADDER_ROBOT_ARENA_ASSETS or mount to ~/.cache/vla-eval/assets/ladder_robot_arena."
        )

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        _ensure_ladder_on_path(self._root)
        from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
        from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnv, RobotArenaEnvConfig
        from ladder.components.inference.robots.robot_arena.scene_builder import set_genesis_seed
        from ladder.core.compute.seeder import SEEDER

        SEEDER.seed_everything(self._seed)
        set_genesis_seed(self._seed)

        comp_task = COMPOSITION_TASKS[task_name]
        offsets = [ROBOT_ARENA_K8_OFFSETS[ep % len(ROBOT_ARENA_K8_OFFSETS)] for ep in episode_ids]
        # RobotArenaEnv's K>=2 path requires len(per_env_object_offsets) >= 2 and skips overlap
        # resolution to match the published K=8 protocol; pad a 1-slot wave to 2 and slice slot 0.
        self._padded_single = len(offsets) == 1
        if self._padded_single:
            offsets = [offsets[0], ROBOT_ARENA_K8_OFFSETS[0]]

        cfg = RobotArenaEnvConfig(
            data_root=self._resolve_data_root(),
            scene_name=comp_task.scene,
            robot_name="widowx",
            max_episode_steps=comp_task.max_episode_steps,
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

        obs_list = self._env.reset()
        if self._padded_single:
            obs_list = obs_list[:1]
        self._last_eef_poses = [np.asarray(o.state.eef_pose, dtype=np.float32) for o in obs_list]
        self._last_target_poses = [np.asarray(o.state.eef_pose, dtype=np.float32) for o in obs_list]
        self._last_joints = [None] * len(obs_list)
        self._last_actions = [self._hold_robot_action(p) for p in self._last_target_poses]
        return obs_list

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        assert self._env is not None
        step_actions = [actions[0], actions[0]] if self._padded_single else actions
        obs_list = self._env.step(step_actions)
        return obs_list[:1] if self._padded_single else obs_list

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "main": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = RAW
        return spec

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
