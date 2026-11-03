"""LADDER ``SimplerEnv-Long`` benchmark adapter (SAPIEN 3 / ManiSkill 3, WidowX)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from vla_eval.benchmarks.ladder.base import _LadderVectorBenchmark, _ensure_ladder_on_path
from vla_eval.benchmarks.ladder.tasks import SIMPLER_LITE_TASKS, SIMPLER_TASKS
from vla_eval.dirs import assets_cache
from vla_eval.render import DEFAULT_RENDER_MODE, assert_lavapipe_vulkan, configure_sapien_render
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec

SIMPLER_LAVAPIPE_ICD_ENV_VAR = "LADDER_SIMPLER_LAVAPIPE_ICD"


class LadderSimplerBenchmark(_LadderVectorBenchmark):
    """LADDER ``SimplerEnv-Long`` benchmark (WidowX on SAPIEN 3 / ManiSkill 3, 54 tasks × 24 episodes)."""

    SUITE_NAME: ClassVar[str] = "ladder_simpler"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = SIMPLER_TASKS
    LITE_TASKS: ClassVar[tuple[str, ...]] = SIMPLER_LITE_TASKS
    DEFAULT_MAX_STEPS: ClassVar[int] = 1000
    step_dt = 0.2

    render_backends = frozenset({"gpu", "cpu"})
    _render_mode: ClassVar[str] = DEFAULT_RENDER_MODE

    @classmethod
    def configure_render(cls, mode: str) -> dict[str, str]:
        applied = configure_sapien_render(mode, SIMPLER_LAVAPIPE_ICD_ENV_VAR)
        cls._render_mode = mode
        return applied

    def __init__(
        self,
        num_envs: int = 24,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "delta_ee",
        control_mode: str | None = None,
        max_episode_steps: int | None = None,
        success_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
        expose_oracle: bool = False,
        expose_privileged: bool = False,
        publish_depth: bool = False,
    ) -> None:
        super().__init__(
            num_envs=num_envs,
            tasks=tasks,
            root=root,
            assets_dir=assets_dir,
            action_mode=action_mode,
            control_mode=control_mode,
            max_episode_steps=max_episode_steps,
            success_mode=success_mode,
            terminate_on_success=terminate_on_success,
            send_state=send_state,
            send_wrist_image=send_wrist_image,
            expose_oracle=expose_oracle,
            expose_privileged=expose_privileged,
            publish_depth=publish_depth,
        )
        self._env: Any = None
        self._current_key: tuple[str, int] | None = None

    def _ensure_assets_env(self) -> None:
        if "MS_ASSET_DIR" in os.environ:
            return
        candidates = [
            self._assets_dir,
            Path(os.environ["LADDER_SIMPLER_ASSETS"]) if "LADDER_SIMPLER_ASSETS" in os.environ else None,
            self._root / "ladder" / "components" / "inference" / "robots" / "simpler" / "maniskill3",
            assets_cache("ladder_simpler"),
        ]
        for cand in candidates:
            if cand is not None and cand.is_dir():
                os.environ["MS_ASSET_DIR"] = str(cand)
                return

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        if self._render_mode == "cpu":
            assert_lavapipe_vulkan(type(self).__name__)
        _ensure_ladder_on_path(self._root)
        self._ensure_assets_env()

        import gymnasium as gym
        import torch

        from ladder.components.inference.robots.simpler.env_builder import ABS_CONTROL_MODE
        from ladder.components.inference.robots.simpler.simpler_env import SimplerEnv, SimplerEnvConfig

        n_envs = len(episode_ids)
        key = (task_name, n_envs)
        if self._env is None or self._current_key != key:
            if self._env is not None:
                self._env.close()
                self._env = None
            cfg = SimplerEnvConfig(
                env_name=f"{task_name}-v0",
                robot_name="widowx",
                scene_name="bridge_table_1_v1",
                rgb_overlay_path="",
                sim_freq=500,
                control_mode=ABS_CONTROL_MODE,
                control_freq=5,
                max_episode_steps=0,
                raytracing=False,
                n_envs=n_envs,
                episode_layouts=max(24, max(episode_ids, default=0) + 1),
                expose_privileged=self._expose_privileged,
                publish_depth=self._publish_depth,
            )
            if self._render_mode == "cpu" or not torch.cuda.is_available():
                from mani_skill.utils.structs.types import SimConfig

                def _create_cpu_env(self_env: Any) -> Any:
                    return gym.make(
                        f"{task_name}-v1",
                        num_envs=n_envs,
                        obs_mode="rgb+depth+segmentation" if self._publish_depth else "rgb+segmentation",
                        sim_backend="cpu",
                        control_mode=ABS_CONTROL_MODE,
                        max_episode_steps=None,
                        sim_config=SimConfig(sim_freq=500, control_freq=5),
                        n_layouts=cfg.episode_layouts,
                    )

                SimplerEnv._create_env = _create_cpu_env  # type: ignore[method-assign]
            self._env = SimplerEnv(cfg)
            self._current_key = key

        obs_list = self._env.reset(episode_ids)
        self._last_eef_poses = [np.asarray(o.state.eef_pose, dtype=np.float32) for o in obs_list]
        self._last_target_poses = [
            np.asarray(getattr(o.state, "eef_pose_from_target", o.state.eef_pose), dtype=np.float32) for o in obs_list
        ]
        self._last_joints = [
            np.asarray(o.state.joints, dtype=np.float32)[:7] if getattr(o.state, "joints", None) is not None else None
            for o in obs_list
        ]
        self._last_actions = [self._hold_robot_action(p) for p in self._last_target_poses]
        return obs_list

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        assert self._env is not None
        return self._env.step(actions)

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
            self._current_key = None
