"""LADDER ``RoboLab-Long`` benchmark adapter (Isaac Sim 6 / Isaac Lab 3, Franka DROID)."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np

from vla_eval.benchmarks.ladder.base import _LadderVectorBenchmark, _ensure_ladder_on_path
from vla_eval.benchmarks.ladder.tasks import ROBOLAB_LITE_TASKS, ROBOLAB_TASKS
from vla_eval.dirs import assets_cache
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec

_T = TypeVar("_T")


class _RoboLabKit:
    """Process-wide Isaac Sim SimulationApp and single worker thread (Kit's async_engine needs its own thread)."""

    app: Any = None
    thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-robolab")
    env: Any = None


class LadderRoboLabBenchmark(_LadderVectorBenchmark):
    """LADDER ``RoboLab-Long`` benchmark (Franka DROID on Isaac Sim 6 / Isaac Lab 3, 44 tasks × 8 episodes)."""

    SUITE_NAME: ClassVar[str] = "ladder_robolab"
    DEFAULT_TASKS: ClassVar[tuple[str, ...]] = ROBOLAB_TASKS
    LITE_TASKS: ClassVar[tuple[str, ...]] = ROBOLAB_LITE_TASKS
    DEFAULT_MAX_STEPS: ClassVar[int] = 1500
    hard_exit: ClassVar[bool] = True
    step_dt = 1.0 / 15.0

    def __init__(
        self,
        num_envs: int = 8,
        tasks: list[str] | str | None = None,
        root: str | None = None,
        assets_dir: str | None = None,
        action_mode: str = "joint_pos",
        control_mode: str | None = None,
        max_episode_steps: int | None = None,
        success_mode: str | None = None,
        terminate_on_success: bool = True,
        send_state: bool = True,
        send_wrist_image: bool = True,
        expose_oracle: bool = False,
        expose_privileged: bool = False,
        publish_depth: bool = False,
        image_size: tuple[int, int] = (224, 224),
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
        self._image_size = (int(image_size[0]), int(image_size[1]))
        self._env: Any = None
        self._env_cfg: Any = None
        self._current_key: tuple[str, int] | None = None
        self._converter: Any = None
        self._raw_obs: Any = None
        self._oracle: Any = None
        self._step_count = 0

    def _on_kit_thread(self, fn: Callable[..., _T], *args: Any) -> _T:
        return _RoboLabKit.thread.submit(fn, *args).result()

    def _resolve_assets_dir(self) -> str:
        if "ROBOLAB_ASSET_DIR" in os.environ and Path(os.environ["ROBOLAB_ASSET_DIR"]).is_dir():
            return os.environ["ROBOLAB_ASSET_DIR"]
        candidates = [
            self._assets_dir,
            Path(os.environ["LADDER_ROBOLAB_ASSETS"]) if "LADDER_ROBOLAB_ASSETS" in os.environ else None,
            self._root / "ladder" / "components" / "inference" / "robots" / "robolab" / "assets",
            assets_cache("ladder_robolab") / "assets",
            assets_cache("ladder_robolab"),
        ]
        for cand in candidates:
            if cand is not None and cand.is_dir():
                os.environ["ROBOLAB_ASSET_DIR"] = str(cand)
                return str(cand)
        raise FileNotFoundError(
            "RoboLab assets not found. Set LADDER_ROBOLAB_ASSETS or mount to ~/.cache/vla-eval/assets/ladder_robolab."
        )

    def _launch_app(self) -> None:
        if _RoboLabKit.app is not None:
            return
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        self._resolve_assets_dir()
        _ensure_ladder_on_path(self._root)

        import signal
        import threading

        import cv2  # noqa: F401 -- must precede isaaclab import
        from isaaclab.app import AppLauncher

        orig_signal = signal.signal
        if threading.current_thread() is not threading.main_thread():
            setattr(signal, "signal", lambda *a, **kw: None)
        try:
            launcher = AppLauncher(headless=True, enable_cameras=True)
        finally:
            setattr(signal, "signal", orig_signal)
        _RoboLabKit.app = launcher.app

        import carb.settings
        import robolab.constants
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        from ladder.components.inference.robots.robolab.composition.specs import SPECS, task_target

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)
        auto_register_droid_envs(
            task_dirs=None,
            task=[task_target(t) if t in SPECS else t for t in self._task_names],
            cameras=None,
        )

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[Any]:
        return self._on_kit_thread(self._reset_wave_sync, task_name, len(episode_ids))

    def _reset_wave_sync(self, task_name: str, n_envs: int) -> list[Any]:
        self._launch_app()
        from robolab.core.environments.runtime import create_env

        from ladder.components.inference.robots.robolab.composition.specs import SPECS, task_class_name
        from ladder.components.inference.robots.robolab.droid_conversions import DroidJointposConverter
        from ladder.components.inference.robots.robolab.oracle import build_oracle

        cls_name = task_class_name(task_name) if task_name in SPECS else task_name
        key = (cls_name, n_envs)
        if self._env is None or self._current_key != key:
            if _RoboLabKit.env is not None:
                _RoboLabKit.env.close()
                _RoboLabKit.env = None
            self._env, self._env_cfg = create_env(
                cls_name,
                device="cuda:0",
                num_envs=n_envs,
                instruction_type="default",
                policy="ladder_runner",
                renderer="realtime",
                rendering_mode="balanced",
            )
            _RoboLabKit.env = self._env
            self._current_key = key
            self._oracle = build_oracle(getattr(self._env.cfg, "subtasks", None), self._env_cfg.instruction)
        else:
            self._env.reset_eval_state()

        obs, _ = self._env.reset()
        obs, _ = self._env.reset()  # native double reset
        self._raw_obs = obs
        self._step_count = 0
        self._converter = DroidJointposConverter(num_envs=n_envs, image_size=self._image_size)
        obs_list = self._build_obs_list()
        self._last_eef_poses = [np.asarray(o["state"]["eef_pose"], dtype=np.float32) for o in obs_list]
        self._last_target_poses = [np.asarray(o["state"]["eef_pose"], dtype=np.float32) for o in obs_list]
        self._last_joints = [np.asarray(o["state"]["joints"], dtype=np.float32)[:7] for o in obs_list]
        self._last_actions = [
            self._hold_robot_action(self._last_eef_poses[i], o["state"]["joints"]) for i, o in enumerate(obs_list)
        ]
        return obs_list

    def _step_wave(self, actions: list[Any]) -> list[Any]:
        return self._on_kit_thread(self._step_wave_sync, actions)

    def _step_wave_sync(self, actions: list[Any]) -> list[Any]:
        import omni.kit.app
        import omni.timeline
        import torch

        assert self._env is not None and self._raw_obs is not None
        joints = self._raw_obs["proprio_obs"]["arm_joint_pos"].detach().cpu().numpy()
        commands = self._converter.actions_to_jointpos(actions, joints)
        timeline = omni.timeline.get_timeline_interface()
        kit_app = omni.kit.app.get_app()
        while not timeline.is_playing():
            kit_app.update()
        obs, _, _, _, _ = self._env.step(torch.as_tensor(commands, device=self._env.device))
        self._raw_obs = obs
        self._step_count += 1
        return self._build_obs_list()

    def _build_obs_list(self) -> list[dict[str, Any]]:
        from ladder.components.inference.robots.robolab.composition.specs import (
            SPECS,
            decomp_cache_entries,
            task_class_name,
        )

        image_obs = self._raw_obs["image_obs"]
        proprio = self._raw_obs["proprio_obs"]
        results = self._env.get_env_results()
        active = set(self._env.active_env_ids)
        max_steps_reached = self._step_count >= int(self._env.max_episode_length)
        cls_name = self._current_key[0] if self._current_key else ""
        plan_entries = decomp_cache_entries()
        ordered_map = {task_class_name(k): s.ordered for k, s in SPECS.items()}
        revisit_map = {task_class_name(k): bool(s.via_legs) for k, s in SPECS.items()}

        achieved_all: list[list[bool]] | None = None
        plan_subtasks: list[str] | None = None
        if self._oracle:
            commands = [g.command for g in self._oracle]
            cached = (plan_entries.get(cls_name) or {}).get("subtasks")
            plan_subtasks = list(cached) if cached and len(cached) == len(commands) else commands
            try:
                achieved_all = [
                    [bool(g.check(self._env, env_id=i)) for g in self._oracle] for i in range(self._env.num_envs)
                ]
            except Exception:
                achieved_all = None

        out: list[dict[str, Any]] = []
        for i in range(self._env.num_envs):
            success = bool(results[i]["success"])
            truncated = max_steps_reached or i not in active
            oracle_block = (
                {
                    "subtasks": plan_subtasks,
                    "achieved": achieved_all[i],
                    "ordered": ordered_map.get(cls_name, False),
                    "revisits": revisit_map.get(cls_name, False),
                }
                if achieved_all is not None
                else {"subtasks": [self._env_cfg.instruction], "achieved": [success]}
            )
            metadata = {
                "env_name": cls_name,
                "robot_name": "franka",
                "instruction": self._env_cfg.instruction,
                "done": success,
                "truncated": truncated,
                "oracle": oracle_block,
            }
            out.append(
                self._converter.observation_dict(
                    env_i=i,
                    main_image=image_obs["over_shoulder_left_camera"][i].detach().cpu().numpy(),
                    wrist_image=image_obs["wrist_cam"][i].detach().cpu().numpy(),
                    joints=proprio["arm_joint_pos"][i].detach().cpu().numpy(),
                    gripper_position=proprio["gripper_pos"][i].detach().cpu().numpy(),
                    metadata=metadata,
                )
            )
        return out

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "main": IMAGE_RGB, "wrist": IMAGE_RGB, "language": LANGUAGE}
        if self._send_state:
            spec["state"] = RAW
        return spec

    def cleanup(self) -> None:
        self._env = None
