"""LADDER ``RoboLab-Long`` adapter (Isaac Sim 6 / Isaac Lab 3, Franka DROID)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import signal
from typing import Any, ClassVar

import numpy as np

from vla_eval.benchmarks.ladder.base import EnvStep, LadderBenchmark
from vla_eval.specs import IMAGE_RGB, DimSpec

# Kit's async engine must own its thread, and the orchestrator's event loop owns the main one.
_KIT_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ladder-robolab")


def _image(frame: Any) -> np.ndarray:
    """A camera tensor, letterboxed to 224 x 224 as RoboLab's DROID client does."""
    from openpi_client.image_tools import resize_with_pad

    return np.asarray(resize_with_pad(frame.cpu().numpy(), 224, 224), dtype=np.uint8)


class LadderRoboLabBenchmark(LadderBenchmark):
    """LADDER ``RoboLab-Long``: Franka DROID on Isaac Sim 6 / Isaac Lab 3, 44 tasks x 8 episodes.

    Mirrors LADDER's RoboLab env server. Observations are the DROID exterior and wrist images plus
    ``[7 joints, gripper]`` (1 = closed); actions are ``[7 absolute joint targets, gripper]`` (> 0.5 = closed).
    """

    SUITE_NAME: ClassVar[str] = "ladder_robolab"
    TASK_FILE: ClassVar[str] = "robolab_long.txt"
    MAX_STEPS: ClassVar[int] = 4950  # longest task budget: 330 s at 15 Hz; each task's env truncates its own
    hard_exit: ClassVar[bool] = True

    def __init__(self, num_envs: int = 8, send_wrist_image: bool = True, **kwargs: Any) -> None:
        super().__init__(num_envs=num_envs, **kwargs)
        self._send_wrist_image = send_wrist_image
        self._env: Any = None
        self._env_key: tuple[str, int] | None = None
        self._instruction = ""
        self._oracle: Any = None
        self._converter: Any = None
        self._raw_obs: Any = None
        self._step_count = 0

    def _reset_wave(self, task_name: str, episode_ids: list[int]) -> list[EnvStep]:
        return _KIT_THREAD.submit(self._reset_sync, task_name, len(episode_ids)).result()

    def _step_wave(self, actions: list[Any]) -> list[EnvStep]:
        return _KIT_THREAD.submit(self._step_sync, actions).result()

    def _launch_app(self) -> None:
        import cv2  # noqa: F401 -- must precede the isaaclab import
        from isaaclab.app import AppLauncher

        # AppLauncher installs signal handlers, which only the main thread may do; this is the Kit thread.
        install = signal.signal
        signal.signal = lambda *a, **kw: None  # type: ignore[assignment]
        try:
            AppLauncher(headless=True, enable_cameras=True)
        finally:
            signal.signal = install

        import robolab.constants
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        from ladder.components.inference.robots.robolab.composition.specs import task_target

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        auto_register_droid_envs(task_dirs=None, task=[task_target(t) for t in self._task_names])

    def _reset_sync(self, task_name: str, n_envs: int) -> list[EnvStep]:
        if self._env is None:
            self._launch_app()
        from robolab.core.environments.runtime import create_env

        from ladder.components.inference.robots.robolab.composition.specs import task_class_name
        from ladder.components.inference.robots.robolab.droid_conversions import DroidJointposConverter
        from ladder.components.inference.robots.robolab.oracle import build_oracle

        key = (task_class_name(task_name), n_envs)
        if self._env_key != key:
            if self._env is not None:
                self._env.close()
            self._env, env_cfg = create_env(key[0], num_envs=n_envs, policy="ladder_runner")  # as LADDER's env server
            self._env_key = key
            self._instruction = env_cfg.instruction
            self._oracle = build_oracle(getattr(self._env.cfg, "subtasks", None), env_cfg.instruction)
        else:
            self._env.reset_eval_state()
        self._env.reset()
        self._raw_obs, _ = self._env.reset()  # RoboLab's own episode runner resets twice
        self._step_count = 0
        # Clamps each step's joint change and freezes a diverged arm, as LADDER's env server does.
        self._converter = DroidJointposConverter(num_envs=n_envs)
        self._actions = [self._joint_action(j) for j in self._joints()]
        return self._steps()

    def _step_sync(self, actions: list[Any]) -> list[EnvStep]:
        import omni.kit.app
        import omni.timeline
        import torch

        commands = self._converter.actions_to_jointpos(actions, self._joints())
        timeline, kit_app = omni.timeline.get_timeline_interface(), omni.kit.app.get_app()
        while not timeline.is_playing():
            kit_app.update()
        self._raw_obs, *_ = self._env.step(torch.as_tensor(commands, device=self._env.device))
        self._step_count += 1
        return self._steps()

    def _joints(self) -> np.ndarray:
        return self._raw_obs["proprio_obs"]["arm_joint_pos"].cpu().numpy()

    def _steps(self) -> list[EnvStep]:
        images, proprio = self._raw_obs["image_obs"], self._raw_obs["proprio_obs"]
        results = self._env.get_env_results()
        active = set(self._env.active_env_ids)
        out_of_time = self._step_count >= int(self._env.max_episode_length)
        steps = []
        for i in range(self._env.num_envs):
            success = bool(results[i]["success"])
            achieved = [bool(g.check(self._env, env_id=i)) for g in self._oracle] if self._oracle else [success]
            obs: dict[str, Any] = {
                "images": {"main": _image(images["over_shoulder_left_camera"][i])},
                "task_description": self._instruction,
                "task_name": self._task_name,
            }
            if self._send_wrist_image:
                obs["images"]["wrist"] = _image(images["wrist_cam"][i])
            if self._send_state:
                gripper = proprio["gripper_pos"][i].cpu().numpy().reshape(-1)[:1]  # RoboLab: 1 = closed
                obs["states"] = np.concatenate([proprio["arm_joint_pos"][i].cpu().numpy(), gripper]).astype(np.float32)
            steps.append(
                EnvStep(
                    obs=obs,
                    done=success,
                    truncated=out_of_time or i not in active,
                    num_goals=len(achieved),
                    num_complete=sum(achieved),
                )
            )
        return steps

    @staticmethod
    def _joint_action(joints: np.ndarray, gripper: float = 1.0) -> Any:
        """A ``ladder`` joint-position action; ``gripper`` uses LADDER's convention (1 = open)."""
        from ladder.components.inference.api.core.action import JointAnglesAction, ManipulatorAction

        return ManipulatorAction(
            arm_action=JointAnglesAction(angles=np.asarray(joints, dtype=np.float32)[:7]), gripper_action=gripper
        )

    def _decode_action(self, env_i: int, raw: np.ndarray) -> Any:
        return self._joint_action(raw[:7], gripper=0.0 if raw[7] > 0.5 else 1.0)

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec = super().get_observation_spec()
        if self._send_wrist_image:
            spec["wrist"] = IMAGE_RGB
        return spec

    def cleanup(self) -> None:
        self._env = None
