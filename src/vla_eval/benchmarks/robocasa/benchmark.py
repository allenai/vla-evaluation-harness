"""RoboCasa benchmark implementation (original pre-v1 release).

RoboCasa is a simulation framework for kitchen manipulation built on
robosuite v1.5 and MuJoCo, providing 24 atomic tasks plus multi-stage
tasks across 100+ procedurally-generated kitchen layouts.

Actions are 7-D — ``[dx, dy, dz, drx, dry, drz, gripper]`` — and are
zero-padded to the robot's full action space, which holds the mobile
base and torso still for the ``PandaOmron`` default.

Observations expose RGB images from configurable cameras (default:
``robot0_agentview_left`` and ``robot0_eye_in_hand``) plus a natural
language task description obtained via ``env.get_ep_meta()["lang"]``.

The successor protocol (365 tasks, 12-D mobile-manipulation actions) lives
in :mod:`vla_eval.benchmarks.robocasa365` and runs on its own image; the
two upstream releases are not API-compatible.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_EULER, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

ACTION_DIM = 7

# The benchmark's 24 atomic tasks: robocasa's ``SINGLE_STAGE_TASK_DATASETS``
# minus ``NavigateKitchen``, which is locomotion rather than manipulation.
ATOMIC_TASKS = [
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
]


# Evaluation distribution of robocasa's own ``eval_utils.create_eval_env``:
# held-out object instances in five fixed layout/style pairs.  That helper
# cannot be called directly because it imports ``load_controller_config``,
# which robosuite v1.5 replaced with the composite-controller loader.
EVAL_LAYOUT_AND_STYLE_IDS = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))


def _task_horizon(task_name: str) -> int:
    from robocasa.utils.dataset_registry import MULTI_STAGE_TASK_DATASETS, SINGLE_STAGE_TASK_DATASETS

    for registry in (SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS):
        if task_name in registry:
            return int(registry[task_name]["horizon"])
    raise ValueError(f"task is not in the RoboCasa registry: {task_name}")


class RoboCasaBenchmark(StepBenchmark):
    """RoboCasa kitchen manipulation benchmark.

    Args:
        tasks: RoboCasa environment names to evaluate.  Defaults to the 24
            atomic tasks.
        robot: Robot model name (default ``"PandaOmron"``).
        camera_names: Camera names for observations.
        camera_size: Camera resolution (square, default 256).
        max_steps: Override the per-task registry horizon.  Leave unset for the
            benchmark protocol; use it to cut episodes short in smoke runs.
        obj_instance_split: Object-instance split — ``"B"`` (the held-out
            evaluation instances), ``"A"`` (the instances the released human
            demonstrations were collected with), or ``None`` for all.
        eval_scenes: Restrict layouts and styles to the benchmark's five fixed
            evaluation scenes.  Disable to sample the full scene distribution.
        seed: Base seed; episode ``i`` of each task runs at ``seed + i``.
            ``None`` leaves the environment unseeded.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "done", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        robot: str = "PandaOmron",
        camera_names: list[str] | None = None,
        camera_size: int = 256,
        max_steps: int | None = None,
        obj_instance_split: str | None = "B",
        eval_scenes: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if obj_instance_split not in {"A", "B", None}:
            raise ValueError("obj_instance_split must be 'A', 'B', or None")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be positive when set")
        self._task_names = tasks or ATOMIC_TASKS
        self._robot = robot
        self._camera_names = camera_names or [
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ]
        self._camera_size = camera_size
        self._max_steps = max_steps
        self._obj_instance_split = obj_instance_split
        self._eval_scenes = eval_scenes
        self._seed = seed
        self._env: Any = None
        self._current_task: str | None = None
        self._lang: str = ""
        self._horizon = 0
        self._steps = 0

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                # Cleanup must not mask an episode or runner failure.
                pass
            finally:
                self._env = None

    def get_tasks(self) -> list[Task]:
        return [{"name": t} for t in self._task_names]

    def _make_env(self, task_name: str) -> Any:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from robocasa.utils.env_utils import create_env

        return create_env(
            env_name=task_name,
            robots=self._robot,
            camera_names=self._camera_names,
            camera_widths=self._camera_size,
            camera_heights=self._camera_size,
            render_onscreen=False,
            obj_instance_split=self._obj_instance_split,
            layout_and_style_ids=EVAL_LAYOUT_AND_STYLE_IDS if self._eval_scenes else None,
            seed=self._seed,
        )

    def reset(self, task: Task) -> Any:
        task_name = task["name"]

        # Reuse env for same task across episodes
        if self._env is None or self._current_task != task_name:
            if self._env is not None:
                self._env.close()
            self._env = self._make_env(task_name)
            self._current_task = task_name

        # Layout, style and object instances are drawn from ``env.rng`` during
        # reset, so reseeding here makes each episode reproducible without
        # paying for a full environment rebuild.
        if self._seed is not None:
            self._env.rng = np.random.default_rng(self._seed + int(task.get("episode_idx", 0)))

        obs = self._env.reset()
        self._lang = self._env.get_ep_meta().get("lang", task_name)
        self._steps = 0
        self._horizon = self._max_steps or _task_horizon(task_name)
        self._recorder.record_video(self._extract_frame(obs))
        return obs

    def step(self, action: Action) -> StepResult:
        raw_action = action.get("actions", action.get("action"))
        if raw_action is None:
            raw_action = np.zeros(ACTION_DIM)
        raw_action = np.asarray(raw_action, dtype=np.float64)
        if raw_action.shape != (ACTION_DIM,):
            raise ValueError(f"RoboCasa expected a {ACTION_DIM}-D action, got {raw_action.shape}")

        # Zero-pad the arm action up to the robot's full action space, which
        # holds the mobile base and torso still.
        env_action = np.zeros(self._env.action_spec[0].shape[0])
        if env_action.shape[0] < ACTION_DIM:
            raise ValueError(f"robot {self._robot!r} has fewer than {ACTION_DIM} action dimensions")
        env_action[:ACTION_DIM] = raw_action

        obs, _, _, info = self._env.step(env_action)
        # create_env passes ignore_done=True, so the environment never reports
        # termination; the registry horizon is the authority.
        self._steps += 1
        done = self._steps >= self._horizon
        success = bool(self._env._check_success())
        info["success"] = success
        self._recorder.record_video(self._extract_frame(obs))
        self._recorder.record_step(reward=float(success), done=done, success=success)
        return StepResult(obs=obs, reward=float(success), done=done, info=info)

    def _extract_frame(self, raw_obs: Any) -> np.ndarray | None:
        if not isinstance(raw_obs, dict):
            return None
        for cam in self._camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                # Match make_obs's vertical flip so recorded video matches what the model sees.
                return np.ascontiguousarray(raw_obs[key][::-1])
        return None

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images: dict[str, Any] = {}
        for cam in self._camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                # RoboCasa images are upside-down — flip vertically
                images[cam] = np.ascontiguousarray(raw_obs[key][::-1])
        return {
            "images": images,
            "task_description": self._lang,
        }

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done or step_result.info.get("success", False)

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": step_result.info.get("success", False)}

    def get_metadata(self) -> dict[str, Any]:
        # Episodes stop at their own registry horizon; the runner-wide cap has
        # to clear the longest of them.
        horizons = [self._max_steps] if self._max_steps else [_task_horizon(t) for t in self._task_names]
        return {
            "max_steps": max(horizons),
            "obj_instance_split": self._obj_instance_split,
            "eval_scenes": self._eval_scenes,
            "seed": self._seed,
        }

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "robot0_agentview_left": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def render(self) -> np.ndarray | None:
        try:
            return self._env.render()
        except Exception:
            return None
