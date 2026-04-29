"""BEHAVIOR-1K benchmark implementation.

BEHAVIOR-1K is a long-horizon household-activity benchmark built on
OmniGibson (NVIDIA Isaac Sim).  The 2025 BEHAVIOR Challenge defines a
50-task evaluation suite (B10/B20/B30/B40/B50) using the R1Pro
mobile-manipulation robot.

References:
    - https://behavior.stanford.edu
    - https://github.com/StanfordVL/BEHAVIOR-1K
    - OmniGibson/omnigibson/learning/eval.py (official Evaluator)

Key facts:
    - Robot: R1Pro (23-D absolute joint-position action space).
    - Action layout (matching ``ACTION_QPOS_INDICES["R1Pro"]``):
        base[0:3], torso[3:7], left_arm[7:14], left_gripper[14:15],
        right_arm[15:22], right_gripper[22:23].
    - Cameras: head 720x720, left_wrist 480x480, right_wrist 480x480.
    - Success: ``info["done"]["success"]`` (binary); the challenge
      separately reports a partial Q-score, but we only surface the
      binary flag here — partial scoring lives in the official
      ``score_utils.compute_final_q_score``.
    - Max steps default: 5000 (or 2× human demo length when known).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anyio
import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

# 50-task BEHAVIOR Challenge 2025 evaluation suite.
# Mirrors omnigibson.learning.utils.eval_utils.TASK_NAMES_TO_INDICES.
B50_TASKS: list[str] = [
    # B10
    "turning_on_radio",
    "picking_up_trash",
    "putting_away_Halloween_decorations",
    "cleaning_up_plates_and_food",
    "can_meat",
    "setting_mousetraps",
    "hiding_Easter_eggs",
    "picking_up_toys",
    "rearranging_kitchen_furniture",
    "putting_up_Christmas_decorations_inside",
    # B20
    "set_up_a_coffee_station_in_your_kitchen",
    "putting_dishes_away_after_cleaning",
    "preparing_lunch_box",
    "loading_the_car",
    "carrying_in_groceries",
    "bringing_in_wood",
    "moving_boxes_to_storage",
    "bringing_water",
    "tidying_bedroom",
    "outfit_a_basic_toolbox",
    # B30
    "sorting_vegetables",
    "collecting_childrens_toys",
    "putting_shoes_on_rack",
    "boxing_books_up_for_storage",
    "storing_food",
    "clearing_food_from_table_into_fridge",
    "assembling_gift_baskets",
    "sorting_household_items",
    "getting_organized_for_work",
    "clean_up_your_desk",
    # B40
    "setting_the_fire",
    "clean_boxing_gloves",
    "wash_a_baseball_cap",
    "wash_dog_toys",
    "hanging_pictures",
    "attach_a_camera_to_a_tripod",
    "clean_a_patio",
    "clean_a_trumpet",
    "spraying_for_bugs",
    "spraying_fruit_trees",
    # B50
    "make_microwave_popcorn",
    "cook_cabbage",
    "chop_an_onion",
    "slicing_vegetables",
    "chopping_wood",
    "cook_hot_dogs",
    "cook_bacon",
    "freeze_pies",
    "canning_food",
    "make_pizza",
]

# 23-D R1Pro action: matches ACTION_QPOS_INDICES["R1Pro"].
R1PRO_ACTION_DIM = 23

# Sensor key suffixes in OmniGibson's flattened observation dict.
# After ``flatten_obs_dict``, RGB lives at ``{camera_name}{RGB_SUFFIX}``
# and the R1Pro proprioceptive vector at ``PROPRIO_KEY``.
RGB_SUFFIX = "::rgb"
PROPRIO_KEY = "robot_r1::proprio"

# Default camera names from ROBOT_CAMERA_NAMES["R1Pro"].
R1PRO_CAMERAS: dict[str, str] = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
}


def _humanize(task_name: str) -> str:
    """``"turning_on_radio"`` → ``"turning on radio"``."""
    return task_name.replace("_", " ")


class Behavior1KBenchmark(StepBenchmark):
    """BEHAVIOR-1K (OmniGibson) household-activity benchmark.

    Non-obvious behaviors:
        - **Heavy lazy imports**: ``omnigibson`` and Isaac Sim are imported
          inside ``_init_og()`` rather than at module top.  Importing
          OmniGibson boots the Isaac Sim runtime and consumes several
          gigabytes of VRAM, so we delay it until ``get_tasks()`` /
          ``reset()`` actually need it.  This also keeps
          ``vla-eval test --validate`` (a pure import-string check) fast.
        - **Action format**: ``env.step()`` expects a ``torch.Tensor``,
          not numpy.  We convert in ``step()``.
        - **Observation flattening**: OmniGibson's nested observation
          (``obs["robot_r1"]["sensors"]["zed"]["rgb"]``) is flattened with
          a ``::`` delimiter via the official ``flatten_obs_dict`` helper.
          We then look up cameras by their canonical sensor key.
        - **Task description**: BehaviorTask does not expose a natural
          language instruction; we use the snake-case task name with
          underscores replaced by spaces, matching common VLA practice.
        - **Single robot supported**: R1Pro only (the BEHAVIOR Challenge
          2025 standard track).  A1 is reachable through OmniGibson but
          not exercised here.

    Args:
        tasks: Subset of B50 task names to evaluate.  ``None`` runs all 50.
        partial_scene_load: Pass through to OmniGibson — load only rooms
            relevant to the task to speed up scene construction.
        max_steps: Per-episode step cap.  ``None`` keeps OmniGibson's
            default (5000 in ``generate_basic_environment_config``).
        send_proprio: Include the R1Pro proprio vector
            (``robot_r1::proprio``, 256-D) in observations.
        camera_names: Which cameras to forward to the model server.
            Defaults to all three (``head``, ``left_wrist``, ``right_wrist``).
        env_wrapper_target: Hydra ``_target_`` for the env wrapper.  By
            default we use OmniGibson's ``EnvironmentWrapper`` no-op
            wrapper; override to plug in challenge-specific behaviour.
        task_instance_id: Per-instance TRO state to load after ``env.reset()``,
            mirroring the official ``Evaluator.load_task_instance``.
            Without this the env starts from BehaviorTask's default
            instance (idx 0); with it set, the cached
            ``<scene>_task_<activity>_instances/<...>-tro_state.json``
            is applied so the initial object placement matches the
            recorded demos.  Required for demo-replay reproductions.
    """

    def __init__(
        self,
        tasks: list[str] | None = None,
        partial_scene_load: bool = True,
        max_steps: int | None = None,
        send_proprio: bool = False,
        camera_names: list[str] | None = None,
        env_wrapper_target: str = "omnigibson.envs.env_wrapper.EnvironmentWrapper",
        task_instance_id: int | None = None,
    ) -> None:
        super().__init__()
        if tasks is not None:
            unknown = [t for t in tasks if t not in B50_TASKS]
            if unknown:
                raise ValueError(f"Unknown BEHAVIOR-1K tasks: {unknown}")
        self._task_names: list[str] = list(tasks) if tasks else list(B50_TASKS)
        self._partial_scene_load = partial_scene_load
        self._max_steps = max_steps
        self._send_proprio = send_proprio
        self._camera_names = camera_names or list(R1PRO_CAMERAS.keys())
        unknown_cams = [c for c in self._camera_names if c not in R1PRO_CAMERAS]
        if unknown_cams:
            raise ValueError(f"Unknown R1Pro cameras: {unknown_cams}. Valid: {list(R1PRO_CAMERAS)}")
        self._env_wrapper_target = env_wrapper_target
        self._task_instance_id = task_instance_id

        self._env: Any = None
        self._current_task_name: str | None = None
        self._available_tasks: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _init_og(self) -> None:
        """First-time import + side-effect setup for OmniGibson."""
        if self._available_tasks is not None:
            return
        from gello.robots.sim_robot.og_teleop_utils import load_available_tasks
        from omnigibson.macros import gm, macros

        # Match the official challenge eval defaults from learning/eval.py.
        # ``HEADLESS=True`` is critical: without it Isaac Sim tries to start
        # the XR viewport extension and segfaults on a headless GPU node.
        gm.HEADLESS = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = True
        with macros.unlocked():
            macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

        self._available_tasks = load_available_tasks()
        missing = [t for t in self._task_names if t not in self._available_tasks]
        if missing:
            raise RuntimeError(
                f"BEHAVIOR-1K tasks not available in installed dataset: {missing}. "
                "Check that the 2025-challenge-task-instances data is mounted at gm.DATA_PATH."
            )

    def _make_env(self, task_name: str) -> Any:
        """Build a fresh OmniGibson env for *task_name*."""
        # Isaac Sim's SimulationApp.__init__ calls signal.signal(SIGINT, ...)
        # which raises ValueError when invoked from a non-main thread —
        # but we *must* off-load env construction to a worker so the
        # orchestrator's asyncio loop survives.  The handler installed
        # at our main-thread import of omnigibson is already in place,
        # so it's safe to no-op the additional registration here.
        import signal as _signal
        import threading

        _orig_signal = None
        if threading.current_thread() is not threading.main_thread():
            _orig_signal = _signal.signal
            _signal.signal = lambda *a, **kw: None  # type: ignore[assignment]

        try:
            return self._make_env_inner(task_name)
        finally:
            if _orig_signal is not None:
                _signal.signal = _orig_signal

    def _make_env_inner(self, task_name: str) -> Any:
        import omnigibson as og
        from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
        from gello.robots.sim_robot.og_teleop_utils import (
            augment_rooms,
            generate_robot_config,
            get_task_relevant_room_types,
        )
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from omnigibson.learning.utils.eval_utils import (
            PROPRIOCEPTION_INDICES,
            generate_basic_environment_config,
        )

        # The official eval disables a curated set of transition rules to
        # match the data-collection setup.
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        assert self._available_tasks is not None
        task_cfg = self._available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)

        if self._partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

        if self._max_steps is not None:
            cfg["task"]["termination_config"]["max_steps"] = self._max_steps
        cfg["task"]["include_obs"] = False

        env = og.Environment(configs=cfg)
        wrapper_cfg = OmegaConf.create({"_target_": self._env_wrapper_target})
        env = instantiate(wrapper_cfg, env=env)
        return env

    # ------------------------------------------------------------------
    # Benchmark ABC
    # ------------------------------------------------------------------

    def get_tasks(self) -> list[Task]:
        # Avoid booting Isaac Sim during config validation: defer the
        # import-side-effect until we actually have a chance to run.
        return [{"name": _humanize(t), "task_name": t, "suite": "behavior_1k"} for t in self._task_names]

    def reset(self, task: Task) -> Any:
        self._init_og()
        task_name = task["task_name"]
        if self._env is None or self._current_task_name != task_name:
            if self._env is not None:
                try:
                    self._env.close()
                except Exception:
                    logger.exception("Failed to close previous OmniGibson env")
            self._env = self._make_env(task_name)
            self._current_task_name = task_name
        obs, _ = self._env.reset()
        # Optional per-instance TRO state load (matches official
        # ``Evaluator.load_task_instance``).  When unset, BehaviorTask
        # uses its default instance (idx 0) — the env still runs, but
        # object placements may diverge from a particular demo.
        if self._task_instance_id is not None:
            obs = self._load_task_instance(self._task_instance_id)
        return obs

    def _load_task_instance(self, instance_id: int) -> Any:
        """Apply per-instance object/robot state JSON, then re-fetch obs.

        Ports the v3.7.2 ``Evaluator.load_task_instance`` (public-test
        branch).  Reads
        ``<get_task_instance_path(scene)>/json/<scene>_task_<activity>_instances/<...>-tro_state.json``
        and pushes the recorded object/robot state into the running env.

        Compatible only with the v3.7.2 OmniGibson API: uses
        ``robot.model_name``, ``entity.is_system`` / ``entity.exists``.
        """
        import json
        import os

        import omnigibson as og
        from omnigibson.utils.asset_utils import get_task_instance_path
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        env = self._env
        task = env.task
        scene_model = task.scene_name
        tro_filename = task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=task.activity_name,
            activity_definition_id=task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))

        robot = env.scene.object_registry("name", "robot_r1")
        for tro_key, tro_substate in tro_state.items():
            if tro_key == "robot_poses":
                if robot is None:
                    raise RuntimeError("BEHAVIOR-1K _load_task_instance: robot 'robot_r1' not found in scene")
                model_name = getattr(robot, "model_name", None) or getattr(robot, "model", None)
                if model_name not in tro_substate:
                    raise KeyError(
                        f"BEHAVIOR-1K instance {instance_id}: no presampled robot pose "
                        f"for robot.model_name={model_name!r}; keys={list(tro_substate)}"
                    )
                pose0 = tro_substate[model_name][0]
                robot.set_position_orientation(pose0["position"], pose0["orientation"])
                env.scene.write_task_metadata(key=tro_key, data=tro_substate)
            else:
                task.object_scope[tro_key].load_state(tro_substate, serialized=False)

        # Settle objects so loaded poses are stable before evaluation.
        for _ in range(25):
            og.sim.step_physics()
            for entity in task.object_scope.values():
                if entity is not None and not getattr(entity, "is_system", False) and getattr(entity, "exists", True):
                    entity.keep_still()

        env.scene.update_initial_file()
        env.scene.reset()

        # Re-fetch the observation after the state load so the model
        # server sees the post-load images / proprio.
        obs, _ = env.get_obs()
        return obs

    def step(self, action: Action) -> StepResult:
        import torch as th

        raw = action.get("actions", action.get("action"))
        tensor = th.as_tensor(raw, dtype=th.float32).flatten()
        if tensor.shape[0] != R1PRO_ACTION_DIM:
            raise ValueError(f"BEHAVIOR-1K expects a {R1PRO_ACTION_DIM}-D R1Pro joint action, got {tensor.shape[0]}D.")

        assert self._env is not None
        obs, reward, terminated, truncated, info = self._env.step(tensor, n_render_iterations=1)
        info = dict(info)
        info["truncated"] = bool(truncated)
        done = bool(terminated) or bool(truncated)
        return StepResult(obs=obs, reward=float(reward), done=done, info=info)

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        from omnigibson.learning.utils.eval_utils import flatten_obs_dict

        flat = flatten_obs_dict(raw_obs)

        images: dict[str, np.ndarray] = {}
        for cam in self._camera_names:
            key = R1PRO_CAMERAS[cam] + RGB_SUFFIX
            if key not in flat:
                continue
            value = flat[key]
            if hasattr(value, "cpu"):  # torch.Tensor
                value = value.cpu().numpy()
            arr = np.asarray(value, dtype=np.uint8)
            # OmniGibson VisionSensor returns (H, W, 4) RGBA — drop alpha.
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]
            images[cam] = np.ascontiguousarray(arr)

        out: Observation = {
            "images": images,
            "task_description": task["name"],
        }

        if self._send_proprio:
            proprio = flat.get(PROPRIO_KEY)
            if proprio is not None:
                if hasattr(proprio, "cpu"):
                    proprio = proprio.cpu().numpy()
                out["states"] = np.asarray(proprio, dtype=np.float32)

        return out

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        done_info = step_result.info.get("done", {}) or {}
        success = bool(done_info.get("success", False))
        return {"success": success}

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": self._max_steps if self._max_steps is not None else 5000,
            "robot": "R1Pro",
            "n_tasks": len(self._task_names),
        }

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                logger.exception("BEHAVIOR-1K env close failed")
            self._env = None
        # Intentionally NOT calling ``omnigibson.shutdown()`` here:
        # Isaac Sim's shutdown path can hang for many minutes
        # (waiting on hydra texture cleanup, render contexts, etc.)
        # which prevents the orchestrator from writing the result JSON
        # at the end of the run.  Process exit reclaims everything;
        # leaving Isaac Sim alone is the lesser evil.

    # ------------------------------------------------------------------
    # Async bridge override: run sync reset()/step() on a worker thread.
    # Booting Isaac Sim from the orchestrator's main thread tears down
    # the running asyncio event loop (SimulationApp installs its own),
    # which makes the next `await conn.act(...)` raise NoEventLoopError.
    # Off-loading to anyio.to_thread.run_sync keeps the orchestrator
    # loop intact while Isaac Sim does its synchronous work.
    # ------------------------------------------------------------------

    async def start_episode(self, task: Task) -> None:
        self._t0 = time.monotonic()
        self._task = task
        # Run imports + signal-handler registration on the main thread
        # (Python's signal module forbids setting handlers from a worker
        # thread, and OmniGibson registers SIGINT during its top-level
        # ``__init__.py``).  Only the env construction / reset itself is
        # offloaded to the worker thread, which is what actually trashes
        # the asyncio event loop.
        self._init_og()
        raw_obs = await anyio.to_thread.run_sync(self.reset, task)  # type: ignore[attr-defined]
        self._last_result = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        self._last_result = await anyio.to_thread.run_sync(self.step, action)  # type: ignore[attr-defined]

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "joints": DimSpec("joints", R1PRO_ACTION_DIM, "joint_positions_r1pro"),
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"language": LANGUAGE}
        for cam in self._camera_names:
            spec[cam] = IMAGE_RGB
        if self._send_proprio:
            spec["state"] = RAW
        return spec
