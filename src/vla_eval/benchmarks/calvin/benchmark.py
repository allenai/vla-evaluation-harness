"""CALVIN benchmark implementation.

Evaluates 1000 sequences of 5 chained subtasks each (ABC→D split).
Each sequence is treated as one episode from the orchestrator's perspective.
"""

from __future__ import annotations

import logging
import math
import os
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.rotation import axisangle_to_matrix, matrix_to_euler_xyz
from vla_eval.specs import (
    GRIPPER_CLOSE_NEG,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_ABSOLUTE,
    POSITION_DELTA,
    ROTATION_EULER,
    ROTATION_EULER_ACCEPTS_AA,
    STATE_EEF_POS_EULER_GRIP,
    DimSpec,
)
from vla_eval.types import Action, EpisodeResult, Observation, Task

logger = logging.getLogger(__name__)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "device")

EP_LEN = 360
NUM_SUBTASKS = 5


# Language annotations for each task (from new_playtable_validation.yaml)
LANG_ANNOTATIONS: dict[str, str] = {
    "rotate_red_block_right": "take the red block and rotate it to the right",
    "rotate_red_block_left": "take the red block and rotate it to the left",
    "rotate_blue_block_right": "take the blue block and rotate it to the right",
    "rotate_blue_block_left": "take the blue block and rotate it to the left",
    "rotate_pink_block_right": "take the pink block and rotate it to the right",
    "rotate_pink_block_left": "take the pink block and rotate it to the left",
    "push_red_block_right": "go push the red block right",
    "push_red_block_left": "go push the red block left",
    "push_blue_block_right": "go push the blue block right",
    "push_blue_block_left": "go push the blue block left",
    "push_pink_block_right": "go push the pink block right",
    "push_pink_block_left": "go push the pink block left",
    "move_slider_left": "push the sliding door to the left side",
    "move_slider_right": "push the sliding door to the right side",
    "open_drawer": "pull the handle to open the drawer",
    "close_drawer": "push the handle to close the drawer",
    "lift_red_block_table": "grasp and lift the red block",
    "lift_blue_block_table": "grasp and lift the blue block",
    "lift_pink_block_table": "grasp and lift the pink block",
    "lift_red_block_slider": "lift the red block from the sliding cabinet",
    "lift_blue_block_slider": "lift the blue block from the sliding cabinet",
    "lift_pink_block_slider": "lift the pink block from the sliding cabinet",
    "lift_red_block_drawer": "Take the red block from the drawer",
    "lift_blue_block_drawer": "Take the blue block from the drawer",
    "lift_pink_block_drawer": "Take the pink block from the drawer",
    "place_in_slider": "store the grasped block in the sliding cabinet",
    "place_in_drawer": "store the grasped block in the drawer",
    "push_into_drawer": "slide the block that it falls into the drawer",
    "stack_block": "stack the grasped block",
    "unstack_block": "remove the stacked block",
    "turn_on_lightbulb": "use the switch to turn on the light bulb",
    "turn_off_lightbulb": "use the switch to turn off the light bulb",
    "turn_on_led": "press the button to turn on the led light",
    "turn_off_led": "press the button to turn off the led light",
}


def _get_env_state_for_initial_condition(initial_condition: dict) -> tuple[np.ndarray, np.ndarray]:
    """Compute robot_obs and scene_obs from an initial condition dict.

    Exact copy of calvin_agent.evaluation.utils.get_env_state_for_initial_condition.
    """
    import fnvhash

    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (math.pi / 2 - math.pi / 8, math.pi / 2 + math.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = fnvhash.fnv1_32(str(initial_condition.values()).encode())
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        # red block
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        # blue block
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        # pink block
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    finally:
        np.random.set_state(state)
    return robot_obs, scene_obs


class CALVINBenchmark(StepBenchmark):
    """CALVIN ABC→D long-horizon benchmark (PyBullet).

    Each "episode" is a sequence of 5 chained subtasks.  The task oracle
    detects subtask completion; on success the benchmark advances to the
    next subtask and includes ``"episode_restart": True`` in the next
    observation dict so the model server can reset internal state.

    Non-obvious behaviors:
        - **Delta actions**: Model outputs are deltas added to the previous
          action.  Gripper is binarized *after* delta addition.
        - **Hardcoded normalization**: Robot state (15 values) and scene state
          (24 values) use training-set statistics embedded in the code.
        - **Max steps per subtask**: 360 steps.  Total max = 360 × 5 = 1800.
        - **Observation space**: RGB (200×200), robot state, scene state.
          Not configurable — matches the training setup.

    Args:
        dataset_path: Path to CALVIN validation dataset.
        num_sequences: Number of 5-subtask sequences to evaluate (default 1000).
        seed: Random seed for sequence generation and PyTorch Lightning.
        send_wrist_image: Include gripper camera image in observations.
        send_state: Include 8-D proprioceptive state
            ``[pos3, euler3, gripper2]`` in observations.
        absolute_action: Use absolute 7-D ``[pos3, euler3, gripper]``
            actions instead of delta accumulation.
        ep_len: Override per-subtask step limit (default 360).
    """

    def __init__(
        self,
        dataset_path: str = "/data/calvin/dataset/validation",
        num_sequences: int = 1000,
        seed: int = 0,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        ep_len: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.num_sequences = num_sequences
        self.seed = seed
        self.send_wrist_image = send_wrist_image
        self.send_state = send_state
        self.absolute_action = absolute_action
        self._ep_len = ep_len
        self._env = None
        self._task_oracle = None
        self._sequences: list | None = None
        # Per-episode state
        self._eval_sequence: list[str] = []
        self._subtask_idx: int = 0
        self._subtask_step: int = 0
        self._completed: int = 0
        self._last_act: np.ndarray | None = None
        self._start_info: dict | None = None
        self._subtask_just_reset: bool = False
        self._device = None

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def _init_calvin(self) -> None:
        """Lazily initialize CALVIN environment and task oracle."""
        if self._env is not None:
            return

        import pytorch_lightning as pl
        import torch
        from omegaconf import OmegaConf
        import hydra

        pl.seed_everything(self.seed, workers=True)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Locate CALVIN conf directory
        import calvin_agent

        conf_dir = Path(calvin_agent.__file__).parent.parent / "conf"

        # Load task oracle
        tasks_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
        self._task_oracle = hydra.utils.instantiate(tasks_cfg)

        # Load validation annotations
        val_ann = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
        self._val_annotations = val_ann

        # Build observation_space, proprio_state, transforms manually
        # (hydra.compose would fail on unresolvable interpolations)
        rgb_obs_list = ["rgb_static"]
        if self.send_wrist_image:
            rgb_obs_list.append("rgb_gripper")
        observation_space = OmegaConf.create(
            {
                "rgb_obs": rgb_obs_list,
                "depth_obs": [],
                "state_obs": ["robot_obs"],
                "actions": ["actions"],
                "language": ["language"],
            }
        )
        proprio_state = OmegaConf.create(
            {
                "n_state_obs": 8,
                "keep_indices": [[0, 7], [14, 15]],
                "robot_orientation_idx": [3, 6],
                "normalize": True,
                "normalize_robot_orientation": True,
            }
        )

        # Val transforms from play_basic.yaml + statistics.yaml
        val_transforms_cfg = OmegaConf.create(
            {
                "rgb_static": [
                    {"_target_": "torchvision.transforms.Resize", "size": 200},
                    {"_target_": "calvin_agent.utils.transforms.ScaleImageTensor"},
                    {"_target_": "torchvision.transforms.Normalize", "mean": [0.5], "std": [0.5]},
                ],
                "robot_obs": [
                    {
                        "_target_": "calvin_agent.utils.transforms.NormalizeVector",
                        "mean": [
                            0.039233,
                            -0.118554,
                            0.507826,
                            1.079174,
                            -0.083069,
                            1.579753,
                            0.054622,
                            -0.736859,
                            1.017769,
                            1.792879,
                            -2.099604,
                            -0.993738,
                            1.790842,
                            0.586534,
                            0.095367,
                        ],
                        "std": [
                            0.150769,
                            0.1104,
                            0.06253,
                            2.883517,
                            0.126405,
                            0.377196,
                            0.030152,
                            0.334392,
                            0.172714,
                            0.240513,
                            0.3842,
                            0.198596,
                            0.158712,
                            0.346865,
                            0.995442,
                        ],
                    },
                ],
                "scene_obs": [
                    {
                        "_target_": "calvin_agent.utils.transforms.NormalizeVector",
                        "mean": [
                            0.150934,
                            0.119917,
                            0.000239,
                            0.042049,
                            0.487755,
                            0.47448,
                            0.057482,
                            -0.088074,
                            0.431237,
                            0.046034,
                            0.030599,
                            0.027333,
                            0.062103,
                            -0.092833,
                            0.430236,
                            -0.054962,
                            0.019381,
                            0.096546,
                            0.064944,
                            -0.093058,
                            0.428381,
                            0.024941,
                            0.002746,
                            -0.031589,
                        ],
                        "std": [
                            0.125757,
                            0.09654,
                            0.002148,
                            0.041916,
                            0.49985,
                            0.499348,
                            0.146225,
                            0.119266,
                            0.050408,
                            1.430807,
                            0.676023,
                            2.017468,
                            0.142979,
                            0.113236,
                            0.049651,
                            1.545888,
                            0.3906,
                            1.763569,
                            0.143077,
                            0.11546,
                            0.050363,
                            1.514873,
                            0.431664,
                            1.860245,
                        ],
                    },
                ],
            }
        )

        if self.send_wrist_image:
            val_transforms_cfg["rgb_gripper"] = [
                {"_target_": "torchvision.transforms.Resize", "size": 200},
                {"_target_": "calvin_agent.utils.transforms.ScaleImageTensor"},
                {"_target_": "torchvision.transforms.Normalize", "mean": [0.5], "std": [0.5]},
            ]

        # Instantiate transforms into Compose objects
        from torchvision import transforms as T

        transforms = {}
        for key, tf_list in val_transforms_cfg.items():
            composed = [hydra.utils.instantiate(tf) for tf in tf_list]
            transforms[key] = T.Compose(composed)

        # Build minimal val_dataset duck-type for CalvinEnvWrapper
        class _ValDS:
            abs_datasets_dir: Any = None
            observation_space: Any = None
            transforms: Any = None
            proprio_state: Any = None

        val_ds = _ValDS()
        val_ds.abs_datasets_dir = Path(self.dataset_path)
        val_ds.observation_space = observation_space
        val_ds.transforms = transforms
        val_ds.proprio_state = proprio_state

        # Instantiate environment
        rollout_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/default.yaml")
        self._env = hydra.utils.instantiate(rollout_cfg.env_cfg, val_ds, self._device, show_gui=False)

        # Generate evaluation sequences
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        self._sequences = get_sequences(self.num_sequences)
        logger.info("CALVIN initialized: %d sequences on %s", len(self._sequences), self._device)

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "completed_subtasks": "mean"}

    def get_tasks(self) -> list[Task]:
        self._init_calvin()
        tasks = []
        assert self._sequences is not None
        for idx, (initial_condition, eval_sequence) in enumerate(self._sequences):
            tasks.append(
                {
                    "name": f"seq_{idx:04d}",
                    "initial_condition": initial_condition,
                    "eval_sequence": eval_sequence,
                    "seq_idx": idx,
                }
            )
        return tasks

    def reset(self, task: Task) -> Any:
        self._init_calvin()
        initial_condition = task["initial_condition"]
        self._eval_sequence = list(task["eval_sequence"])
        self._subtask_idx = 0
        self._subtask_step = 0
        self._completed = 0
        self._subtask_just_reset = False

        # Reset env to initial condition
        robot_obs, scene_obs = _get_env_state_for_initial_condition(initial_condition)
        assert self._env is not None
        self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        obs = self._env.get_obs()

        if not self.absolute_action:
            # Initialize last_act from raw robot observation (delta mode only)
            raw = obs["robot_obs_raw"].cpu().numpy()
            self._last_act = np.concatenate([raw[0:6], raw[14:15]])

        # Capture start_info for task oracle
        self._start_info = self._env.get_info()

        return obs

    def step(self, action: Action) -> StepResult:
        import torch

        ep_len = self._ep_len or EP_LEN

        if self.absolute_action:
            act = self._process_absolute_action(action)
        else:
            act = self._process_delta_action(action)

        # Step environment (expects shape [1, 1, 7] torch tensor)
        assert self._env is not None
        action_tensor = torch.tensor(act, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self._device)
        obs, _, _, info = self._env.step(action_tensor)
        self._subtask_step += 1

        # Check task oracle for current subtask
        assert self._task_oracle is not None
        current_subtask = self._eval_sequence[self._subtask_idx]
        current_info = self._env.get_info()
        if self._task_oracle.get_task_info_for_set(self._start_info, current_info, {current_subtask}):
            # Subtask succeeded
            self._completed += 1
            if self._completed >= NUM_SUBTASKS:
                # All 5 subtasks done — sequence success
                return StepResult(obs=obs, reward=1.0, done=True, info={"success": True})

            # Move to next subtask
            self._subtask_idx += 1
            self._subtask_step = 0
            self._start_info = current_info

            if not self.absolute_action:
                # Re-init last_act from current robot state (delta mode only)
                raw = obs["robot_obs_raw"].cpu().numpy()
                self._last_act = np.concatenate([raw[0:6], raw[14:15]])

            self._subtask_just_reset = True
            return StepResult(
                obs=obs,
                reward=0.0,
                done=False,
                info={"completed": self._completed},
            )

        # Check if subtask timed out
        if self._subtask_step >= ep_len:
            return StepResult(
                obs=obs,
                reward=0.0,
                done=True,
                info={"success": False, "completed": self._completed},
            )

        return StepResult(obs=obs, reward=0.0, done=False, info={})

    def _process_absolute_action(self, action: Action) -> np.ndarray:
        """Process 7D absolute action [pos3, axisangle3, gripper] → [pos3, euler3, gripper].

        Model servers return axis-angle rotation; CALVIN env expects euler XYZ.
        """
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, (list, np.ndarray)):
            raw = np.asarray(raw_action, dtype=np.float64).flatten()[:7]
        else:
            raw = np.zeros(7, dtype=np.float64)

        pos = raw[:3]
        euler = matrix_to_euler_xyz(axisangle_to_matrix(raw[3:6].astype(np.float32)))
        gripper = -1.0 if raw[6] > 0 else 1.0

        return np.concatenate([pos, euler, [gripper]])

    def _process_delta_action(self, action: Action) -> np.ndarray:
        """Process 7D delta action (original CogACT mode)."""
        raw_action = action.get("actions", action.get("action"))
        if isinstance(raw_action, (list, np.ndarray)):
            raw_action = np.asarray(raw_action, dtype=np.float64).flatten()[:7]
        else:
            raw_action = np.zeros(7, dtype=np.float64)

        assert self._last_act is not None
        original = copy(self._last_act)
        original[6:] = 0.0  # zero out gripper in base
        act = original + raw_action

        # Binarize gripper
        act[6] = 1.0 if act[6] > 0 else -1.0

        # Normalize rotation angles to [-pi, pi]
        act[3:6] = (act[3:6] + np.pi) % (2 * np.pi) - np.pi

        self._last_act = act.copy()
        return act

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # Convert rgb tensor [C, H, W] float [-1, 1] → uint8 [H, W, C]
        rgb_tensor = raw_obs["rgb_obs"]["rgb_static"][0, 0]  # [C, H, W]
        img = ((rgb_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

        # Get language annotation for current subtask
        subtask_name = self._eval_sequence[self._subtask_idx]
        lang = LANG_ANNOTATIONS.get(subtask_name, subtask_name)

        obs: dict[str, Any] = {
            "images": {"rgb_static": img},
            "task_description": lang,
        }

        if self.send_wrist_image:
            wrist_tensor = raw_obs["rgb_obs"]["rgb_gripper"][0, 0]  # [C, H, W]
            wrist_img = ((wrist_tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            obs["images"]["rgb_gripper"] = wrist_img

        if self.send_state:
            # 8D: [pos3, euler3, gripper2] — CALVIN proprio_state subset
            raw = raw_obs["robot_obs_raw"].cpu().numpy()
            obs["states"] = np.concatenate([raw[0:7], raw[14:15]]).astype(np.float32)

        # Signal model server to reset internal state on subtask transition
        if self._subtask_just_reset:
            obs["episode_restart"] = True
            self._subtask_just_reset = False

        return obs

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {
            "success": step_result.info.get("success", False),
            "completed_subtasks": step_result.info.get("completed", self._completed),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": (self._ep_len or EP_LEN) * NUM_SUBTASKS}

    def get_action_spec(self) -> dict[str, DimSpec]:
        if self.absolute_action:
            return {
                "position": POSITION_ABSOLUTE,
                "rotation": ROTATION_EULER_ACCEPTS_AA,
                "gripper": GRIPPER_CLOSE_NEG,
            }
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {
            "rgb_static": IMAGE_RGB,
            "language": LANGUAGE,
        }
        if self.send_wrist_image:
            spec["rgb_gripper"] = IMAGE_RGB
        if self.send_state:
            spec["state"] = STATE_EEF_POS_EULER_GRIP
        return spec
