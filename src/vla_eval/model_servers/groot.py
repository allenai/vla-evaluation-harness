# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "gr00t",
#     "torch>=2.2,<2.8",
#     "numpy>=1.24",
#     "pillow>=9.0",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# gr00t = { git = "https://github.com/NVIDIA/Isaac-GR00T.git", rev = "e29d8fc50b0e4745120ae3fb72447986fe638aa6" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# no-build-isolation-package = ["flash-attn"]
# ///
"""GR00T N1.6 model server.

Uses NVIDIA Isaac-GR00T ``Gr00tPolicy`` for inference with the
nvidia/GR00T-N1.6-3B foundation model (or fine-tuned checkpoints).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from vla_eval.specs import (
    GRIPPER_01,
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    ROTATION_EULER,
    DimSpec,
)
from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

logger = logging.getLogger(__name__)


class GR00TModelServer(PredictModelServer):
    """GR00T N1.6 model server using Isaac-GR00T Gr00tPolicy."""

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.6-3B",
        embodiment_tag: str = "GR1",
        video_key: str | None = None,
        action_keys: list[str] | None = None,
        invert_gripper: bool = False,
        image_resolution: int | None = None,
        bridge_rotation: bool = False,
        observation_params: str | dict | None = None,
        *,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.embodiment_tag = embodiment_tag
        tag_name = str(embodiment_tag).rsplit(".", 1)[-1].upper()
        self._is_simpler = tag_name in self._SIMPLER_TAGS
        self._is_libero = tag_name in self._LIBERO_TAGS
        self.video_key = video_key  # None = auto-detect from modality config
        self.action_keys = action_keys
        self.invert_gripper = invert_gripper
        self.image_resolution = image_resolution
        self.bridge_rotation = bridge_rotation
        self._extra_obs_params: dict[str, Any] = {}
        if observation_params:
            import json

            self._extra_obs_params = (
                json.loads(observation_params) if isinstance(observation_params, str) else observation_params
            )
        self._state_dims: dict[str, int] = {}

        self._init_policy()

    # Embodiment families that need benchmark-specific observation handling.
    # SimplerEnv / ManiSkill2 report eef_pos as ``[xyz, quat_wxyz, gripper]``, which is
    # not the ``statistics.json`` state layout and has to be rewritten in
    # ``predict_batch``.  Every other embodiment (LIBERO, GR1, RoboCasa) already sends
    # state in its own layout and must pass through untouched.
    _SIMPLER_TAGS = frozenset({"OXE_GOOGLE", "OXE_WIDOWX"})
    _LIBERO_TAGS = frozenset({"LIBERO_PANDA"})

    # Data files that Isaac-GR00T's pip package omits from Eagle backbone.
    _EAGLE_DATA_FILES = [
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ]

    @classmethod
    def _ensure_eagle_data(cls) -> None:
        """Download missing Eagle backbone data files.

        Isaac-GR00T's pip package only ships ``.py`` files; the JSON/tokenizer
        data required by ``AutoProcessor`` / ``AutoConfig`` must be fetched
        from the GitHub repo on first use.
        """
        import gr00t.model.modules as _mod

        eagle_dir = os.path.join(
            os.path.dirname(_mod.__file__),
            "nvidia",
            "Eagle-Block2A-2B-v2",
        )
        missing = [f for f in cls._EAGLE_DATA_FILES if not os.path.isfile(os.path.join(eagle_dir, f))]
        if not missing:
            return
        import urllib.request

        base_url = (
            "https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/"
            "e29d8fc50b0e4745120ae3fb72447986fe638aa6/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/"
        )
        os.makedirs(eagle_dir, exist_ok=True)
        for fname in missing:
            url = base_url + fname
            dst = os.path.join(eagle_dir, fname)
            logger.info("Downloading missing Eagle data file: %s", fname)
            urllib.request.urlretrieve(url, dst)

    def _init_policy(self) -> None:
        from vla_eval.dirs import require_model_available

        require_model_available(self.model_path)

        import json

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from huggingface_hub import hf_hub_download

        self._ensure_eagle_data()

        tag = getattr(EmbodimentTag, self.embodiment_tag, self.embodiment_tag)
        logger.info("Loading GR00T from %s (embodiment=%s)", self.model_path, tag)

        self._policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag=tag,
            device="cuda:0",
            strict=False,
        )
        self._modality_config = self._policy.get_modality_config()
        self._language_key = self._policy.language_key

        # Load state dimensions from statistics.json
        tag_value = tag.value if hasattr(tag, "value") else str(tag)
        stats_path = hf_hub_download(self.model_path, "statistics.json")
        with open(stats_path) as f:
            all_stats = json.load(f)
        state_stats = all_stats.get(tag_value, {}).get("state", {})
        self._state_dims = {k: len(v["mean"]) for k, v in state_stats.items()}

        logger.info(
            "GR00T model loaded. video_keys=%s, state_keys=%s (dims=%s), action_keys=%s",
            self._modality_config["video"].modality_keys,
            self._modality_config["state"].modality_keys,
            self._state_dims,
            self._modality_config["action"].modality_keys,
        )

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "send_state": True,
            "success_mode": "accumulate",
            "deterministic_episodes": False,
        }
        if self._is_libero:
            # libero_panda was trained on IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot:
            # two 256x256 views plus an 8-D proprio state
            # ``[xyz3, axis_angle3, gripper_qpos2]``, where the axis-angle comes from
            # robosuite's ``quat2axisangle`` (no antipodal normalization).  Dropping the
            # wrist view, or switching to an antipodal-normalized axis-angle (its sign
            # flips frame to frame once the gripper points down and ``w ~ 0``, so roll
            # jumps between +-pi), silently shifts the input distribution.
            params["send_wrist_image"] = True
            params["quat_no_antipodal"] = True
        else:
            # LIBEROBenchmark does not accept this parameter (it truncates with a
            # per-suite ``max_steps`` of 220..520), so it only reaches benchmarks that
            # cut episodes by step count (SimplerEnv, ManiSkill2, ...).
            params["max_episode_steps"] = 300
        params.update(self._extra_obs_params)
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:
        gripper = GRIPPER_CLOSE_POS if self.invert_gripper else GRIPPER_01
        # libero_panda names its rotation keys roll/pitch/yaw, but the values are
        # axis-angle: the training set calls the same features axis_angle1..3.
        rotation = ROTATION_AA if self._is_libero else ROTATION_EULER
        return {"position": POSITION_DELTA, "rotation": rotation, "gripper": gripper}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    _BRIDGE_DEFAULT_ROT = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        B = len(obs_batch)

        if self.image_resolution:
            import cv2

        from vla_eval.rotation import quat_wxyz_to_xyzw

        if self.bridge_rotation:
            from vla_eval.rotation import matrix_to_euler_xyz, quat_to_matrix

        video_keys = self._modality_config["video"].modality_keys
        if self.video_key is not None:
            video_keys = [self.video_key]

        # Collect per-key image lists and language across batch
        per_key_imgs: dict[str, list[np.ndarray]] = {k: [] for k in video_keys}
        langs = []
        for obs in obs_batch:
            images_dict = obs.get("images", {})
            img_values = list(images_dict.values()) if isinstance(images_dict, dict) else []
            if len(img_values) < len(video_keys):
                # Missing views used to be padded with a black image: the model still
                # returns actions and the run still reports a score, which is how GR00T's
                # wrist_image stayed black for a whole LIBERO sweep.  Fail loudly instead.
                raise RuntimeError(
                    f"embodiment {self.embodiment_tag} needs {len(video_keys)} views {video_keys}, "
                    f"but the benchmark sent {len(img_values)}: "
                    f"{list(images_dict) if isinstance(images_dict, dict) else images_dict}. "
                    "Declare send_wrist_image=True in get_observation_params(), "
                    "or narrow the model to a single view with video_key."
                )
            for idx, vk in enumerate(video_keys):
                img = np.asarray(img_values[idx], dtype=np.uint8)
                if self.image_resolution and img.shape[:2] != (self.image_resolution, self.image_resolution):
                    img = cv2.resize(img, (self.image_resolution, self.image_resolution))
                if img.ndim == 3:
                    img = img[np.newaxis, ...]  # (T=1, H, W, C)
                per_key_imgs[vk].append(img)
            langs.append([obs.get("task_description", "")])

        # Stack into batched observation: video {key: (B, T=1, H, W, C)}, language (B, 1)
        observation: dict[str, Any] = {
            "video": {k: np.stack(v, axis=0) for k, v in per_key_imgs.items()},
            "state": {},
            "language": {self._language_key: langs},
        }

        # Initialize state arrays, then fill from observation if available
        state_keys = self._modality_config["state"].modality_keys
        for sk in state_keys:
            dim = self._state_dims.get(sk, 1)
            observation["state"][sk] = np.zeros((B, 1, dim), dtype=np.float32)
        expected_state_dim = sum(self._state_dims.get(sk, 1) for sk in state_keys)

        # Decompose flat state vector into per-key arrays
        for obs_idx, obs in enumerate(obs_batch):
            raw_state = obs.get("states", obs.get("state"))
            if raw_state is None:
                continue
            state_arr = np.asarray(raw_state, dtype=np.float32).flatten()

            # SimplerEnv / ManiSkill2 only: eef_pos is
            # [x, y, z, qw, qx, qy, qz, gripper_openness], which is not the
            # statistics.json state layout and has to be rewritten.  Everything else
            # passes through -- LIBERO sends [xyz3, axis_angle3, gripper_qpos2], which
            # already lines up dimension by dimension with libero_panda's state keys
            # (x, y, z, roll, pitch, yaw, gripper; the last one is 2-D).  The guard used
            # to be `len(state_arr) >= 8`, which sliced LIBERO's axis-angle as if it were
            # a quaternion: roll/pitch/yaw got qx, qy, qz and gripper got [qw, 1 - g].
            if self._is_simpler:
                if len(state_arr) < 8:
                    raise RuntimeError(
                        f"embodiment {self.embodiment_tag} needs an 8-D eef_pos "
                        f"[xyz3, quat_wxyz4, gripper], got {len(state_arr)} dims"
                    )
                if self.bridge_rotation:
                    # WidowX: convert quaternion to bridge-frame euler angles
                    quat_xyzw = quat_wxyz_to_xyzw(state_arr[3:7])
                    rm = quat_to_matrix(quat_xyzw)
                    rpy = matrix_to_euler_xyz(rm @ self._BRIDGE_DEFAULT_ROT.T)
                    state_arr = np.array([*state_arr[:3], *rpy, 0.0, state_arr[7]], dtype=np.float32)
                else:
                    # Google Robot: reorder quaternion wxyz→xyzw, invert gripper
                    quat_xyzw = quat_wxyz_to_xyzw(state_arr[3:7])
                    gripper_closedness = 1.0 - state_arr[7]
                    state_arr = np.array([*state_arr[:3], *quat_xyzw, gripper_closedness], dtype=np.float32)

            # A mismatch silently zero-pads or shifts every key past the short one.
            # LIBERO is verified dimension by dimension, so a mismatch there means the
            # benchmark contract changed: fail.  Other embodiments are not individually
            # verified yet, so warn.
            if len(state_arr) != expected_state_dim:
                msg = (
                    f"embodiment {self.embodiment_tag} state dim mismatch: statistics.json expects "
                    f"{expected_state_dim} dims {list(state_keys)}, benchmark sent {len(state_arr)}"
                )
                if self._is_libero:
                    raise RuntimeError(msg)
                logger.warning("%s", msg)

            offset = 0
            for sk in state_keys:
                dim = self._state_dims.get(sk, 1)
                if offset + dim <= len(state_arr):
                    observation["state"][sk][obs_idx, 0, :] = state_arr[offset : offset + dim]
                offset += dim

        action_dict, _ = self._policy.get_action(observation)

        keys = self.action_keys or self._modality_config["action"].modality_keys
        outputs = []
        for i in range(B):
            parts = [action_dict[k][i] for k in keys if k in action_dict]
            actions = np.concatenate(parts, axis=-1) if parts else np.zeros((1, 7), dtype=np.float32)
            if self.invert_gripper:
                # Model outputs gripper in [0,1] (0=close, 1=open).
                # LIBERO expects [-1,1] (-1=open, +1=close).
                # Transform: normalize [0,1]→[-1,1] then invert sign.
                actions[..., -1] = 1.0 - 2.0 * actions[..., -1]
            outputs.append({"actions": actions})
        return outputs

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._policy.reset()
        await super().on_episode_start(config, ctx)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(GR00TModelServer)
