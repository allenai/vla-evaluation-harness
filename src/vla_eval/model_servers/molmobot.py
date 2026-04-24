# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "molmobot @ git+https://github.com/allenai/MolmoBot.git#subdirectory=MolmoBot",
#     "molmo-spaces @ git+https://github.com/allenai/molmospaces.git",
#     "torch>=2.3.1",
#     "transformers>=4.37.1",
#     "huggingface_hub",
#     "numpy",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# ///
"""MolmoBot model server (Molmo2-4B + DiT flow-matching).

Mirrors the behavior of olmo.eval.configure_molmo_spaces.SynthVLAPolicy
so scores match the official evaluation pipeline.

Key details (from FrankaState8ClampAbsPosConfig + SynthVLAPolicyConfig):
- action_type: joint_pos (absolute joint positions, NOT relative)
- action_horizon: 16, execute_horizon: 8
- clamp_gripper: True (values > 128 → 255, else → 0)
- states_mode: cross_attn
- relative_max_joint_delta: [0.2] * 7 (per-step safety clamp)
- Cameras: exo_camera_1 + wrist_camera (maps to droid_shoulder_light_randomization
  and wrist_camera_zed_mini in MolmoSpaces env)
- State: 8D (7 arm joint qpos + 1 gripper qpos)
- Frame history: n_obs_steps from model config (MolmoBot-DROID: 2 frames,
  MolmoBot-Img: 1 frame), sampled at obs_step_delta intervals.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_eval.model_servers.base import ModelServer, SessionContext
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, STATE_JOINT, DimSpec
from vla_eval.types import Observation

logger = logging.getLogger(__name__)


class MolmoBotModelServer(ModelServer):
    """MolmoBot VLA model server replicating SynthVLAPolicy behavior.

    Args:
        hf_repo: HuggingFace repo or local path for weights (e.g.
            ``allenai/MolmoBot-DROID``).
        execute_horizon: Actions executed before re-querying the model.
        states_mode: Model config override (paper uses ``cross_attn``).
        max_joint_delta: Per-step safety clamp on arm joint deltas.
        primary_image_key: Primary camera key in obs["images"].
        wrist_image_key: Wrist camera key in obs["images"].
    """

    def __init__(
        self,
        hf_repo: str = "allenai/MolmoBot-DROID",
        execute_horizon: int = 8,
        states_mode: str = "cross_attn",
        max_joint_delta: float = 0.2,
        primary_image_key: str = "exo_camera_1",
        wrist_image_key: str = "wrist_camera",
    ) -> None:
        super().__init__()
        self.hf_repo = hf_repo
        self.execute_horizon = execute_horizon
        self.states_mode = states_mode
        self.max_joint_delta = float(max_joint_delta)
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key

        self._wrapper: Any = None
        self.n_obs_steps: int = 1
        self.obs_step_delta: int = 8
        self.action_horizon: int = 16

        # Per-session state: obs_history (list of per-cam image lists),
        # action_buffer (list of (arm, gripper) tuples), buffer_index (int).
        self._sessions: dict[str, dict[str, Any]] = {}

    # -- lifecycle --------------------------------------------------------

    def _load_model(self) -> None:
        if self._wrapper is not None:
            return
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper

        logger.info("Loading MolmoBot from %s (states_mode=%s)", self.hf_repo, self.states_mode)
        self._wrapper = SynthManipMolmoInferenceWrapper(
            checkpoint_path=self.hf_repo,
            states_mode=self.states_mode,
        )
        mc = self._wrapper.model_config
        self.n_obs_steps = int(getattr(mc, "n_obs_steps", 1))
        self.obs_step_delta = int(getattr(mc, "obs_step_delta", 8))
        self.action_horizon = int(getattr(mc, "action_horizon", 16))
        logger.info(
            "MolmoBot loaded: n_obs_steps=%d obs_step_delta=%d action_horizon=%d",
            self.n_obs_steps,
            self.obs_step_delta,
            self.action_horizon,
        )

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        # Reset per-episode state.
        self._sessions[ctx.session_id] = {
            "obs_history": [],  # list[dict[cam_name, np.ndarray]]
            "action_buffer": [],  # list[dict{"arm": (7,), "gripper": (1,)}]
            "buffer_index": 0,
            "step_count": 0,
        }

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._sessions.pop(ctx.session_id, None)

    # -- specs ------------------------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "image": IMAGE_RGB,
            "wrist_image": IMAGE_RGB,
            "state": STATE_JOINT,
            "language": LANGUAGE,
        }

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    # -- inference --------------------------------------------------------

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        self._load_model()
        assert self._wrapper is not None

        state = self._sessions.setdefault(
            ctx.session_id,
            {"obs_history": [], "action_buffer": [], "buffer_index": 0, "step_count": 0},
        )

        # Extract per-camera images from this step's observation.
        images_dict = obs.get("images", {})
        cam_obs: dict[str, np.ndarray] = {}
        for cam_key in (self.primary_image_key, self.wrist_image_key):
            img = images_dict.get(cam_key)
            if img is None:
                raise KeyError(
                    f"MolmoBotModelServer: missing camera '{cam_key}' in obs. Available: {list(images_dict.keys())}"
                )
            cam_obs[cam_key] = np.asarray(img, dtype=np.uint8)
        state["obs_history"].append(cam_obs)

        # Refill action buffer if exhausted.
        if state["buffer_index"] >= self.execute_horizon or not state["action_buffer"]:
            self._refill_buffer(obs, state)

        # Consume next action from buffer.
        action_dict = state["action_buffer"][state["buffer_index"]]
        state["buffer_index"] += 1
        state["step_count"] += 1

        # Per-step safety clamp on arm joint deltas (absolute joint mode).
        current_state = obs.get("states", obs.get("state"))
        if current_state is None:
            raise KeyError("MolmoBotModelServer: missing 'state' in obs")
        current_qpos_arm = np.asarray(current_state, dtype=np.float32).flatten()[:7]

        predicted_arm = action_dict["arm"].astype(np.float32).copy()
        predicted_deltas = predicted_arm - current_qpos_arm
        rel_scale = np.abs(predicted_deltas) / self.max_joint_delta
        max_rel = float(np.max(rel_scale)) if rel_scale.size else 0.0
        if max_rel > 1.0:
            scaled_deltas = predicted_deltas / max_rel
            predicted_arm = current_qpos_arm + scaled_deltas

        # Concatenate arm (7) + gripper (1) into flat 8D action vector for the wire.
        action_vec = np.concatenate([predicted_arm.astype(np.float32), action_dict["gripper"].astype(np.float32)])
        await ctx.send_action({"actions": action_vec})

    # -- internals --------------------------------------------------------

    def _refill_buffer(self, obs: Observation, state: dict[str, Any]) -> None:
        """Call the model to produce a new action chunk (16 actions)."""
        # Build image list from frame history matching SynthVLAPolicy logic:
        # for i in range(n_obs_steps):
        #     frame_idx = current_step - (n_obs_steps - 1 - i) * obs_step_delta
        #     if 0 <= frame_idx < len(history): use that frame
        history = state["obs_history"]
        current_step = len(history) - 1

        # Per-camera frame lists, then flatten in the camera order expected by
        # the wrapper (primary first, wrist second), as in SynthVLAPolicy.
        primary_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        for i in range(self.n_obs_steps):
            frame_idx = current_step - (self.n_obs_steps - 1 - i) * self.obs_step_delta
            if 0 <= frame_idx < len(history):
                primary_frames.append(history[frame_idx][self.primary_image_key])
                wrist_frames.append(history[frame_idx][self.wrist_image_key])

        assert primary_frames, "No frames available for inference"
        images: list[np.ndarray] = [*primary_frames, *wrist_frames]

        # State is the 8D qpos vector sent by the benchmark.
        raw_state = obs.get("states", obs.get("state"))
        if raw_state is None:
            raise KeyError("MolmoBotModelServer: missing 'state' in obs for refill")
        state_vec = np.asarray(raw_state, dtype=np.float32).flatten()

        task_description = obs.get("task_description", "")

        chunk = self._wrapper.get_action_chunk(
            images=images,
            task_description=task_description,
            state=state_vec,
        )  # shape (action_horizon, action_dim) — (16, 8) for MolmoBot

        chunk = np.asarray(chunk, dtype=np.float32)
        assert chunk.ndim == 2 and chunk.shape[1] >= 8, f"Unexpected action chunk shape: {chunk.shape}"

        # Split into per-timestep dicts, clamping gripper (SynthVLAPolicyConfig clamp_gripper=True)
        buffer: list[dict[str, np.ndarray]] = []
        for t in range(chunk.shape[0]):
            arm = chunk[t, :7].copy()
            gripper_raw = chunk[t, 7:8]
            gripper = np.where(gripper_raw > 128, 255.0, 0.0).astype(np.float32)
            buffer.append({"arm": arm, "gripper": gripper})

        state["action_buffer"] = buffer
        state["buffer_index"] = 0


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(MolmoBotModelServer)
