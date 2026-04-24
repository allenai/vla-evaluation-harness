# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "torchvision>=0.17",
#     "transformers>=4.56,<5.0",  # transformers 5.x made get_rope_index's mm_token_type_ids positional
#     "diffusers>=0.25",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-04-09T00:00:00Z"
# ///
"""VLANeXt model server.

Upstream VLANeXt ships a non-SPDX license string in its pyproject.toml and
cannot be installed as a wheel via pip/uv.  We shallow-clone the repo on
first use — following the same pattern as ``mme_vla.py`` — so users don't
have to configure anything.  Set ``VLANEXT_ROOT`` to point at an editable
local clone if needed.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_VLANEXT_REPO = "https://github.com/DravenALG/VLANeXt.git"
_VLANEXT_REV = "ff134c8"
_VLANEXT_CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "vla-eval/vlanext")


def _ensure_vlanext() -> None:
    """Make ``src.models.VLANeXt`` importable by shallow-cloning on first use.

    If ``VLANEXT_ROOT`` is set, it's used as-is and must already be a valid
    clone — we never ``git clone`` into a user-specified directory.  Without
    the env var, the repo is cloned lazily into ``_VLANEXT_CACHE``.
    """
    user_root = os.environ.get("VLANEXT_ROOT")
    if user_root:
        if not os.path.isdir(os.path.join(user_root, "src", "models")):
            raise RuntimeError(
                f"VLANEXT_ROOT={user_root} is not a valid VLANeXt clone "
                f"(missing src/models). Unset it to auto-clone into {_VLANEXT_CACHE}."
            )
        root = user_root
    else:
        root = _VLANEXT_CACHE
        if not os.path.isdir(os.path.join(root, "src", "models")):
            logger.info("Cloning VLANeXt from %s @ %s …", _VLANEXT_REPO, _VLANEXT_REV)
            # Full clone (GitHub rejects shallow-fetching arbitrary SHAs by
            # default) followed by a pinned checkout.
            subprocess.check_call(["git", "clone", _VLANEXT_REPO, root])
            subprocess.check_call(["git", "-C", root, "checkout", _VLANEXT_REV])
    if root not in sys.path:
        sys.path.insert(0, root)


# LIBERO suite-specific action denormalization bounds (first 6 dims, excluding gripper).
# From VLANeXt/src/datasets/libero_act.py
ACTION_BOUNDS: dict[str, tuple[list[float], list[float]]] = {
    "libero_spatial": (
        [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475],
        [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375],
    ),
    "libero_object": (
        [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943],
        [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174],
    ),
    "libero_goal": (
        [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414],
        [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
    ),
    "libero_10": (
        [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574],
        [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375],
    ),
}


class VLANeXtModelServer(PredictModelServer):
    """VLANeXt model server (DravenALG/VLANeXt).

    Loads a VLANeXt checkpoint (Qwen3-VL-2B + SigLIP2 + diffusion action head)
    and runs inference with flow-matching denoising.  Returns 8-action chunks.
    """

    def __init__(
        self,
        checkpoint: str,
        suite: str = "libero_spatial",
        *,
        attn_implementation: str = "sdpa",
        center_crop_ratio: float = 1.0,
        chunk_size: int = 8,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self.suite = suite
        self.attn_implementation = attn_implementation
        self.center_crop_ratio = center_crop_ratio
        self._model = None
        self._device = None
        self._state_histories: dict[str, list[np.ndarray]] = {}

        if suite not in ACTION_BOUNDS:
            raise ValueError(f"Unknown suite {suite!r}. Choose from {list(ACTION_BOUNDS)}")
        self._action_min = np.array(ACTION_BOUNDS[suite][0], dtype=np.float32)
        self._action_max = np.array(ACTION_BOUNDS[suite][1], dtype=np.float32)

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_state": True, "send_wrist_image": True, "quat_no_antipodal": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "wrist": IMAGE_RGB, "language": LANGUAGE, "state": STATE_EEF_POS_AA_GRIP}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_start(config, ctx)
        self._state_histories[ctx.session_id] = []

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        await super().on_episode_end(result, ctx)
        self._state_histories.pop(ctx.session_id, None)

    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        """Capture proprioception from every observation (even buffered steps).

        The base PredictModelServer only calls predict() when the chunk buffer
        is empty (every chunk_size steps).  VLANeXt needs dense proprioception
        history — one state per env step — so we extract and accumulate here
        before delegating to the normal chunking logic.
        """
        # LIBERO sends "states" (plural); other benchmarks / the smoke stub use "state".
        state = obs.get("states", obs.get("state"))
        if state is not None:
            state_arr = np.asarray(state, dtype=np.float32)
            if len(state_arr) == 8:
                gripper_scalar = np.clip(
                    1.0 - (np.mean(np.abs(state_arr[6:8])) / 0.04),
                    0.0,
                    1.0,
                )
                state_arr = np.concatenate([state_arr[:6], [gripper_scalar]])
            sid = ctx.session_id
            if sid not in self._state_histories:
                self._state_histories[sid] = []
            self._state_histories[sid].append(state_arr)
            hist_len = len(self._state_histories[sid])
            if hist_len <= 2 or hist_len % 8 == 0:
                logger.debug(
                    "on_obs step=%d hist_len=%d state_7d=%s",
                    ctx.step,
                    hist_len,
                    np.array2string(state_arr, precision=4, separator=","),
                )

        await super().on_observation(obs, ctx)

    @staticmethod
    def _center_crop(img: np.ndarray, ratio: float) -> np.ndarray:
        h, w = img.shape[:2]
        crop = max(1, int(round(min(h, w) * ratio)))
        top = (h - crop) // 2
        left = (w - crop) // 2
        return img[top : top + crop, left : left + crop]

    @staticmethod
    def _resolve_checkpoint(checkpoint: str, suite: str) -> str:
        """Resolve *checkpoint* to a local ``.pt`` path.

        If *checkpoint* is already a local file it is returned as-is.
        Otherwise it is treated as a HuggingFace model ID and downloaded
        via ``huggingface_hub.snapshot_download``.  The checkpoint file
        matching *suite* is returned.
        """
        path = Path(checkpoint)
        if path.is_file() and path.suffix == ".pt":
            return str(path)

        from huggingface_hub import snapshot_download

        logger.info("Downloading model from HuggingFace Hub: %s", checkpoint)
        local_dir = Path(snapshot_download(checkpoint))
        candidates = sorted(
            [p for p in local_dir.iterdir() if p.suffix == ".pt"],
            key=lambda p: p.name,
        )
        if not candidates:
            raise FileNotFoundError(f"No .pt files in {local_dir} (contents: {[p.name for p in local_dir.iterdir()]})")
        # Require a checkpoint matching the suite name — otherwise we'd silently
        # load the wrong weights, which also breaks the suite-specific
        # action-denormalization bounds.
        for c in candidates:
            if suite in c.stem:
                return str(c)
        raise FileNotFoundError(
            f"No checkpoint matching suite {suite!r} in {local_dir}. Available: {[c.name for c in candidates]}"
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return

        _ensure_vlanext()
        from src.models.VLANeXt import VLANeXt

        ckpt_path = self._resolve_checkpoint(self.checkpoint, self.suite)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading VLANeXt from %s on %s", ckpt_path, self._device)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        train_config = checkpoint["config"]
        model_config = train_config["model"]
        data_config = train_config["data"]

        model = VLANeXt(
            lmm_path=model_config["lmm_path"],
            vision_encoder_path=model_config.get("vision_encoder_path", "google/siglip2-base-patch16-256"),
            action_dim=model_config["action_dim"],
            num_actions=data_config["future_len"],
            num_queries=model_config["num_queries"],
            num_history=data_config["history_len"],
            loss_type=model_config.get("loss_type", "diffusion"),
            future_image_loss_weight=float(model_config.get("future_image_loss_weight", 1.0)),
            num_train_timesteps=model_config.get("num_train_timesteps", model_config.get("diffusion_steps", 1000)),
            num_inference_timesteps=5,  # Official eval uses 5 (overrides checkpoint default of 10)
            scheduler_type=model_config["scheduler_type"],
            condition_type=model_config.get("condition_type", "loose"),
            policy_hidden_size=model_config.get("policy_hidden_size", 1024),
            policy_depth=model_config.get("policy_depth", 29),
            policy_num_heads=model_config.get("policy_num_heads", 12),
            policy_mlp_ratio=model_config.get("policy_mlp_ratio", 4.0),
            use_proprio_input_vlm=model_config.get("use_proprio_input_vlm", True),
            use_action_input_policy=model_config.get("use_action_input_policy", False),
            use_transformer_proprio_projector=model_config.get("use_transformer_proprio_projector", False),
            projector_depth=model_config["projector_depth"],
            projector_num_heads=model_config["projector_num_heads"],
            use_transformer_connector=model_config["use_transformer_connector"],
            connector_depth=model_config["connector_depth"],
            connector_num_heads=model_config["connector_num_heads"],
            backbone_mode=model_config.get("backbone_mode", "finetune"),
            gradient_checkpointing=False,
            num_bins=model_config.get("num_bins", 256),
            action_vqvae=model_config.get("action_vqvae", None),
            generator_hidden_size=model_config.get("generator_hidden_size", 768),
            generator_depth=model_config.get("generator_depth", 12),
            generator_num_heads=model_config.get("generator_num_heads", 12),
            generator_mlp_ratio=model_config.get("generator_mlp_ratio", 4.0),
            attn_implementation=self.attn_implementation,
            dct_loss_weight=model_config.get("dct_loss_weight", 0.1),
            dct_low_freq_weight=model_config.get("dct_low_freq_weight", 1.0),
            dct_high_freq_weight=model_config.get("dct_high_freq_weight", 1.0),
            dct_freq_split=model_config.get("dct_freq_split", 0.125),
            dct_similarity_type=model_config.get("dct_similarity_type", "mae"),
        )

        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded state dict. Missing: %d, Unexpected: %d", len(missing), len(unexpected))

        model.to(self._device, dtype=torch.bfloat16)
        model.eval()
        model.train_config = train_config
        self._model = model
        logger.info("VLANeXt model loaded.")

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        from PIL import Image as PILImage

        self._load_model()
        assert self._model is not None

        device = self._device
        data_cfg = self._model.train_config["data"]
        view_mode = data_cfg.get("view_mode", "single")

        # --- Images (center crop + resize to match training preprocessing) ---
        images_dict = obs.get("images", {})
        agentview = images_dict.get("agentview", next(iter(images_dict.values())))
        if isinstance(agentview, np.ndarray) and self.center_crop_ratio < 1.0:
            agentview = self._center_crop(agentview, self.center_crop_ratio)
        pil_agentview = (
            PILImage.fromarray(agentview).convert("RGB") if isinstance(agentview, np.ndarray) else agentview
        )
        if self.center_crop_ratio < 1.0:
            pil_agentview = pil_agentview.resize((256, 256), PILImage.Resampling.LANCZOS)

        images = [pil_agentview]
        if view_mode == "multi" and "wrist" in images_dict:
            wrist = images_dict["wrist"]
            if isinstance(wrist, np.ndarray) and self.center_crop_ratio < 1.0:
                wrist = self._center_crop(wrist, self.center_crop_ratio)
            pil_wrist = PILImage.fromarray(wrist).convert("RGB") if isinstance(wrist, np.ndarray) else wrist
            if self.center_crop_ratio < 1.0:
                pil_wrist = pil_wrist.resize((256, 256), PILImage.Resampling.LANCZOS)
            images.append(pil_wrist)

        # --- Task description ---
        task_description = obs.get("task_description", "")

        # --- Proprioception (dense history accumulated via on_observation) ---
        proprioception = None
        num_history = self._model.num_history  # 8
        if self._model.use_proprio_input_vlm:
            sid = ctx.session_id
            history = self._state_histories.get(sid, [])
            if history:
                states = history[-num_history:]
                if len(states) < num_history:
                    states = [states[0]] * (num_history - len(states)) + states
                proprio_np = np.stack(states)
                proprioception = (
                    torch.tensor(
                        proprio_np,
                        dtype=torch.bfloat16,
                    )
                    .unsqueeze(0)
                    .to(device)
                )
                logger.debug(
                    "predict step=%d hist_total=%d proprio_shape=%s grip_vals=%s",
                    ctx.step,
                    len(history),
                    proprioception.shape,
                    np.array2string(proprio_np[:, 6], precision=4, separator=","),
                )
            else:
                proprioception = torch.zeros(
                    1,
                    num_history,
                    self._model.action_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )

        # --- Processor (Qwen3-VL) ---
        processor = self._model.processor
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": task_description})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Processor may emit `mm_token_type_ids` with newer transformers, but
        # upstream VLANeXt.predict_action() does not accept it — filter it out.
        valid_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        inputs = {k: v.to(device) for k, v in inputs.items() if k in valid_keys}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # --- Inference ---
        with torch.no_grad():
            action_pred = self._model.predict_action(
                proprioception=proprioception,
                **inputs,
            )

        # action_pred shape: (1, num_actions, action_dim) = (1, 8, 7)
        actions = action_pred[0].float().cpu().numpy()  # (8, 7)

        # --- Denormalize & binarize ---
        actions[:, :6] = (actions[:, :6] + 1) / 2 * (self._action_max - self._action_min) + self._action_min
        actions[:, 6] = np.where(actions[:, 6] > 0, 1.0, -1.0)

        return {"actions": actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(VLANeXtModelServer)
