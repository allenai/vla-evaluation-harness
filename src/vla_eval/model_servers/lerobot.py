# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "vla-eval",
#     "lerobot[pi]",
#     "torch>=2.7,<2.12",
#     "numpy>=1.24",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# lerobot = { git = "https://github.com/huggingface/lerobot.git", rev = "e275ea3960332543e2a9f441356775a53720543f" }
#
# [tool.uv]
# exclude-newer = "2026-07-04T00:00:00Z"
# ///
"""🤗 LeRobot policy model server.

Bridges any LeRobot ``PreTrainedPolicy`` checkpoint (pi0, pi05, ACT,
diffusion, SmolVLA, …) into vla-eval.  The checkpoint is loaded with its
saved pre/post processors (normalisation) and inference runs through
LeRobot's ``predict_action_chunk`` API.

Chunking / concurrency:
    ``predict_action_chunk`` returns the full ``(n_action_steps, action_dim)``
    chunk in one call.  We return that chunk and let vla-eval's
    ``PredictModelServer`` buffer it per-session (``chunk_size`` defaults to the
    policy's ``n_action_steps``).  Same pattern as the π₀ server.  This
    keeps the model server stateless per call and therefore safe under sharded
    multi-session evaluation.  Only **single-observation-step** policies
    (pi0, pi05, SmolVLA, ACT) are supported: chunk buffering skips the
    observations a multi-obs-step policy (diffusion, VQ-BeT) needs, so those are
    rejected at load.

Install the extra matching the target policy, e.g. ``lerobot[pi]`` (pi0/pi05)
or ``lerobot[smolvla]``; the PEP 723 header pins ``lerobot[pi]`` by default.
Runs on Python 3.12 (lerobot's floor; the header caps <3.13 because 3.14's
argparse breaks lerobot's draccus config parsing).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

# Running this file directly (`uv run .../lerobot.py`) puts its own dir first on
# sys.path, shadowing the installed `lerobot` package; drop it before importing.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _here]

logger = logging.getLogger(__name__)

_IMAGE_PREFIX = "observation.images."


def _qualify_image_key(key: str) -> str:
    """Add the ``observation.images.`` prefix unless ``key`` is already an
    ``observation.image*`` feature name (covers both singular and plural forms)."""
    return key if key.startswith("observation.image") else _IMAGE_PREFIX + key


class LeRobotModelServer(PredictModelServer):
    """Model server wrapping a pretrained LeRobot policy."""

    def __init__(
        self,
        policy_type: str,
        checkpoint: str,
        image_keys: dict[str, str] | None = None,
        state_key: str | None = "observation.state",
        device: str = "cuda",
        robot_type: str = "",
        *,
        chunk_size: int | None = None,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            policy_type: LeRobot policy registry name of a single-observation-step
                policy (e.g. ``"pi0"``, ``"pi05"``, ``"act"``, ``"smolvla"``).
                Multi-obs-step policies (diffusion, VQ-BeT) are rejected at load.
            checkpoint: HuggingFace Hub id or local path of the trained checkpoint.
            image_keys: Map from benchmark camera name (key in ``obs["images"]``)
                to the policy's image feature.  The ``observation.images.`` prefix
                is added automatically if omitted.  When ``None``, benchmark
                cameras are mapped positionally (sorted) onto the policy's
                expected image features.
            state_key: Policy feature name for proprioceptive state, or ``None``
                to send no state.
            device: Torch device for the policy.
            robot_type: Optional embodiment tag forwarded to the preprocessor
                (needed by some multi-embodiment checkpoints).
            chunk_size: Actions buffered per inference.  ``None`` (default) uses
                the policy's ``n_action_steps``.
        """
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.policy_type = policy_type
        self.checkpoint = checkpoint
        self.state_key = None if state_key in (None, "None", "none") else state_key
        self.robot_type = robot_type
        self._image_keys_arg = image_keys
        self._logged_image_map = False

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        self._torch = torch
        if "cuda" in device and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"
        self._device = torch.device(device)

        logger.info("Loading LeRobot policy %s from %s", policy_type, checkpoint)
        policy_cls = get_policy_class(policy_type)
        # Put device on the config so from_pretrained loads weights straight onto it,
        # not the checkpoint's device (which can OOM the wrong GPU for cpu / cuda:N).
        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.device = str(self._device)
        self._policy = policy_cls.from_pretrained(checkpoint, config=cfg)
        self._policy.to(self._device)
        self._policy.eval()

        # Chunk buffering skips predict() for the buffered steps, so a policy needing a
        # multi-step observation history (n_obs_steps>1) would see stale observations.
        n_obs = getattr(self._policy.config, "n_obs_steps", 1) or 1
        if n_obs > 1:
            raise ValueError(
                f"{policy_type} has n_obs_steps={n_obs}; this bridge supports only single-"
                f"observation-step policies (pi0, pi05, SmolVLA, ACT). Multi-obs-step policies "
                f"(diffusion, VQ-BeT, …) need an observation history that chunk buffering skips."
            )

        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self._device)}},
        )

        self._expected_image_keys = list(self._policy.config.image_features.keys())
        self._validate_image_keys()
        if self.chunk_size is None:
            self.chunk_size = getattr(self._policy.config, "n_action_steps", None)
        logger.info(
            "LeRobot policy loaded: image_features=%s state=%s chunk_size=%s",
            self._expected_image_keys,
            self.state_key,
            self.chunk_size,
        )

    def _validate_image_keys(self) -> None:
        """Fail fast when an explicit image_keys target isn't a policy image feature."""
        if not self._image_keys_arg:
            return
        valid = set(self._expected_image_keys)
        for bench, key in self._image_keys_arg.items():
            qkey = _qualify_image_key(key)
            if qkey not in valid:
                raise ValueError(
                    f"image_keys[{bench!r}]={key!r} -> {qkey!r} is not an image feature of "
                    f"{self.policy_type}; expected one of {sorted(valid)}."
                )

    # ------------------------------------------------------------------
    # Interface declarations
    # ------------------------------------------------------------------

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.state_key:
            params["send_state"] = True
        if len(self._expected_image_keys) > 1:
            params["send_wrist_image"] = True
        return params

    def get_action_spec(self) -> dict[str, DimSpec]:
        # Action convention is checkpoint-specific (set by the training dataset);
        # declared RAW so the orchestrator does not warn spuriously.
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "language": LANGUAGE}
        if self.state_key:
            spec["state"] = RAW
        return spec

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _resolve_image_map(self, images: dict[str, np.ndarray]) -> dict[str, str]:
        """Map benchmark camera names to policy image feature keys."""
        if self._image_keys_arg:
            resolved = {
                bench: _qualify_image_key(key) for bench, key in self._image_keys_arg.items() if bench in images
            }
            missing = [b for b in self._image_keys_arg if b not in images]
        else:
            # Positional fallback: sorted benchmark cameras onto sorted policy features.
            resolved = dict(zip(sorted(images), self._expected_image_keys))
            missing = []
        if not self._logged_image_map:
            if missing:
                logger.warning("image_keys cameras absent from observation: %s (available: %s)", missing, list(images))
            unfed = [
                k for k in self._expected_image_keys if k not in set(resolved.values()) and "empty_camera" not in k
            ]
            if unfed:
                logger.warning("policy image features left unfed (model will mask them): %s", unfed)
            logger.info("Image mapping (benchmark -> policy): %s", resolved)
            self._logged_image_map = True
        return resolved

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        torch = self._torch
        from lerobot.policies.utils import prepare_observation_for_inference

        images = obs.get("images", {}) or {}
        frame: dict[str, np.ndarray] = {}
        for bench_key, policy_key in self._resolve_image_map(images).items():
            frame[policy_key] = np.asarray(images[bench_key], dtype=np.uint8)

        if self.state_key:
            raw_state = obs.get("states", obs.get("state"))
            if raw_state is not None:
                frame[self.state_key] = np.asarray(raw_state, dtype=np.float32)

        task = obs.get("task_description", "")
        frame = prepare_observation_for_inference(frame, self._device, task=task, robot_type=self.robot_type)
        frame = self._preprocess(frame)

        with torch.inference_mode():
            chunk = self._policy.predict_action_chunk(frame)
        chunk = self._postprocess(chunk)

        # (B, n_steps, action_dim) -> (n_steps, action_dim); vla-eval buffers per session.
        actions = chunk.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        return {"actions": actions}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        # Clear any per-policy observation-history queues between episodes.
        self._policy.reset()
        await super().on_episode_start(config, ctx)


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(LeRobotModelServer)
