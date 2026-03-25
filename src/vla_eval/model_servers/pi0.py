# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "numpy>=1.24",
#     "pytest",
#     "chex",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# openpi = { git = "https://github.com/Physical-Intelligence/openpi.git", rev = "981483dca0fd9acba698fea00aa6e52d56a66c58" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""π₀ / π₀-FAST model server.

Loads an OpenPI policy checkpoint directly and runs inference
in-process.  No external server required.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


class Pi0ModelServer(PredictModelServer):
    """π₀ / π₀-FAST model server using OpenPI direct inference."""

    def __init__(
        self,
        config_name: str = "pi05_libero",
        checkpoint: str | None = None,
        image_key: str = "observation/image",
        wrist_image_key: str | None = "observation/wrist_image",
        state_key: str | None = "observation/state",
        state_dim: int = 8,
        image_resolution: int | None = None,
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.state_key = state_key
        self.state_dim = state_dim
        self.image_resolution = image_resolution
        self._policy = None

    def _maybe_resize(self, img: np.ndarray) -> np.ndarray:
        """Resize image to ``image_resolution`` if set and size differs."""
        if self.image_resolution is None or img.shape[:2] == (self.image_resolution, self.image_resolution):
            return img
        from PIL import Image

        pil = Image.fromarray(img)
        pil = pil.resize((self.image_resolution, self.image_resolution), Image.Resampling.BILINEAR)
        return np.asarray(pil)

    def _load_model(self) -> None:
        if self._policy is not None:
            return
        from openpi.policies import policy_config
        from openpi.training import config as _config

        logger.info("Loading OpenPI config: %s", self.config_name)
        config = _config.get_config(self.config_name)

        checkpoint = self.checkpoint
        if checkpoint is None:
            checkpoint = f"gs://openpi-assets/checkpoints/{self.config_name}"

        logger.info("Loading policy from checkpoint: %s", checkpoint)
        self._policy = policy_config.create_trained_policy(config, checkpoint)
        logger.info("π₀ policy loaded successfully.")

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.wrist_image_key:
            params["send_wrist_image"] = True
        if self.state_key:
            params["send_state"] = True
        return params

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._policy is not None

        openpi_obs: dict[str, Any] = {}

        images_dict = obs.get("images", {})
        img_list = list(images_dict.values()) if isinstance(images_dict, dict) else []
        base_img = np.asarray(img_list[0], dtype=np.uint8) if img_list else np.zeros((256, 256, 3), dtype=np.uint8)
        base_img = self._maybe_resize(base_img)
        openpi_obs[self.image_key] = base_img

        if self.wrist_image_key:
            wrist_img = np.asarray(img_list[1], dtype=np.uint8) if len(img_list) > 1 else np.zeros_like(base_img)
            wrist_img = self._maybe_resize(wrist_img)
            openpi_obs[self.wrist_image_key] = wrist_img

        openpi_obs["prompt"] = obs.get("task_description", "")

        # LIBERO sends "states" (plural), other benchmarks may use "state" (singular)
        if self.state_key:
            raw_state = obs.get("states", obs.get("state"))
            if raw_state is not None:
                openpi_obs[self.state_key] = np.asarray(raw_state, dtype=np.float64)
            else:
                openpi_obs[self.state_key] = np.zeros(self.state_dim, dtype=np.float64)

        result = self._policy.infer(openpi_obs)
        return {"actions": result["actions"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="π₀/π₀-FAST model server (direct loading)")
    parser.add_argument("--config_name", default="pi05_libero", help="OpenPI config name")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (GCS/local). Auto-resolved if omitted.")
    parser.add_argument("--image_key", default="observation/image", help="OpenPI observation key for the image")
    parser.add_argument(
        "--wrist_image_key",
        default="observation/wrist_image",
        help="OpenPI observation key for wrist image (None to disable)",
    )
    parser.add_argument(
        "--state_key", default="observation/state", help="OpenPI observation key for robot state (None to disable)"
    )
    parser.add_argument("--state_dim", type=int, default=8, help="Dimension of the state vector for zero-fill")
    parser.add_argument(
        "--image_resolution", type=int, default=None, help="Resize images to this resolution (e.g. 224 for pi05)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=10, help="Action chunk size")
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--ci", action="store_true", help="Enable Continuous Inference (DRAFT)")
    parser.add_argument("--laas", action="store_true", help="Enable LAAS (DRAFT)")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.laas and not args.ci:
        parser.error("--laas requires --ci")

    # Convert "None" strings to actual None
    wrist_key = None if args.wrist_image_key in (None, "None", "none") else args.wrist_image_key
    state_key = None if args.state_key in (None, "None", "none") else args.state_key

    server = Pi0ModelServer(
        config_name=args.config_name,
        checkpoint=args.checkpoint,
        image_key=args.image_key,
        wrist_image_key=wrist_key,
        state_key=state_key,
        state_dim=args.state_dim,
        image_resolution=args.image_resolution,
        chunk_size=args.chunk_size,
        action_ensemble=args.action_ensemble,
        continuous_inference=args.ci,
        laas=args.laas,
        hz=args.hz,
    )

    logger.info("Pre-loading model...")
    server._load_model()
    logger.info("Model ready, starting server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
