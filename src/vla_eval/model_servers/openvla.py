# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.40.1",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.types import Action, Observation
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


class OpenVLAModelServer(PredictModelServer):
    """OpenVLA model server (openvla/openvla-7b).

    Uses HuggingFace transformers ``AutoModelForVision2Seq`` with the
    built-in ``predict_action()`` method that returns a 7-dim numpy action.
    No native action chunking (chunk_size=1).
    """

    def __init__(
        self,
        model_path: str = "openvla/openvla-7b",
        unnorm_key: str | None = None,
        *,
        chunk_size: int = 1,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self._model = None
        self._processor = None
        self._device = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading OpenVLA from %s on %s", self.model_path, self._device)

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self._device)
        logger.info("OpenVLA model loaded.")

    @staticmethod
    def _obs_to_pil(obs: Observation) -> Any:
        from PIL import Image as PILImage

        images_dict = obs.get("images", {})
        img_array = next(iter(images_dict.values())) if isinstance(images_dict, dict) else images_dict
        return PILImage.fromarray(img_array).convert("RGB") if isinstance(img_array, np.ndarray) else img_array

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        pil_image = self._obs_to_pil(obs)
        task_description = obs.get("task_description", "")
        prompt = f"In: What action should the robot take to {task_description}?\nOut:"

        inputs = self._processor(prompt, pil_image).to(self._device, dtype=torch.bfloat16)

        kwargs: dict[str, Any] = {"do_sample": False}
        if self.unnorm_key:
            kwargs["unnorm_key"] = self.unnorm_key

        action = self._model.predict_action(**inputs, **kwargs)
        # Gripper: RLDS [0=close,1=open] → robosuite [-1=open,+1=close]
        action_arr = np.asarray(action, dtype=np.float32)
        action_arr[..., -1] = -np.sign(2 * action_arr[..., -1] - 1)
        return {"actions": action_arr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVLA model server (uv script)")
    parser.add_argument("--model_path", default="openvla/openvla-7b", help="HuggingFace model ID or local path")
    parser.add_argument("--unnorm_key", default=None, help="Unnormalization key (e.g. 'bridge_orig')")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=1)
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

    server = OpenVLAModelServer(
        model_path=args.model_path,
        unnorm_key=args.unnorm_key,
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
