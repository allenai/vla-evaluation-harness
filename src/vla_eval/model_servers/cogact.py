# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "cogact",
#     "torch>=2.2",
#     "transformers==4.40.1",
#     "timm==0.9.10",
#     "tokenizers==0.19.1",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate>=0.25.0",
#     "einops",
#     "sentencepiece==0.1.99",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../.." }
# cogact = { git = "https://github.com/microsoft/CogACT.git", rev = "b174a1b86deedfab4d198d935207e7bb0527994e" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
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


class CogACTModelServer(PredictModelServer):
    """CogACT VLA model server (microsoft/CogACT).

    Uses the official ``vla`` package with ``load_vla()`` and
    ``predict_action()`` / ``predict_action_batch()``.  Denormalization
    is handled internally by the model via ``unnorm_key``.
    """

    def __init__(
        self,
        model_path: str = "CogACT/CogACT-Base",
        action_model_type: str = "DiT-B",
        future_action_window_size: int = 15,
        unnorm_key: str | None = None,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        *,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.action_model_type = action_model_type
        self.future_action_window_size = future_action_window_size
        self.unnorm_key = unnorm_key
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from vla import load_vla

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Loading CogACT from %s (type=%s, window=%d) on %s",
            self.model_path,
            self.action_model_type,
            self.future_action_window_size,
            device,
        )

        self._model = load_vla(
            self.model_path,
            load_for_training=False,
            action_model_type=self.action_model_type,
            future_action_window_size=self.future_action_window_size,
        )
        self._model.to(device).eval()
        logger.info("CogACT model loaded.")

    @staticmethod
    def _obs_to_pil(obs: Observation) -> Any:
        """Extract the first image from an observation and convert to PIL RGB."""
        from PIL import Image as PILImage

        images_dict = obs.get("images", {})
        img_array = next(iter(images_dict.values())) if isinstance(images_dict, dict) else images_dict
        return PILImage.fromarray(img_array).convert("RGB") if isinstance(img_array, np.ndarray) else img_array

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._model is not None

        pil_image = self._obs_to_pil(obs)
        prompt = obs.get("task_description", "")

        actions, _ = self._model.predict_action(
            pil_image,
            prompt,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )
        return {"actions": actions}

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[dict[str, Any]]:
        self._load_model()
        assert self._model is not None

        pil_images = [self._obs_to_pil(obs) for obs in obs_batch]
        prompts = [obs.get("task_description", "") for obs in obs_batch]

        actions, _ = self._model.predict_action_batch(
            pil_images,
            prompts,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
        )
        return [{"actions": actions[i]} for i in range(len(obs_batch))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogACT model server (uv script)")
    parser.add_argument("--model_path", default="CogACT/CogACT-Base", help="HuggingFace model ID or local path")
    parser.add_argument("--action_model_type", default="DiT-B", choices=["DiT-S", "DiT-B", "DiT-L"])
    parser.add_argument("--future_action_window_size", type=int, default=15)
    parser.add_argument("--unnorm_key", default=None, help="Dataset key for action denormalization")
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--use_ddim", action="store_true", default=True)
    parser.add_argument("--no_ddim", dest="use_ddim", action="store_false")
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # CI/LAAS flags — DRAFT, untested
    parser.add_argument("--ci", action="store_true", help="Enable Continuous Inference (DRAFT, untested)")
    parser.add_argument("--laas", action="store_true", help="Enable Latency-Aware Action Selection (DRAFT, untested)")
    parser.add_argument("--hz", type=float, default=10.0, help="Environment Hz for LAAS delay computation")
    parser.add_argument("--max_batch_size", type=int, default=16, help="Max batch size for batched inference")
    parser.add_argument(
        "--max_wait_time", type=float, default=0.05, help="Max wait time in seconds for batch accumulation"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if args.laas and not args.ci:
        parser.error("--laas requires --ci")

    kwargs: dict[str, Any] = {
        "action_ensemble": args.action_ensemble,
        "continuous_inference": args.ci,
        "laas": args.laas,
        "hz": args.hz,
    }
    if args.max_batch_size > 1:
        kwargs["max_batch_size"] = args.max_batch_size
        kwargs["max_wait_time"] = args.max_wait_time

    server = CogACTModelServer(
        model_path=args.model_path,
        action_model_type=args.action_model_type,
        future_action_window_size=args.future_action_window_size,
        unnorm_key=args.unnorm_key,
        cfg_scale=args.cfg_scale,
        use_ddim=args.use_ddim,
        num_ddim_steps=args.num_ddim_steps,
        chunk_size=args.chunk_size,
        **kwargs,
    )

    logger.info("Pre-loading model...")
    server._load_model()
    mode = (
        f"batch (max_batch={args.max_batch_size}, wait={args.max_wait_time}s)" if args.max_batch_size > 1 else "single"
    )
    logger.info("Model ready [%s], starting server on ws://%s:%d", mode, args.host, args.port)
    serve(server, host=args.host, port=args.port)
