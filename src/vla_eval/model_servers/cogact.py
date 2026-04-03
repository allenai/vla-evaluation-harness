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
# vla-eval = { path = "../../..", editable = true }
# cogact = { git = "https://github.com/microsoft/CogACT.git", rev = "b174a1b86deedfab4d198d935207e7bb0527994e" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

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

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "language": LANGUAGE}

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
    from vla_eval.model_servers.serve import run_server

    run_server(CogACTModelServer)
