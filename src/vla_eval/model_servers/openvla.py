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

import io
import logging
from typing import Any

import numpy as np

from vla_eval.specs import GRIPPER_CLOSE_POS, IMAGE_RGB, LANGUAGE, POSITION_DELTA, ROTATION_AA, DimSpec
from vla_eval.model_servers.base import SessionContext
from vla_eval.types import Action, Observation
from vla_eval.model_servers.predict import PredictModelServer

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
        jpeg_roundtrip: bool = False,
        center_crop: bool = False,
        chunk_size: int = 1,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.unnorm_key = unnorm_key
        self.jpeg_roundtrip = jpeg_roundtrip
        self.center_crop = center_crop
        self._model = None
        self._processor = None
        self._device = None

    def get_observation_params(self) -> dict[str, Any]:
        return {"seed": 0}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "language": LANGUAGE}

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

    def _preprocess_image(self, obs: Observation) -> Any:
        """Convert observation image to PIL with optional RLDS-matching preprocessing."""
        from PIL import Image as PILImage

        images_dict = obs.get("images", {})
        img_array = next(iter(images_dict.values())) if isinstance(images_dict, dict) else images_dict
        if isinstance(img_array, np.ndarray):
            pil: PILImage.Image = PILImage.fromarray(img_array).convert("RGB")
        elif isinstance(img_array, PILImage.Image):
            pil = img_array
        else:
            raise TypeError(f"expected ndarray or PIL Image, got {type(img_array).__name__}")

        if self.jpeg_roundtrip:
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            buf.seek(0)
            pil = PILImage.open(buf).convert("RGB")

        # Resize to 224×224 with Lanczos (matches reference eval).
        pil = pil.resize((224, 224), resample=PILImage.Resampling.LANCZOS)

        if self.center_crop:
            # Center crop (scale=0.9) then resize back — matches training augmentation.
            w, h = pil.size
            crop_h = int(h * (0.9**0.5))
            crop_w = int(w * (0.9**0.5))
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            pil = pil.crop((left, top, left + crop_w, top + crop_h))
            pil = pil.resize((w, h), resample=PILImage.Resampling.LANCZOS)

        return pil

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        pil_image = self._preprocess_image(obs)
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
    from vla_eval.model_servers.serve import run_server

    run_server(OpenVLAModelServer)
