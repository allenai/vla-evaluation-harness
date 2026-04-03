# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "torchvision>=0.17",
#     "transformers>=4.57",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "safetensors",
#     "scipy>=1.11",
#     "einops",
#     "timm",
#     "accelerate",
#     "peft",
#     "fastapi",
#     "uvicorn",
#     "json_numpy",
#     "websockets",
#     "opencv-python"
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""SimVLA model server — SmolVLM-based VLA with action chunking.

SimVLA uses a SmolVLM vision-language backbone with a continuous action head.
It accepts two camera views (primary + wrist) and 8D proprioceptive state,
producing chunked 7D actions (delta_xyz, delta_axisangle, gripper).
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

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

_DISABLED_STRINGS = {"", "none", "null"}
_SIMVLA_REPO_URL = "https://github.com/LUOyk1999/SimVLA.git"
_SIMVLA_REPO_REV = "32700d0ad8991996e123e4b685abe370ce6e9aab"


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    return None if value.strip().lower() in _DISABLED_STRINGS else value


class SimVLAModelServer(PredictModelServer):
    """SimVLA (SmolVLM-VLA) model server.

    Loads a SmolVLM-VLA checkpoint and runs inference with two camera views
    (primary image + wrist image), proprioceptive state, and a language
    instruction.  Produces chunked 7D actions per step.

    Args:
        checkpoint: Path to SimVLA checkpoint directory.
        norm_stats: Path to normalization statistics JSON file.
        smolvlm_model: SmolVLM base model path or HuggingFace repo ID.
        image_size: Input image resolution (square).
        chunk_size: Number of actions per inference call.
        action_ensemble: Strategy for blending overlapping action chunks.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        norm_stats: str | None = None,
        norm_stats_subdir: str = "norm_stats",
        norm_stats_filename: str = "libero_norms.json",
        smolvlm_model: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        reference_repo_dir: str | None = None,
        repo_url: str = _SIMVLA_REPO_URL,
        repo_rev: str = _SIMVLA_REPO_REV,
        repo_cache_dir: str | None = None,
        image_size: int = 384,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.checkpoint = checkpoint
        self.norm_stats = _normalize_optional_str(norm_stats)
        self.norm_stats_subdir = norm_stats_subdir
        self.norm_stats_filename = norm_stats_filename
        self.smolvlm_model = smolvlm_model
        self.reference_repo_dir = _normalize_optional_str(reference_repo_dir)
        self.repo_url = repo_url
        self.repo_rev = repo_rev
        self.repo_cache_dir = _normalize_optional_str(repo_cache_dir)
        self.image_size = image_size
        self._model = None
        self._processor = None
        self._device = None
        self._transform = None
        self._reference_repo_path: Path | None = None

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": STATE_EEF_POS_AA_GRIP, "language": LANGUAGE}

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def _resolve_reference_repo(self) -> Path:
        """Resolve the SimVLA reference repo, cloning from GitHub if needed."""
        if self._reference_repo_path is not None:
            return self._reference_repo_path

        # 1. Explicit local path
        if self.reference_repo_dir is not None:
            path = Path(self.reference_repo_dir).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"reference_repo_dir does not exist: {path}")
            self._reference_repo_path = path
            return path

        # 2. Clone from GitHub into cache
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("git is required to fetch the SimVLA reference repo")

        cache_root = Path(self.repo_cache_dir or "~/.cache/vla-eval/reference-repos").expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / f"simvla-{self.repo_rev[:12]}"
        if target.exists():
            self._reference_repo_path = target
            return target

        logger.info("Cloning SimVLA reference repo from %s (rev %s)...", self.repo_url, self.repo_rev[:12])
        tmp_dir = Path(tempfile.mkdtemp(prefix="simvla-", dir=cache_root))
        try:
            subprocess.run([git, "init"], cwd=tmp_dir, check=True, capture_output=True, text=True)
            subprocess.run(
                [git, "remote", "add", "origin", self.repo_url], cwd=tmp_dir, check=True, capture_output=True
            )
            subprocess.run(
                [git, "fetch", "--depth", "1", "origin", self.repo_rev],
                cwd=tmp_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([git, "checkout", "FETCH_HEAD"], cwd=tmp_dir, check=True, capture_output=True, text=True)
            try:
                tmp_dir.rename(target)
            except FileExistsError:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        self._reference_repo_path = target
        return target

    def _ensure_reference_imports(self) -> Path:
        """Add SimVLA reference repo to sys.path and return the repo root."""
        repo_root = self._resolve_reference_repo()
        if not (repo_root / "models" / "modeling_smolvlm_vla.py").exists():
            raise FileNotFoundError(f"SimVLA reference repo missing models/modeling_smolvlm_vla.py: {repo_root}")
        repo_str = str(repo_root)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        return repo_root

    def _resolve_norm_stats(self) -> Path | None:
        """Resolve normalization statistics file.

        Resolution order:
        1. Explicit ``norm_stats`` path (if provided and exists).
        2. ``<reference_repo>/<norm_stats_subdir>/<norm_stats_filename>``
           (auto-discovered from the cloned/local SimVLA repo).
        """
        # 1. Explicit path
        if self.norm_stats is not None:
            p = Path(self.norm_stats).expanduser().resolve()
            if p.is_file():
                return p
            logger.warning("Explicit norm_stats path does not exist: %s — trying reference repo", p)

        # 2. Reference repo subdirectory
        repo_root = self._resolve_reference_repo()
        candidate = repo_root / self.norm_stats_subdir / self.norm_stats_filename
        if candidate.is_file():
            return candidate

        return None

    def _load_model(self) -> None:
        if self._model is not None:
            return

        import torch
        from torchvision import transforms

        self._ensure_reference_imports()

        from models.modeling_smolvlm_vla import SmolVLMVLA
        from models.processing_smolvlm_vla import SmolVLMVLAProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading SimVLA from %s on %s", self.checkpoint, self._device)

        self._model = SmolVLMVLA.from_pretrained(self.checkpoint)
        self._model = self._model.to(self._device)
        self._model.eval()

        self._processor = SmolVLMVLAProcessor.from_pretrained(self.smolvlm_model)

        norm_stats_path = self._resolve_norm_stats()
        if norm_stats_path is not None:
            logger.info("Loading norm stats from: %s", norm_stats_path)
            self._model.action_space.load_norm_stats(str(norm_stats_path))
        else:
            logger.warning("No norm_stats found — actions may be unnormalized")

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        logger.info("SimVLA model loaded (image_size=%d, chunk_size=%s)", self.image_size, self.chunk_size)

    def _preprocess_images(self, image0: np.ndarray, image1: np.ndarray) -> tuple[Any, Any]:
        """Preprocess two camera views into model input tensors."""
        import torch
        from PIL import Image as PILImage

        img0 = PILImage.fromarray(image0.astype(np.uint8)).convert("RGB")
        img1 = PILImage.fromarray(image1.astype(np.uint8)).convert("RGB")

        assert self._transform is not None
        img0_t = self._transform(img0)
        img1_t = self._transform(img1)

        # Pad to 3 views (model expects all views stacked)
        padding = torch.zeros_like(img0_t)
        images = torch.stack([img0_t, img1_t, padding], dim=0)
        image_mask = torch.tensor([[True, True, False]])

        return images.unsqueeze(0), image_mask

    def _extract_images(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Extract primary and wrist images from an observation."""
        images_dict = obs.get("images", {})
        image_keys = list(images_dict.keys()) if isinstance(images_dict, dict) else []
        if len(image_keys) >= 2:
            image0 = np.asarray(images_dict[image_keys[0]], dtype=np.uint8)
            image1 = np.asarray(images_dict[image_keys[1]], dtype=np.uint8)
        elif len(image_keys) == 1:
            image0 = np.asarray(images_dict[image_keys[0]], dtype=np.uint8)
            image1 = np.zeros_like(image0)
        else:
            image0 = (
                np.asarray(images_dict, dtype=np.uint8)
                if not isinstance(images_dict, dict)
                else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            )
            image1 = np.zeros_like(image0)
        return image0, image1

    def _extract_state(self, obs: Observation) -> np.ndarray:
        """Extract and pad/truncate 8D proprioceptive state."""
        state = np.asarray(obs.get("state", obs.get("states", np.zeros(8))), dtype=np.float32).flatten()
        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)))
        return state[:8]

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        import torch

        self._load_model()
        assert self._model is not None
        assert self._processor is not None

        batch_size = len(obs_batch)
        all_images = []
        all_masks = []
        all_proprios = []
        prompts = []

        for obs in obs_batch:
            image0, image1 = self._extract_images(obs)
            state = self._extract_state(obs)
            prompts.append(obs.get("task_description", ""))

            images, image_mask = self._preprocess_images(image0, image1)
            all_images.append(images)
            all_masks.append(image_mask)
            all_proprios.append(torch.tensor(state, dtype=torch.float32))

        # Stack into batched tensors
        images_batch = torch.cat(all_images, dim=0).to(self._device)  # (B, 3, C, H, W)
        masks_batch = torch.cat(all_masks, dim=0).to(self._device)  # (B, 3)
        proprio_batch = torch.stack(all_proprios, dim=0).to(self._device)  # (B, 8)

        lang = self._processor.encode_language(prompts)
        lang = {k: v.to(self._device) for k, v in lang.items()}

        with torch.no_grad():
            actions = self._model.generate_actions(
                input_ids=lang["input_ids"],
                image_input=images_batch,
                image_mask=masks_batch,
                proprio=proprio_batch,
                steps=self.chunk_size or 10,
            )

        actions = actions.cpu().numpy()  # (B, chunk_size, 7)
        return [{"actions": actions[i]} for i in range(batch_size)]


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(SimVLAModelServer)
