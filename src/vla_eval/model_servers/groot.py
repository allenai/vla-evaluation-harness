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
# vla-eval = { path = "../../.." }
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

import argparse
import logging
import os
from typing import Any

import numpy as np

from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


class GR00TModelServer(PredictModelServer):
    """GR00T N1.6 model server using Isaac-GR00T Gr00tPolicy."""

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.6-3B",
        embodiment_tag: str = "GR1",
        video_key: str | None = None,
        action_keys: list[str] | None = None,
        *,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.embodiment_tag = embodiment_tag
        self.video_key = video_key  # None = auto-detect from modality config
        self.action_keys = action_keys
        self._policy = None
        self._modality_config: dict[str, Any] | None = None
        self._state_dims: dict[str, int] = {}
        self._language_key: str = "task"

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
            "https://raw.githubusercontent.com/NVIDIA/Isaac-GR00T/main/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/"
        )
        os.makedirs(eagle_dir, exist_ok=True)
        for fname in missing:
            url = base_url + fname
            dst = os.path.join(eagle_dir, fname)
            logger.info("Downloading missing Eagle data file: %s", fname)
            urllib.request.urlretrieve(url, dst)

    def _load_model(self) -> None:
        if self._policy is not None:
            return
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

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._load_model()
        assert self._policy is not None and self._modality_config is not None

        # Determine video key: explicit arg or first from modality config
        video_key = self.video_key
        if video_key is None:
            video_key = self._modality_config["video"].modality_keys[0]

        # Build GR00T observation dict
        images_dict = obs.get("images", {})
        img_array = next(iter(images_dict.values())) if isinstance(images_dict, dict) and images_dict else None

        observation: dict[str, Any] = {
            "video": {},
            "state": {},
            "language": {self._language_key: [[obs.get("task_description", "")]]},
        }

        # Video: (B=1, T=1, H, W, C=3) uint8
        if img_array is not None:
            img = np.asarray(img_array, dtype=np.uint8)
            if img.ndim == 3:
                img = img[np.newaxis, np.newaxis, ...]
            observation["video"][video_key] = img

        # State: fill ALL required keys (zeros if not provided)
        for state_key in self._modality_config["state"].modality_keys:
            dim = self._state_dims.get(state_key, 1)
            observation["state"][state_key] = np.zeros((1, 1, dim), dtype=np.float32)

        action_dict, _ = self._policy.get_action(observation)

        # Concatenate action keys into single array, remove batch dim
        keys = self.action_keys or self._modality_config["action"].modality_keys
        parts = [action_dict[k][0] for k in keys if k in action_dict]
        if parts:
            actions = np.concatenate(parts, axis=-1)
        else:
            actions = np.zeros((1, 7), dtype=np.float32)

        return {"actions": actions}

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        if self._policy is not None:
            self._policy.reset()
        await super().on_episode_start(config, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR00T N1.6 model server (uv script)")
    parser.add_argument("--model_path", default="nvidia/GR00T-N1.6-3B", help="HF model ID or local path")
    parser.add_argument("--embodiment_tag", default="GR1", help="Embodiment tag (e.g. GR1, ROBOCASA_PANDA_OMRON)")
    parser.add_argument("--video_key", default=None, help="Video modality key (auto-detected if omitted)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_size", type=int, default=16)
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

    server = GR00TModelServer(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        video_key=args.video_key,
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
