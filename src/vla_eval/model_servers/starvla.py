# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "starvla",
#     "torch>=2.0",
#     "torchvision>=0.17",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "kernels>=0.11.0",
#     "qwen-vl-utils",
#     "omegaconf",
#     "rich",
#     "diffusers",
#     "timm",
#     "einops",
#     "scipy",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../.." }
# starvla = { git = "https://github.com/starVLA/starVLA.git", rev = "eaa51c4c2f4012d42f1036ee318d41942e8f97a3" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""starVLA model server — supports all Qwen* frameworks.

Supported frameworks (auto-detected from checkpoint config):
  - ``QwenGR00T``: Qwen2.5-VL + Flow-matching action head (GR00T-style)
  - ``QwenOFT``:   Qwen2.5-VL + MLP action head, parallel continuous decoding
  - ``QwenPI``:    Qwen2.5-VL + Layerwise Flow-matching DiT (π₀-style)
  - ``QwenFast``:  Qwen2.5-VL + Fast tokenizer, autoregressive discrete actions
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.types import Action, Observation

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


class StarVLAModelServer(PredictModelServer):
    """Generic starVLA model server for all Qwen* frameworks."""

    chunk_size = 1
    action_ensemble: str = "newest"

    def __init__(self, checkpoint: str, *, unnorm_key: str | None = None, use_bf16: bool = False) -> None:
        super().__init__()
        self.checkpoint = checkpoint
        self.unnorm_key = unnorm_key
        self.use_bf16 = use_bf16
        self._model = None

    @staticmethod
    def _resolve_checkpoint(checkpoint: str) -> str:
        """Resolve *checkpoint* to a local ``.pt`` / ``.safetensors`` path.

        If *checkpoint* is already a local file it is returned as-is.
        Otherwise it is treated as a HuggingFace model ID and downloaded
        via ``huggingface_hub.snapshot_download``.  The first checkpoint
        file found under the ``checkpoints/`` sub-directory is returned.
        """
        path = Path(checkpoint)
        if path.is_file() and path.suffix in (".pt", ".safetensors"):
            return str(path)

        # Treat as HuggingFace model ID — download the repo
        from huggingface_hub import snapshot_download

        logger.info("Downloading model from HuggingFace Hub: %s", checkpoint)
        local_dir = Path(snapshot_download(checkpoint))
        ckpt_dir = local_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(
                f"Downloaded repo {checkpoint} has no 'checkpoints/' directory "
                f"(contents: {[p.name for p in local_dir.iterdir()]})"
            )
        candidates = sorted(
            [p for p in ckpt_dir.iterdir() if p.suffix in (".pt", ".safetensors")],
            key=lambda p: p.name,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .pt / .safetensors files in {ckpt_dir} (contents: {[p.name for p in ckpt_dir.iterdir()]})"
            )
        return str(candidates[-1])  # latest by name

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from starVLA.model.framework.base_framework import baseframework

        ckpt_path = self._resolve_checkpoint(self.checkpoint)
        logger.info("Loading starVLA model from checkpoint: %s", ckpt_path)

        # ------------------------------------------------------------------
        # Monkey-patches to work around upstream starVLA incompatibilities.
        # All patches are collected in *_patches* and restored in the
        # ``finally`` block so the global state is never left dirty.
        # ------------------------------------------------------------------
        _patches: list[tuple] = []  # (obj, attr_name, original_value)

        # 1) flash_attention_2 → kernels-community/flash-attn2
        #    starVLA hardcodes attn_implementation="flash_attention_2" which
        #    requires a manually-compiled flash-attn wheel.  The ``kernels``
        #    package provides a pre-compiled, env-compatible drop-in.
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

        def _patch_from_pretrained(cls_to_patch: Any) -> None:
            orig = cls_to_patch.from_pretrained.__func__

            @classmethod
            def _patched(cls, *args, **kwargs):
                if kwargs.get("attn_implementation") == "flash_attention_2":
                    kwargs["attn_implementation"] = "kernels-community/flash-attn2"
                return orig(cls, *args, **kwargs)

            _patches.append((cls_to_patch, "from_pretrained", classmethod(orig)))
            cls_to_patch.from_pretrained = _patched

        _patch_from_pretrained(Qwen2_5_VLForConditionalGeneration)
        _patch_from_pretrained(Qwen3VLForConditionalGeneration)

        # 2) Legacy framework name aliases + broken base_vlm paths
        #    Some released checkpoints use old framework names (e.g. "QwenFM")
        #    that were later renamed (to "QwenGR00T").  Some also embed local
        #    training paths for base_vlm that don't exist on other machines.
        #    Patch in both the __init__ module and base_framework (which has
        #    its own import).
        import starVLA.model.framework as _fw_mod
        import starVLA.model.framework.base_framework as _bf_mod

        _orig_build = _fw_mod.build_framework
        _ALIASES = {"QwenFM": "QwenGR00T"}
        _VLM_FALLBACK = "starVLA/Qwen2.5-VL-3B-Instruct-Action"
        # Known local basenames → HuggingFace model IDs
        _VLM_BASENAME_MAP = {
            "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-3B-Action": "starVLA/Qwen2.5-VL-3B-Instruct-Action",
            "Qwen2.5-VL-3B-Instruct-Action": "starVLA/Qwen2.5-VL-3B-Instruct-Action",
            "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
        }

        def _resolve_base_vlm(path_str: str) -> str:
            """Resolve a broken local base_vlm path to a HuggingFace model ID."""
            basename = Path(path_str).name
            if basename in _VLM_BASENAME_MAP:
                resolved = _VLM_BASENAME_MAP[basename]
                logger.info("Resolved base_vlm %r → %s", path_str, resolved)
                return resolved
            logger.warning("base_vlm path %r not found, falling back to %s", path_str, _VLM_FALLBACK)
            return _VLM_FALLBACK

        def _aliased_build(cfg):
            fid = getattr(cfg.framework, "name", None) or getattr(cfg.framework, "framework_py", None)
            if fid in _ALIASES:
                cfg.framework.name = _ALIASES[fid]
                if hasattr(cfg.framework, "framework_py"):
                    cfg.framework.framework_py = _ALIASES[fid]
            # Fix broken local base_vlm paths from training configs.
            # Valid HF repo IDs have the form "org/repo" (at most one '/').
            # Paths with 2+ '/' or starting with './' or '/' are local paths.
            base_vlm = getattr(cfg.framework.qwenvl, "base_vlm", "")
            if base_vlm and (base_vlm.startswith("./") or base_vlm.startswith("/") or base_vlm.count("/") >= 2):
                if not Path(base_vlm).exists():
                    cfg.framework.qwenvl.base_vlm = _resolve_base_vlm(base_vlm)
            return _orig_build(cfg)

        _patches.append((_fw_mod, "build_framework", _orig_build))
        _fw_mod.build_framework = _aliased_build
        _patches.append((_bf_mod, "build_framework", _orig_build))
        _bf_mod.build_framework = _aliased_build

        # 3) QwenPI checkpoint compat — fix DiT num_layers & MLP hidden_dim
        #    QwenPI.__init__ hardcodes num_vl_layers=36 (VLM depth) which is
        #    propagated to DiTConfig["num_layers"], but released checkpoints
        #    use the value from diffusion_model_cfg.num_layers (e.g. 16).
        #    The MLP state_encoder/action_decoder default to hidden_dim=1024
        #    but checkpoints expect action_hidden_dim (e.g. 2048).
        import starVLA.model.framework.QwenPI as _qpi_mod
        from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
            MLP as _MLP,
        )
        from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
            get_action_model as _orig_gam,
        )

        def _compat_get_action_model(config=None):
            assert config is not None
            # Fix num_vl_layers to match checkpoint's DiT depth
            dit_cfg = config.framework.action_model.diffusion_model_cfg
            dit_num_layers = getattr(dit_cfg, "num_layers", None)
            if dit_num_layers is not None:
                config.framework.qwenvl.num_vl_layers = dit_num_layers

            model = _orig_gam(config=config)

            # Fix MLP hidden_dim to match checkpoint's action_hidden_dim
            ahd = getattr(config.framework.action_model, "action_hidden_dim", None)
            if ahd and ahd != 1024:
                if getattr(config.framework.action_model, "state_dim", None):
                    model.state_encoder = _MLP(
                        input_dim=config.framework.action_model.state_dim,
                        hidden_dim=ahd,
                        output_dim=model.input_embedding_dim,
                    )
                model.action_decoder = _MLP(
                    input_dim=model.input_embedding_dim,
                    hidden_dim=ahd,
                    output_dim=model.action_dim,
                )
            return model

        _patches.append((_qpi_mod, "get_action_model", _orig_gam))
        _qpi_mod.get_action_model = _compat_get_action_model

        # 4) QwenFast — fix hardcoded local FAST tokenizer path
        #    fast_ActionHeader.py defaults to "playground/Pretrained_models/fast"
        #    but the actual HF repo is "physical-intelligence/fast".
        import starVLA.model.framework.QwenFast as _qfast_mod
        from starVLA.model.modules.action_model.fast_ActionHeader import (
            Fast_Action_Tokenizer as _FAT,
        )
        from starVLA.model.modules.action_model.fast_ActionHeader import (
            get_action_model as _orig_fast_gam,
        )

        def _patched_fast_gam(config=None):
            return _FAT(fast_tokenizer_name="physical-intelligence/fast")

        _patches.append((_qfast_mod, "get_action_model", _orig_fast_gam))
        _qfast_mod.get_action_model = _patched_fast_gam

        try:
            self._model = baseframework.from_pretrained(ckpt_path)
        finally:
            for obj, attr, orig in reversed(_patches):
                setattr(obj, attr, orig)

        if self.use_bf16:
            self._model = self._model.to(torch.bfloat16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device).eval()

        # Resolve unnorm_key and cache action stats for unnormalization.
        # get_action_stats() is mis-decorated as @classmethod in starVLA,
        # so we access norm_stats on the instance directly.
        norm_stats = self._model.norm_stats
        unnorm_key = self.unnorm_key
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    f"Model trained on multiple datasets, pass unnorm_key from: {list(norm_stats.keys())}"
                )
            unnorm_key = next(iter(norm_stats))
        if unnorm_key not in norm_stats:
            raise ValueError(f"unnorm_key={unnorm_key!r} not found, available: {list(norm_stats.keys())}")
        self._action_stats = norm_stats[unnorm_key]["action"]
        logger.info("Model loaded on %s (unnorm_key=%s)", device, unnorm_key)

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        from PIL import Image as PILImage

        self._load_model()
        assert self._model is not None

        # Convert harness observation format to starVLA format
        def _to_pil(img: Any) -> PILImage.Image:
            if isinstance(img, np.ndarray):
                return PILImage.fromarray(img).convert("RGB")
            return img

        images_source = obs.get("images", {})
        if isinstance(images_source, dict):
            pil_images = [_to_pil(v) for v in images_source.values()]
        else:
            pil_images = [_to_pil(images_source)]

        task_description = obs.get("task_description", "")

        example = {
            "image": pil_images,
            "lang": task_description,
        }

        # Add state if present
        state = obs.get("state")
        if state is not None:
            if isinstance(state, np.ndarray):
                example["state"] = state.reshape(1, -1)
            else:
                example["state"] = np.array(state).reshape(1, -1)

        result = self._model.predict_action([example])
        actions = result["normalized_actions"]  # [B, T, action_dim]

        # First batch item
        actions = np.asarray(actions[0])

        # Unnormalize: map [-1, 1] back to original action space using q01/q99 stats
        from starVLA.model.framework.base_framework import baseframework

        actions = baseframework.unnormalize_actions(actions, self._action_stats)

        return {"actions": actions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="starVLA model server (uv script)")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="HuggingFace model ID (e.g. StarVLA/Qwen-PI-Bridge-RT-1) or local path to .pt file",
    )
    parser.add_argument(
        "--unnorm_key", default=None, help="Dataset key for action unnormalization (auto-detected if single dataset)"
    )
    parser.add_argument("--chunk_size", type=int, default=1, help="Action chunk size (replan steps)")
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    server = StarVLAModelServer(args.checkpoint, unnorm_key=args.unnorm_key, use_bf16=args.use_bf16)
    server.chunk_size = args.chunk_size
    server.action_ensemble = args.action_ensemble

    logger.info("Pre-loading model...")
    server._load_model()
    logger.info("Model ready, starting server on ws://%s:%d", args.host, args.port)
    serve(server, host=args.host, port=args.port)
