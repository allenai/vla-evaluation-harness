# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "dexbotic",
#     "torch>=2.0",
#     "transformers>=4.40,<5",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate",
#     "einops",
#     "timm",
#     "sentencepiece",
#     "diffusers",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../../..", editable = true }
# dexbotic = { git = "https://github.com/MilkClouds/dexbotic.git", rev = "42f72859dfe48bb4c30a09ab151a018c2ca0700a" }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve

logger = logging.getLogger(__name__)


def _patch_llm_labels_compat(model: Any) -> None:
    """Patch Qwen2Model.forward to ignore ``labels`` kwarg.

    ``CogACTForCausalLM.forward()`` always passes ``labels`` to the inner LLM,
    but ``Qwen2Model.forward()`` in transformers ≥4.46 no longer accepts it.
    """
    import inspect

    llm = model.model.llm
    if "labels" not in inspect.signature(llm.forward).parameters:
        _orig = llm.forward

        def _fwd_no_labels(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("labels", None)
            return _orig(*args, **kwargs)

        llm.forward = _fwd_no_labels
        logger.debug("Patched %s.forward to drop 'labels' kwarg", type(llm).__name__)


class CogACTModelServer(PredictModelServer):
    """CogACT VLA model server using diffusion-based action head."""

    def __init__(
        self,
        model_path: str,
        cfg_scale: float = 1.5,
        num_ddim_steps: int = 10,
        use_text_template: bool = False,
        *,
        chunk_size: int | None = None,
        chunk_size_map: dict[str, int] | None = None,
        camera_keys: list[str] | None = None,
        action_ensemble: str = "newest",
        image_resolution: int = 224,
        **kwargs: Any,
    ) -> None:
        if chunk_size is not None and chunk_size_map is not None:
            raise ValueError(
                "chunk_size and chunk_size_map are mutually exclusive. "
                "Use --chunk_size for a fixed value, or --chunk_size_map for per-suite values."
            )
        resolved_chunk_size = chunk_size if chunk_size is not None else 12
        super().__init__(
            chunk_size=resolved_chunk_size,
            action_ensemble=action_ensemble,
            **kwargs,
        )
        self.model_path = model_path
        self.cfg_scale = cfg_scale
        self.num_ddim_steps = num_ddim_steps
        self.use_text_template = use_text_template
        self.chunk_size_map = chunk_size_map
        self.camera_keys = camera_keys
        self.image_resolution = image_resolution
        self._model = None
        self._tokenizer = None
        self._norm_stats = None
        self._device = None

    def get_observation_params(self) -> dict[str, Any]:
        return {"image_size": [self.image_resolution, self.image_resolution]}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {"image": IMAGE_RGB, "language": LANGUAGE}
        if self.camera_keys and len(self.camera_keys) > 1:
            for key in self.camera_keys[1:]:
                spec[key] = IMAGE_RGB
        return spec

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from dexbotic.model.cogact.cogact_arch import CogACTForCausalLM
        from transformers import AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CogACT from %s on %s", self.model_path, self._device)

        self._model = CogACTForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        _patch_llm_labels_compat(self._model)

        self._norm_stats = self._load_norm_stats()
        logger.info("Model loaded. Norm stats: %s", self._norm_stats)

    def _load_norm_stats(self) -> dict[str, Any]:
        """Load norm_stats.json from local path or HuggingFace Hub."""
        default = {"min": -1, "max": 1}

        # Try local path first
        local_file = Path(self.model_path) / "norm_stats.json"
        if local_file.exists():
            return self._parse_norm_stats(local_file, default)

        # Try HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(self.model_path, "norm_stats.json")
            return self._parse_norm_stats(Path(path), default)
        except Exception:
            logger.debug("norm_stats.json not found for %s, using defaults", self.model_path)
            return default

    @staticmethod
    def _parse_norm_stats(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        with open(path) as f:
            stats = json.load(f)
        if "norm_stats" in stats:
            stats = stats["norm_stats"]
        return stats.get("default", default)

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Resolve per-suite chunk_size from ``chunk_size_map`` if set.

        Stores the resolved chunk_size in ``_session_chunk_sizes`` instead
        of mutating ``self.chunk_size`` to avoid race conditions when
        multiple sessions run concurrently.
        """
        if self.chunk_size_map is not None:
            suite = config.get("task", {}).get("suite", "")
            if suite not in self.chunk_size_map:
                raise ValueError(
                    f"Suite {suite!r} not found in chunk_size_map "
                    f"{set(self.chunk_size_map)}. Use --chunk_size for a "
                    f"fixed value, or add the suite to --chunk_size_map."
                )
            cs = self.chunk_size_map[suite]
            self._session_chunk_sizes[ctx.session_id] = cs
            logger.info("Suite %r → chunk_size=%d (session=%s)", suite, cs, ctx.session_id)
        await super().on_episode_start(config, ctx)

    def _obs_to_pil_images(self, obs: Observation) -> list[Any]:
        """Extract images from an observation and convert to PIL RGB.

        When ``camera_keys`` is set, extracts those specific keys in order.
        Otherwise, extracts only the first image (backward compatible).
        """
        from PIL import Image as PILImage

        images_dict = obs.get("images", {})
        if not isinstance(images_dict, dict):
            images_dict = {"image": images_dict}

        if self.camera_keys is not None:
            arrays = [images_dict[k] for k in self.camera_keys]
        else:
            arrays = [next(iter(images_dict.values()))]

        return [PILImage.fromarray(img).convert("RGB") if isinstance(img, np.ndarray) else img for img in arrays]

    @staticmethod
    def _convert_actions(raw_actions: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        """Convert cumulative-delta actions to absolute joint positions.

        Used by RoboTwin where the model outputs cumulative deltas for arm
        joints and absolute values for grippers.

        Args:
            raw_actions: (chunk_size, >=14) raw model output after denorm.
            joint_state: (14,) current joint state.

        Returns:
            (chunk_size, 14) absolute joint positions.
        """
        out = np.zeros((len(raw_actions), 14), dtype=np.float64)
        for i, raw in enumerate(raw_actions):
            out[i, 0:6] = joint_state[0:6] + raw[0:6]
            out[i, 6] = raw[6]
            out[i, 7:13] = joint_state[7:13] + raw[7:13]
            out[i, 13] = raw[13]
        return out.astype(np.float32)

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        import torch
        from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from dexbotic.tokenization import conversation as conversation_lib
        from dexbotic.tokenization.tokenization import tokenizer_image_token

        self._load_model()
        assert self._model is not None
        assert self._tokenizer is not None

        pil_images = self._obs_to_pil_images(obs)
        image_tensor = self._model.process_images(pil_images).to(dtype=self._model.dtype)
        if len(pil_images) > 1:
            image_tensor = image_tensor.unsqueeze(0)

        text = obs.get("task_description", "")
        if self.use_text_template:
            text = f"What action should the robot take to {text}?"

        conv = conversation_lib.conv_templates[self._model.config.chat_template].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], " ")

        input_ids = (
            tokenizer_image_token(conv.get_prompt(), self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self._model.device)
        )

        with torch.inference_mode():
            actions = self._model.inference_action(
                input_ids,
                image_tensor,
                {"cfg_scale": self.cfg_scale, "num_ddim_steps": self.num_ddim_steps, "action_norms": self._norm_stats},
            )
        raw_actions = np.array(actions, dtype=np.float32)

        joint_state = obs.get("joint_state")
        if joint_state is not None:
            raw_actions = self._convert_actions(raw_actions, np.asarray(joint_state))
        return {"actions": raw_actions}

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[dict[str, Any]]:
        import torch
        from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from dexbotic.tokenization import conversation as conversation_lib
        from dexbotic.tokenization.tokenization import tokenizer_image_token

        self._load_model()
        assert self._model is not None
        assert self._tokenizer is not None
        B = len(obs_batch)

        # --- Preprocess all observations ---
        all_pil_images = []
        prompts = []
        for obs in obs_batch:
            all_pil_images.extend(self._obs_to_pil_images(obs))

            text = obs.get("task_description", "")
            if self.use_text_template:
                text = f"What action should the robot take to {text}?"
            conv = conversation_lib.conv_templates[self._model.config.chat_template].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
            conv.append_message(conv.roles[1], " ")
            prompts.append(conv.get_prompt())

        # Batch image processing
        num_views = len(all_pil_images) // B
        if num_views == 1:
            image_tensor = self._model.process_images(all_pil_images).to(dtype=self._model.dtype)
        else:
            per_sample_tensors = []
            for i in range(B):
                t = self._model.process_images(all_pil_images[i * num_views : (i + 1) * num_views]).to(
                    dtype=self._model.dtype
                )
                per_sample_tensors.append(t.unsqueeze(0))
            image_tensor = torch.cat(per_sample_tensors, dim=0)

        # Tokenize and pad input_ids
        all_ids = [tokenizer_image_token(p, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for p in prompts]
        max_len = max(ids.shape[0] for ids in all_ids)
        pad_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        padded = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        for i, ids in enumerate(all_ids):
            padded[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1
        input_ids = padded.to(self._model.device)
        attention_mask = attention_mask.to(self._model.device)

        # --- Batched forward pass ---
        # Call model components directly to access the transformed attention_mask
        # (needed to locate last real token in each padded sequence).
        with torch.inference_mode():
            (
                _input_ids,
                _position_ids,
                _attention_mask,
                _past_kv,
                _inputs_embeds,
                _labels,
                _cache_pos,
            ) = self._model.model._prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, None, None, image_tensor
            )

            outputs = self._model.model.llm(
                input_ids=_input_ids,
                position_ids=_position_ids,
                attention_mask=_attention_mask,
                past_key_values=_past_kv,
                inputs_embeds=_inputs_embeds,
                use_cache=True,
                output_hidden_states=True,
            )

            last_hidden_state = outputs.hidden_states[-1]

            # Extract cognition features (last real token per sample)
            if _attention_mask is not None:
                cum = _attention_mask.cumsum(dim=1)
                last_idx = (cum == cum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
                expanded = last_idx.unsqueeze(-1).expand(-1, last_hidden_state.size(-1))
                cognition_features = last_hidden_state.gather(1, expanded.unsqueeze(1))
            else:
                cognition_features = last_hidden_state[:, -1:, :]  # [B, 1, D]

            # Diffusion sampling (same logic as inference_action but for full batch)
            noise = torch.randn(
                B,
                self._model.config.chunk_size,
                self._model.config.action_dim,
                device=cognition_features.device,
                dtype=cognition_features.dtype,
            )

            if self.cfg_scale > 1.0:
                noise = torch.cat([noise, noise], 0)
                uncondition = self._model.model.action_head.net.z_embedder.uncondition
                uncondition = uncondition.unsqueeze(0).expand(B, 1, -1)
                z = torch.cat([cognition_features, uncondition], 0)
                model_kwargs = dict(z=z, cfg_scale=self.cfg_scale)
                sample_fn = self._model.model.action_head.net.forward_with_cfg
            else:
                model_kwargs = dict(z=cognition_features)
                sample_fn = self._model.model.action_head.net.forward

            if self._model.model.action_head.ddim_diffusion is None:
                self._model.model.action_head.create_ddim(ddim_step=self.num_ddim_steps)

            samples = self._model.model.action_head.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=cognition_features.device,
                eta=0.0,
            )
            if self.cfg_scale > 1.0:
                samples, _ = samples.chunk(2, dim=0)

        # Denormalize each sample and optionally convert deltas to absolute
        results = []
        for i in range(B):
            raw_actions = np.array(
                self._model._denorm(samples[i].cpu().numpy(), self._norm_stats),
                dtype=np.float32,
            )
            joint_state = obs_batch[i].get("joint_state")
            if joint_state is not None:
                raw_actions = self._convert_actions(raw_actions, np.asarray(joint_state))
            results.append({"actions": raw_actions})
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CogACT model server (uv script)")
    parser.add_argument("--model_path", required=True, help="Path to CogACT checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    chunk_group = parser.add_mutually_exclusive_group(required=True)
    chunk_group.add_argument("--chunk_size", type=int, help="Fixed replan step (actions per inference)")
    chunk_group.add_argument(
        "--chunk_size_map",
        type=json.loads,
        help='JSON suite→chunk_size map, e.g. \'{"libero_spatial":12,"libero_object":16}\'',
    )
    parser.add_argument("--action_ensemble", default="newest")
    parser.add_argument("--use_text_template", action="store_true")
    parser.add_argument(
        "--camera_keys",
        type=json.loads,
        default=None,
        help='JSON list of camera keys, e.g. \'["head_camera", "left_camera", "right_camera"]\'',
    )
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
        cfg_scale=args.cfg_scale,
        num_ddim_steps=args.num_ddim_steps,
        use_text_template=args.use_text_template,
        chunk_size=args.chunk_size,
        chunk_size_map=args.chunk_size_map,
        camera_keys=args.camera_keys,
        **kwargs,
    )
    if args.chunk_size_map:
        logger.info("Per-suite chunk_size: %s", server.chunk_size_map)

    logger.info("Pre-loading model...")
    server._load_model()
    mode = (
        f"batch (max_batch={args.max_batch_size}, wait={args.max_wait_time}s)" if args.max_batch_size > 1 else "single"
    )
    logger.info("Model ready [%s], starting server on ws://%s:%d", mode, args.host, args.port)
    serve(server, host=args.host, port=args.port)
