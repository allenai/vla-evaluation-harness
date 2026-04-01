# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch>=2.2",
#     "transformers==4.44.0",
#     "tiktoken==0.6.0",
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "scipy>=1.11",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""CoW-VLA model server.

Loads released CoW-VLA checkpoints from Hugging Face while reusing the pinned
upstream reference code for the custom Emu3/CoW-VLA model classes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
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
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

_DISABLED_STRINGS = {"", "none", "null"}
_COWVLA_REPO_URL = "https://github.com/fx-hit/CoWVLA.git"
_COWVLA_REPO_REV = "fd3b514a3f56eded14228b700f656892ff00c3e3"


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    return None if value.strip().lower() in _DISABLED_STRINGS else value


def _select_image(images: Any, preferred_key: str | None, fallback_index: int = 0) -> np.ndarray:
    """Select one image from an observation and return it as uint8 ndarray."""
    if isinstance(images, dict):
        if preferred_key and preferred_key in images:
            return np.asarray(images[preferred_key], dtype=np.uint8)
        values = list(images.values())
        if not values:
            raise ValueError("Observation contained an empty images dict")
        idx = fallback_index if fallback_index < len(values) else 0
        return np.asarray(values[idx], dtype=np.uint8)
    return np.asarray(images, dtype=np.uint8)


def _unnormalize_actions(actions: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return 0.5 * (actions + 1.0) * (q99 - q01) + q01


def _build_inputs_with_latent_query(model: Any, input_ids: Any, attention_mask: Any) -> tuple[Any, Any]:
    """Insert CoW-VLA's learned latent-action query after the first image block."""
    import torch

    device = input_ids.device
    batch_size = input_ids.shape[0]
    word_embeds = model.model.embed_tokens(input_ids)
    latent_query_embeds = model.latent_action_query.expand(batch_size, -1, -1)

    new_inputs_embeds_list = []
    new_attention_mask_list = []
    for i in range(batch_size):
        img_end_indices = (input_ids[i] == model.img_end_token_id).nonzero(as_tuple=True)[0]
        insert_idx = img_end_indices[0].item() + 1 if len(img_end_indices) > 0 else input_ids.shape[1]

        pre_embeds = word_embeds[i, :insert_idx]
        post_embeds = word_embeds[i, insert_idx:]
        pre_mask = attention_mask[i, :insert_idx]
        post_mask = attention_mask[i, insert_idx:]
        query_mask = torch.ones((1,), dtype=attention_mask.dtype, device=device)

        new_inputs_embeds_list.append(torch.cat([pre_embeds, latent_query_embeds[i], post_embeds], dim=0))
        new_attention_mask_list.append(torch.cat([pre_mask, query_mask, post_mask], dim=0))

    return torch.stack(new_inputs_embeds_list, dim=0), torch.stack(new_attention_mask_list, dim=0)


@dataclass
class _SessionBuffers:
    image_inputs: deque[Any]
    action_tokens: deque[Any]


@dataclass
class _PreparedRequest:
    ctx: SessionContext
    buffers: _SessionBuffers
    final_inputs: Any
    prompt_len: int


class CoWVLAModelServer(PredictModelServer):
    """CoW-VLA model server using the upstream Emu3-based reference implementation."""

    def __init__(
        self,
        model_hub: str = "hitfx/CoWVLA",
        model_subfolder: str | None = "CoWVLA_LIBERO_BS128_8K",
        vq_hub: str = "BAAI/Emu3-Stage1",
        vision_hub: str = "BAAI/Emu3-VisionTokenizer",
        reference_repo_dir: str | None = None,
        repo_url: str = _COWVLA_REPO_URL,
        repo_rev: str = _COWVLA_REPO_REV,
        repo_cache_dir: str | None = None,
        action_tokenizer_subdir: str = "pretrain/fast",
        normalizer_subdir: str | None = "configs/normalizer_libero",
        normalizer_key: str | None = "libero",
        image_key: str | None = None,
        wrist_image_key: str | None = "wrist",
        image_size: int = 200,
        window_size: int = 2,
        context_frames: int = 1,
        predict_frames: int = 1,
        action_dim: int = 7,
        eoa_token_id: int = 151845,
        max_new_tokens: int = 100,
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_hub = model_hub
        self.model_subfolder = _normalize_optional_str(model_subfolder)
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.reference_repo_dir = _normalize_optional_str(reference_repo_dir)
        self.repo_url = repo_url
        self.repo_rev = repo_rev
        self.repo_cache_dir = _normalize_optional_str(repo_cache_dir)
        self.action_tokenizer_subdir = action_tokenizer_subdir
        self.normalizer_subdir = _normalize_optional_str(normalizer_subdir)
        self.normalizer_key = _normalize_optional_str(normalizer_key)
        self.image_key = _normalize_optional_str(image_key)
        self.wrist_image_key = _normalize_optional_str(wrist_image_key)
        self.image_size = image_size
        self.window_size = window_size
        self.context_frames = context_frames
        self.predict_frames = predict_frames
        self.action_dim = action_dim
        self.eoa_token_id = eoa_token_id
        self.max_new_tokens = max_new_tokens

        self._device = None
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._image_processor = None
        self._image_tokenizer = None
        self._action_tokenizer = None
        self._generation_config = None
        self._norm_stats: dict[str, Any] | None = None
        self._use_latent_action = False
        self._reference_repo_path: Path | None = None
        self._session_buffers: dict[str, _SessionBuffers] = {}

    def get_observation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.wrist_image_key is not None:
            params["send_wrist_image"] = True
        return params

    def _make_session_buffers(self) -> _SessionBuffers:
        action_history = max(self.window_size - 1, 1)
        return _SessionBuffers(
            image_inputs=deque(maxlen=max(self.window_size, 1)),
            action_tokens=deque(maxlen=action_history),
        )

    def _resolve_reference_repo(self) -> Path:
        if self._reference_repo_path is not None:
            return self._reference_repo_path

        if self.reference_repo_dir is not None:
            path = Path(self.reference_repo_dir).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"reference_repo_dir does not exist: {path}")
            self._reference_repo_path = path
            return path

        git = shutil.which("git")
        if git is None:
            raise RuntimeError("git is required to fetch the CoW-VLA reference repo")

        cache_root = Path(self.repo_cache_dir or "~/.cache/vla-eval/reference-repos").expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / f"cowvla-{self.repo_rev[:12]}"
        if target.exists():
            self._reference_repo_path = target
            return target

        tmp_dir = Path(tempfile.mkdtemp(prefix="cowvla-", dir=cache_root))
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
        repo_root = self._resolve_reference_repo()
        emu3_root = repo_root / "reference" / "Emu3"
        if not emu3_root.exists():
            raise FileNotFoundError(f"CoW-VLA reference checkout missing Emu3 code: {emu3_root}")
        for path in (repo_root, emu3_root):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        return repo_root

    def _load_normalizer(self, repo_root: Path) -> dict[str, Any] | None:
        if self.normalizer_subdir is None or self.normalizer_key is None:
            return None
        stats_path = repo_root / self.normalizer_subdir / "norm_stats.json"
        with open(stats_path) as f:
            data = json.load(f)
        norm_stats = data.get("norm_stats", data)
        if self.normalizer_key not in norm_stats:
            raise KeyError(f"Normalizer key {self.normalizer_key!r} not found in {stats_path}")
        return norm_stats[self.normalizer_key]

    def _load_model(self) -> None:
        if self._model is not None:
            return

        repo_root = self._ensure_reference_imports()

        import torch
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor, GenerationConfig

        from emu3.mllm import Emu3MoE, Emu3MoEConfig, Emu3Processor, Emu3Tokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32

        cfg_kwargs = {"subfolder": self.model_subfolder} if self.model_subfolder is not None else {}
        model_config = Emu3MoEConfig.from_pretrained(self.model_hub, **cfg_kwargs)

        model_kwargs: dict[str, Any] = {
            "config": model_config,
            "torch_dtype": model_dtype,
            "trust_remote_code": True,
        }
        if self._device.type == "cuda":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                logger.info("flash_attn not available; falling back to default attention backend")
            else:
                model_kwargs["attn_implementation"] = "flash_attention_2"

        logger.info(
            "Loading CoW-VLA from %s%s on %s",
            self.model_hub,
            f"/{self.model_subfolder}" if self.model_subfolder else "",
            self._device,
        )
        self._model = Emu3MoE.from_pretrained(self.model_hub, **cfg_kwargs, **model_kwargs).to(self._device).eval()
        self._tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            model_max_length=self._model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self._image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self._image_processor.min_pixels = 80 * 80
        self._image_tokenizer = (
            AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(self._device).eval()
        )
        self._processor = Emu3Processor(self._image_processor, self._image_tokenizer, self._tokenizer)

        action_tokenizer_dir = repo_root / self.action_tokenizer_subdir
        self._action_tokenizer = AutoProcessor.from_pretrained(str(action_tokenizer_dir), trust_remote_code=True)
        self._generation_config = GenerationConfig(
            pad_token_id=self._model.config.pad_token_id,
            bos_token_id=self._model.config.bos_token_id,
            eos_token_id=self.eoa_token_id,
            do_sample=False,
        )
        self._norm_stats = self._load_normalizer(repo_root)
        self._use_latent_action = bool(getattr(model_config, "use_latent_action", False))

        logger.info(
            "CoW-VLA ready (latent_action=%s, chunk_size=%s, wrist_image=%s)",
            self._use_latent_action,
            self.chunk_size,
            self.wrist_image_key is not None,
        )

    def _encode_image(self, image: np.ndarray) -> Any:
        from PIL import Image

        assert self._device is not None
        assert self._image_processor is not None
        assert self._image_tokenizer is not None

        pil_image = Image.fromarray(image).convert("RGB")
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        pixel_values = self._image_processor(pil_image, return_tensors="pt")["pixel_values"].to(self._device)
        return self._image_tokenizer.encode(pixel_values)

    @staticmethod
    def _feature_tensor(batch_feature: Any, key: str) -> Any:
        import torch

        if key in batch_feature:
            return batch_feature[key]
        if key == "attention_mask":
            return torch.ones_like(batch_feature["input_ids"])
        if key == "token_type_ids":
            return torch.zeros_like(batch_feature["input_ids"])
        raise KeyError(key)

    def _prepare_request(self, obs: Observation, ctx: SessionContext, buffers: _SessionBuffers) -> _PreparedRequest:
        import torch
        from transformers.feature_extraction_utils import BatchFeature

        assert self._tokenizer is not None
        assert self._processor is not None

        images = obs.get("images", {})
        image = _select_image(images, self.image_key, fallback_index=0)
        image_code = self._encode_image(image).unsqueeze(1)

        wrist_code = None
        if self.wrist_image_key is not None:
            try:
                wrist = _select_image(images, self.wrist_image_key, fallback_index=1)
            except ValueError:
                wrist = None
            if wrist is not None:
                wrist_code = self._encode_image(wrist).unsqueeze(1)

        task_description = obs.get("task_description", "")
        text_tokens = BatchFeature(
            data={**self._processor.tokenizer([self._tokenizer.bos_token + task_description])},
            tensor_type="pt",
        )
        current_inputs = self._processor.video_process(
            text=task_description,
            video_tokens=image_code,
            gripper_tokens=wrist_code,
            context_frames=self.context_frames,
            frames=self.predict_frames,
            return_tensors="pt",
            mode="VLA_Video",
            padding="longest",
        )

        buffers.image_inputs.append(current_inputs)
        history = list(buffers.image_inputs)
        action_history = list(buffers.action_tokens)

        all_input_ids = [self._feature_tensor(text_tokens, "input_ids")]
        all_token_type_ids = [self._feature_tensor(text_tokens, "token_type_ids")]
        all_attention_mask = [self._feature_tensor(text_tokens, "attention_mask")]

        for idx, image_inputs in enumerate(history):
            img_input_ids = self._feature_tensor(image_inputs, "input_ids")
            img_token_type_ids = self._feature_tensor(image_inputs, "token_type_ids")
            img_attention_mask = self._feature_tensor(image_inputs, "attention_mask")

            if idx < len(action_history):
                action_ids = action_history[idx]
                all_input_ids.extend([img_input_ids, action_ids])
                all_token_type_ids.extend([img_token_type_ids, torch.zeros_like(action_ids)])
                all_attention_mask.extend([img_attention_mask, torch.ones_like(action_ids)])
            else:
                all_input_ids.append(img_input_ids)
                all_token_type_ids.append(img_token_type_ids)
                all_attention_mask.append(img_attention_mask)

        final_inputs = current_inputs.copy()
        final_inputs["input_ids"] = torch.cat(all_input_ids, dim=1)
        final_inputs["token_type_ids"] = torch.cat(all_token_type_ids, dim=1)
        final_inputs["attention_mask"] = torch.cat(all_attention_mask, dim=1)
        prompt_len = int(final_inputs["input_ids"].shape[-1])
        return _PreparedRequest(ctx=ctx, buffers=buffers, final_inputs=final_inputs, prompt_len=prompt_len)

    def _pad_prepared_requests(self, prepared_requests: list[_PreparedRequest]) -> tuple[dict[str, Any], int]:
        import torch
        import torch.nn.functional as F

        assert self._tokenizer is not None

        max_prompt_len = max(req.prompt_len for req in prepared_requests)
        pad_token_id = int(self._tokenizer.pad_token_id)

        input_ids_batch = []
        token_type_ids_batch = []
        attention_mask_batch = []
        for req in prepared_requests:
            pad_len = max_prompt_len - req.prompt_len
            input_ids = req.final_inputs["input_ids"]
            token_type_ids = req.final_inputs["token_type_ids"]
            attention_mask = req.final_inputs["attention_mask"]
            if pad_len > 0:
                # Decoder-only generation must be left-padded so the final
                # non-pad token stays aligned across the batch.
                input_ids = F.pad(input_ids, (pad_len, 0), value=pad_token_id)
                token_type_ids = F.pad(token_type_ids, (pad_len, 0), value=0)
                attention_mask = F.pad(attention_mask, (pad_len, 0), value=0)
            input_ids_batch.append(input_ids)
            token_type_ids_batch.append(token_type_ids)
            attention_mask_batch.append(attention_mask)

        return {
            "input_ids": torch.cat(input_ids_batch, dim=0),
            "token_type_ids": torch.cat(token_type_ids_batch, dim=0),
            "attention_mask": torch.cat(attention_mask_batch, dim=0),
        }, max_prompt_len

    def _trim_generated_token_rows(self, raw_tokens: Any) -> tuple[list[Any], list[list[int]]]:
        assert self._tokenizer is not None

        pad_token_id = int(self._tokenizer.pad_token_id)
        last_token_id = pad_token_id - 1
        trimmed_raw_tokens = []
        processed_token_rows: list[list[int]] = []

        for row in raw_tokens:
            row = row.detach()
            eoa_positions = (row == self.eoa_token_id).nonzero(as_tuple=True)[0]
            if len(eoa_positions) > 0:
                keep_len = int(eoa_positions[0].item()) + 1
            else:
                non_pad_positions = (row != pad_token_id).nonzero(as_tuple=True)[0]
                keep_len = int(non_pad_positions[-1].item()) + 1 if len(non_pad_positions) > 0 else 0

            trimmed = row[:keep_len]
            trimmed_raw_tokens.append(trimmed.unsqueeze(0).cpu())

            if keep_len > 0 and int(trimmed[-1].item()) == self.eoa_token_id:
                decode_tokens = trimmed[:-1]
            else:
                decode_tokens = trimmed
            processed_token_rows.append((last_token_id - decode_tokens).detach().cpu().tolist())

        return trimmed_raw_tokens, processed_token_rows

    def _predict_prepared_requests(self, prepared_requests: list[_PreparedRequest]) -> list[Action]:
        import torch
        from transformers import LogitsProcessor

        assert self._action_tokenizer is not None
        assert self._generation_config is not None
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._device is not None
        if not prepared_requests:
            return []

        class _ActionIDConstraintLogitsProcessor(LogitsProcessor):
            def __init__(self, allowed_token_ids: list[int]) -> None:
                self.allowed_token_ids = allowed_token_ids

            def __call__(self, input_ids: Any, scores: Any) -> Any:
                mask = torch.zeros_like(scores, dtype=torch.bool)
                if mask.ndim == 1:
                    mask[self.allowed_token_ids] = True
                else:
                    mask[:, self.allowed_token_ids] = True
                scores[~mask] = -float("inf")
                return scores

        last_token_id = self._tokenizer.pad_token_id - 1
        allowed_token_ids = list(range(last_token_id - self._action_tokenizer.vocab_size, last_token_id + 1))
        allowed_token_ids.append(self.eoa_token_id)
        logits_processor = [_ActionIDConstraintLogitsProcessor(allowed_token_ids)]

        batch_inputs, max_prompt_len = self._pad_prepared_requests(prepared_requests)
        input_ids = batch_inputs["input_ids"].to(self._device)
        attention_mask = batch_inputs["attention_mask"].to(self._device)

        with torch.inference_mode():
            if self._use_latent_action:
                inputs_embeds, attention_mask = _build_inputs_with_latent_query(self._model, input_ids, attention_mask)
                generated = self._model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    generation_config=self._generation_config,
                    max_new_tokens=self.max_new_tokens,
                    logits_processor=logits_processor,
                )
                raw_tokens = generated
            else:
                generated = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self._generation_config,
                    max_new_tokens=self.max_new_tokens,
                    logits_processor=logits_processor,
                )
                raw_tokens = generated[:, max_prompt_len:]
        trimmed_raw_tokens, processed_token_rows = self._trim_generated_token_rows(raw_tokens)
        action_outputs = self._action_tokenizer.decode(
            processed_token_rows,
            time_horizon=self.chunk_size or 1,
            action_dim=self.action_dim,
        )
        outputs: list[Action] = []
        for idx, req in enumerate(prepared_requests):
            if self.window_size > 1 and trimmed_raw_tokens[idx].numel() > 0:
                req.buffers.action_tokens.append(trimmed_raw_tokens[idx])
            actions = np.asarray(action_outputs[idx], dtype=np.float32)
            if self._norm_stats is not None:
                actions = _unnormalize_actions(actions, self._norm_stats)
            outputs.append({"actions": np.asarray(actions, dtype=np.float32)})
        return outputs

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        self._session_buffers[ctx.session_id] = self._make_session_buffers()
        await super().on_episode_start(config, ctx)

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        self._session_buffers.pop(ctx.session_id, None)
        await super().on_episode_end(result, ctx)

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        return self.predict_batch([obs], [ctx])[0]

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        self._load_model()
        outputs: list[Action] = []
        prepared_requests: list[_PreparedRequest] = []
        seen_session_ids: set[str] = set()
        for obs, ctx in zip(obs_batch, ctx_batch, strict=True):
            if prepared_requests and ctx.session_id in seen_session_ids:
                outputs.extend(self._predict_prepared_requests(prepared_requests))
                prepared_requests = []
                seen_session_ids = set()
            buffers = self._session_buffers.setdefault(ctx.session_id, self._make_session_buffers())
            prepared_requests.append(self._prepare_request(obs, ctx, buffers))
            seen_session_ids.add(ctx.session_id)
        outputs.extend(self._predict_prepared_requests(prepared_requests))
        return outputs


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(CoWVLAModelServer)
