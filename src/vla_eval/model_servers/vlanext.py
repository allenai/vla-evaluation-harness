# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "transformers==5.5.0",
#     "diffusers==0.36.0",
#     "tensorflow-cpu==2.18.0",  # the official eval's JPEG round trip + lanczos3 resize
#     "pillow>=9.0",
#     "numpy>=1.24",
#     "accelerate>=1.1.1",
#     "huggingface-hub",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-10-31T00:00:00Z"
# ///
"""VLANeXt model server (DravenALG/VLANeXt, the VLANeXt-Family codebase).

Upstream is not pip-installable (non-SPDX license string), so it is cloned at ``_VLANEXT_REV`` on first use; set
``VLANEXT_ROOT`` to use a local clone instead.  Model loading and inference call the upstream LIBERO eval client
(``src/evaluation/libero_bench/VLANeXt_utils.py``); the image preprocessing mirrors its ``libero_utils.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vla_eval import columns
from vla_eval.model_servers.base import ModelServer, Row, Rows
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ROTATION_AA,
    STATE_EEF_POS_AA_GRIP,
    DimSpec,
)
from vla_eval.types import Action, Batch, EpisodeResult, Observation

logger = logging.getLogger(__name__)

_VLANEXT_REPO = "https://github.com/DravenALG/VLANeXt.git"
_VLANEXT_REV = "d911e6b7b5d5615540faee8365e88da9397341b9"


def _ensure_vlanext() -> None:
    """Make upstream's ``src`` package importable: ``VLANEXT_ROOT`` as-is, else a clone at ``_VLANEXT_REV``."""
    from vla_eval.dirs import assets_cache, ensure_git_clone

    user_root = os.environ.get("VLANEXT_ROOT")
    if user_root:
        if not os.path.isdir(os.path.join(user_root, "src", "evaluation", "libero_bench")):
            raise RuntimeError(
                f"VLANEXT_ROOT={user_root} is not a VLANeXt-Family clone (missing src/evaluation/libero_bench). "
                f"Unset it to auto-clone into {assets_cache()}."
            )
        root = user_root
    else:  # one directory per revision, so a clone at another revision is never reused
        name = f"vlanext-{_VLANEXT_REV[:12]}"
        root = str(ensure_git_clone(name=name, repo=_VLANEXT_REPO, rev=_VLANEXT_REV, shallow=False))
    if root not in sys.path:
        sys.path.insert(0, root)


# Action min/max (first 6 dims) from upstream src/datasets/libero_act.py, which imports tensorflow_datasets.
LIBERO_ACTION_STATS: dict[str, dict[str, tuple[list[float], list[float]]]] = {
    "original": {
        "spatial": (
            [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475],
            [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375],
        ),
        "object": (
            [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943],
            [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174],
        ),
        "goal": (
            [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
        "10": (
            [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574],
            [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375],
        ),
        "mixed": (
            [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.3675000071525574],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
    },
    "legacy": {
        "spatial": (
            [-0.9375, -0.9375, -0.9375, -0.18857142329216003, -0.3675000071525574, -0.36000001430511475],
            [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375],
        ),
        "object": (
            [-0.8839285969734192, -0.9375, -0.9375, -0.15000000596046448, -0.29035714268684387, -0.32892856001853943],
            [0.9375, 0.8919642567634583, 0.9375, 0.17678570747375488, 0.35035714507102966, 0.1810714304447174],
        ),
        "goal": (
            [-0.9375, -0.9375, -0.9375, -0.24214285612106323, -0.375, -0.2871428430080414],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
        "10": (
            [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3642857074737549],
            [0.9375, 0.9375, 0.9375, 0.32892856001853943, 0.36964285373687744, 0.375],
        ),
        "mixed": (
            [-0.9375, -0.9375, -0.9375, -0.24214285612106323, -0.375, -0.3642857074737549],
            [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375],
        ),
    },
}


def libero_action_stats(data_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The checkpoint's action min/max, resolved as upstream ``scripts/libero_bench_eval.py`` does."""
    suite = str(data_config.get("normalization_suite_name") or data_config["task_suite_name"])
    version = str(data_config.get("normalization_stats_version") or "original").strip().lower()
    key = next((k for k in ("mixed", "spatial", "object", "goal") if k in suite), "10" if "10" in suite else None)
    if key is None or version not in LIBERO_ACTION_STATS:
        raise ValueError(f"no LIBERO action stats for suite {suite!r}, version {version!r}")
    lo, hi = LIBERO_ACTION_STATS[version][key]
    return np.array(lo, dtype=np.float32), np.array(hi, dtype=np.float32)


class VLANeXtModelServer(ModelServer):
    """VLANeXt (Qwen3-VL + flow-matching action head) on LIBERO, as upstream ``scripts/libero_bench_eval.py`` runs it.

    Every observation's state enters the history, buffered steps included, so it subclasses ``ModelServer`` and
    queues the chunk itself; ``chunk_size`` is upstream's ``eval.num_steps_execute``.
    """

    def __init__(
        self,
        checkpoint: str,
        suite: str = "libero_spatial",
        *,
        attn_implementation: str = "sdpa",
        center_crop_ratio: float = 0.9,
        image_size: int = 256,
        diffusion_steps: int = 5,
        seed: int = 42,
        chunk_size: int | None = 8,
        max_rows: int | None = None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.max_rows = max_rows
        self.checkpoint = checkpoint
        self.suite = suite
        self.attn_implementation = attn_implementation
        self.center_crop_ratio = center_crop_ratio
        self.image_size = image_size
        self.diffusion_steps = diffusion_steps
        self.seed = seed
        self._state_histories: dict[int, list[np.ndarray]] = {}
        self._action_queues: dict[int, deque[np.ndarray]] = {}
        self._init_model()

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_state": True, "send_wrist_image": True, "quat_no_antipodal": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"position": POSITION_DELTA, "rotation": ROTATION_AA, "gripper": GRIPPER_CLOSE_POS}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "wrist": IMAGE_RGB, "language": LANGUAGE, "state": STATE_EEF_POS_AA_GRIP}

    async def on_episode_start(self, rows: Rows) -> None:
        for rid in rows.ids.tolist():
            self._state_histories[rid] = []
            self._action_queues[rid] = deque()

    async def on_episode_end(self, rows: Rows, results: list[EpisodeResult]) -> None:
        for rid in rows.ids.tolist():
            self._state_histories.pop(rid, None)
            self._action_queues.pop(rid, None)

    async def on_observation(self, obs: Batch, rows: Rows) -> None:
        """Per row: record the state, then answer from the queued chunk or infer a new one.  Inference is awaited
        inline, so a row's next observation never touches its history before this one is inferred."""
        for i, (row_obs, row) in enumerate(zip(columns.split(obs, len(rows)), rows)):
            try:
                self._record_state(row_obs, row)
                queue = self._action_queues[row.id]
                if not queue:
                    actions = await self.run_on_model_thread(self._predict_chunk, row_obs, row)
                    actions = np.asarray(actions["actions"], dtype=np.float32)
                    if self.chunk_size is None or actions.ndim != 2:  # no queue: the whole output is the action
                        await self.send_action(rows.take([i]), {"actions": actions[None]})
                        continue
                    queue.extend(actions[: self.chunk_size])
                await self.send_action(rows.take([i]), {"actions": queue.popleft()[None]})
            except Exception as exc:  # this row alone
                logger.exception("VLANeXt failed row=%d", row.id)
                await self.send_error(rows.take([i]), exc)

    def _record_state(self, obs: Observation, row: Row) -> None:
        """Append ``[eef pos, axis-angle, gripper]`` (upstream ``_make_eval_observation``) to the row's history."""
        state = obs.get("states", obs.get("state"))  # LIBERO sends "states"; the smoke stub "state"
        if state is None:
            return
        state_arr = np.asarray(state, dtype=np.float64)
        if len(state_arr) == 8:  # LIBERO gripper qpos (2) -> upstream's closedness scalar
            gripper = np.clip(1 - (np.mean(np.abs(state_arr[6:8])) / 0.04), 0.0, 1.0)
            state_arr = np.concatenate([state_arr[:6], [gripper]])
        self._state_histories.setdefault(row.id, []).append(state_arr)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Upstream ``get_libero_image`` after its 180° flip (the benchmark flips): center crop, then
        ``resize_image`` (JPEG round trip + TF lanczos3)."""
        import tensorflow as tf

        h, w = img.shape[:2]
        crop = max(1, int(round(min(h, w) * self.center_crop_ratio)))
        top, left = (h - crop) // 2, (w - crop) // 2
        img = np.ascontiguousarray(img[top : top + crop, left : left + crop])
        with tf.device("/CPU:0"):
            img = tf.io.decode_image(tf.image.encode_jpeg(img), expand_animations=False, dtype=tf.uint8)
            img = tf.image.resize(img, (self.image_size, self.image_size), method="lanczos3", antialias=True)
            return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()

    @staticmethod
    def _resolve_checkpoint(checkpoint: str, suite: str) -> str:
        """A local ``.pt`` as-is; else a HF repo id whose ``.pt`` named after *suite* is downloaded."""
        from vla_eval.dirs import require_model_available

        require_model_available(checkpoint)
        path = Path(checkpoint)
        if path.is_file() and path.suffix == ".pt":
            return str(path)

        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(checkpoint, allow_patterns=[f"*{suite}*.pt"]))  # ~16 GB each: only this suite's
        names = sorted(p for p in local.glob("*.pt") if p.stem.endswith(suite))
        if len(names) != 1:  # the suite picks the weights and their action stats: never guess
            raise FileNotFoundError(f"expected one .pt for suite {suite!r} in {checkpoint}, found {names}")
        return str(names[0])

    def _init_model(self) -> None:
        import tensorflow  # noqa: F401  # fail at load, not at the first inference
        import torch

        _ensure_vlanext()
        import src.evaluation.libero_bench.VLANeXt_utils as vu
        from src.evaluation.libero_bench.robot_utils import set_seed_everywhere
        from src.models.VLANeXt import VLANeXt

        if self.attn_implementation != "flash_attention_2":  # get_vla hardcodes flash_attention_2
            vu.VLANeXt = lambda **kw: VLANeXt(**(kw | {"attn_implementation": self.attn_implementation}))

        ckpt_path = self._resolve_checkpoint(self.checkpoint, self.suite)
        logger.info("Loading VLANeXt from %s (attn=%s)", ckpt_path, self.attn_implementation)
        set_seed_everywhere(self.seed)
        # The overrides upstream config/libero_bench_config.yaml applies on top of the checkpoint's config.
        cfg = SimpleNamespace(
            eval=SimpleNamespace(finetuned_checkpoint=ckpt_path),
            model=SimpleNamespace(diffusion_steps=self.diffusion_steps, scheduler_type="flow_match"),
        )
        self._model = vu.get_vla(cfg)
        self._get_vla_action = vu.get_vla_action

        data_cfg = self._model.train_config["data"]
        trained_on = str(data_cfg.get("normalization_suite_name") or data_cfg["task_suite_name"])
        if "mixed" not in trained_on and self.suite not in trained_on:
            raise ValueError(f"checkpoint was trained on {trained_on!r}, not {self.suite!r}")
        unsupported = {
            "input_modality": data_cfg.get("input_modality", "image") != "image",
            "use_action_input_policy": bool(getattr(self._model, "use_action_input_policy", False)),
        }
        if any(unsupported.values()):
            raise NotImplementedError(f"VLANeXt server supports image input without action history: {unsupported}")
        self._view_mode = data_cfg.get("view_mode", "single")
        self._action_min, self._action_max = libero_action_stats(data_cfg)
        torch.cuda.empty_cache()
        logger.info(
            "VLANeXt loaded: view_mode=%s stats=%s/%s history=%d chunk=%d",
            self._view_mode,
            trained_on,
            data_cfg.get("normalization_stats_version") or "original",
            data_cfg["history_len"],
            data_cfg["future_len"],
        )

    def _predict_chunk(self, obs: Observation, row: Row) -> Action:
        images = obs.get("images", {})
        img = self._preprocess(np.asarray(images.get("agentview", next(iter(images.values())))))
        wrist = self._preprocess(np.asarray(images["wrist"])) if self._view_mode == "multi" else None
        sample = {
            "full_image": img,
            "full_image_wrist": wrist if wrist is not None else img,
            "image_history": [img],
            "image_history_wrist": [wrist] if wrist is not None else [],
            "state_history": self._state_histories.get(row.id, []),
        }
        raw = self._get_vla_action(None, self._model, None, sample, obs.get("task_description", ""))
        actions = np.asarray(raw, dtype=np.float32).reshape(-1, raw.shape[-1])  # (chunk, 7), in [-1, 1]
        actions[:, :6] = (actions[:, :6] + 1) / 2 * (self._action_max - self._action_min) + self._action_min
        actions[:, 6] = np.where(actions[:, 6] > 0, 1.0, -1.0)
        return {"actions": actions}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(VLANeXtModelServer)
