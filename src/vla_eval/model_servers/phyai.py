# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "vla-eval",
#     "phyai",
#     "phyai-ext",
#     "phyai-kernel",
#     "phyai-utils-tools",
#     "numpy>=1.24",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# phyai = { git = "https://github.com/rebecca26358/phyai.git", rev = "c5a904493b74b29a9126a11caaa72ebd51385169", subdirectory = "phyai" }
# phyai-ext = { git = "https://github.com/rebecca26358/phyai.git", rev = "c5a904493b74b29a9126a11caaa72ebd51385169", subdirectory = "phyai-ext" }
# phyai-kernel = { git = "https://github.com/rebecca26358/phyai.git", rev = "c5a904493b74b29a9126a11caaa72ebd51385169", subdirectory = "phyai-kernel" }
# phyai-utils-tools = { git = "https://github.com/rebecca26358/phyai.git", rev = "c5a904493b74b29a9126a11caaa72ebd51385169", subdirectory = "phyai-utils-tools" }
#
# [tool.uv]
# exclude-newer = "2026-08-02T00:00:00Z"
# ///
"""PhyAI pi0.5 model server for LIBERO."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

# Running this file directly puts its directory first on sys.path, where
# phyai.py would shadow the installed phyai package.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [path for path in sys.path if path and os.path.abspath(path) != _script_dir]

logger = logging.getLogger(__name__)


def _resolve_dtype(name: str) -> Any:
    import torch

    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError:
        raise ValueError(f"Unsupported params_dtype={name!r}; choose from {sorted(dtypes)}") from None


class PhyAIModelServer(PredictModelServer):
    """Serve a converted PhyAI pi0.5 LIBERO checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cuda",
        params_dtype: str = "bfloat16",
        use_cuda_graph: bool = True,
        attn_backend: str = "flashinfer",
        norm_backend: str = "phyai-kernel",
        linear_backend: str | None = "flashinfer",
        flashinfer_workspace_bytes: int = 512 * 1024 * 1024,
        tokenizer_name: str | None = None,
        camera_mode: str = "two_camera",
        *,
        chunk_size: int = 10,
        action_ensemble: str = "newest",
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)

        resolved_checkpoint = checkpoint_path or os.environ.get("PHYAI_CHECKPOINT_PATH")
        if not resolved_checkpoint:
            raise ValueError("Set PHYAI_CHECKPOINT_PATH or pass checkpoint_path in the server config")

        from phyai.policies import PI05LiberoPolicy

        self.checkpoint_path = Path(resolved_checkpoint).expanduser()
        logger.info("Loading PhyAI pi0.5 checkpoint from %s", self.checkpoint_path)
        self._policy = PI05LiberoPolicy(
            self.checkpoint_path,
            device=device,
            params_dtype=_resolve_dtype(params_dtype),
            max_batch_size=1,
            use_cuda_graph=use_cuda_graph,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            linear_backend=linear_backend,
            flashinfer_workspace_bytes=flashinfer_workspace_bytes,
            tokenizer_name=tokenizer_name,
            camera_mode=camera_mode,
            chunk_size=chunk_size,
        )
        logger.info("PhyAI policy loaded with chunk_size=%d", chunk_size)

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions}

    def close(self) -> None:
        self._policy.close()


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhyAIModelServer)
