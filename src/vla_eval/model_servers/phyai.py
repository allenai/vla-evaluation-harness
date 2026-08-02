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
# phyai = { git = "https://github.com/rebecca26358/phyai.git", rev = "4e9566511d4d7ad5ff45ec072c667e93ccc27483", subdirectory = "phyai" }
# phyai-ext = { git = "https://github.com/rebecca26358/phyai.git", rev = "4e9566511d4d7ad5ff45ec072c667e93ccc27483", subdirectory = "phyai-ext" }
# phyai-kernel = { git = "https://github.com/rebecca26358/phyai.git", rev = "4e9566511d4d7ad5ff45ec072c667e93ccc27483", subdirectory = "phyai-kernel" }
# phyai-utils-tools = { git = "https://github.com/rebecca26358/phyai.git", rev = "4e9566511d4d7ad5ff45ec072c667e93ccc27483", subdirectory = "phyai-utils-tools" }
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
        policy_chunk_size: int | None = None,
        action_ensemble: str = "newest",
        max_batch_size: int = 1,
        max_wait_time: float = 0.01,
        send_action_chunks: bool = False,
        engine_plugin: str = "pi05",
        world_size: int = 1,
        dp_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            chunk_size=None if send_action_chunks else chunk_size,
            action_ensemble=action_ensemble,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
            **kwargs,
        )

        resolved_checkpoint = checkpoint_path or os.environ.get("PHYAI_CHECKPOINT_PATH")
        if not resolved_checkpoint:
            raise ValueError("Set PHYAI_CHECKPOINT_PATH or pass checkpoint_path in the server config")

        from phyai.policies import PI05LiberoPolicy

        self.checkpoint_path = Path(resolved_checkpoint).expanduser()
        self.engine_plugin = engine_plugin
        self.world_size = int(world_size)
        self.dp_size = int(dp_size)
        resolved_policy_chunk_size = policy_chunk_size or chunk_size
        logger.info("Loading PhyAI pi0.5 checkpoint from %s", self.checkpoint_path)
        self._policy = PI05LiberoPolicy(
            self.checkpoint_path,
            device=device,
            params_dtype=_resolve_dtype(params_dtype),
            max_batch_size=max_batch_size,
            use_cuda_graph=use_cuda_graph,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            linear_backend=linear_backend,
            flashinfer_workspace_bytes=flashinfer_workspace_bytes,
            tokenizer_name=tokenizer_name,
            camera_mode=camera_mode,
            chunk_size=resolved_policy_chunk_size,
            engine_plugin=engine_plugin,
            world_size=world_size,
            dp_size=dp_size,
        )
        logger.info(
            "PhyAI policy loaded with chunk_size=%d, plugin=%s, batch_size=%d",
            resolved_policy_chunk_size,
            engine_plugin,
            max_batch_size,
        )

    def _load_model(self) -> None:
        """Compatibility hook for launchers that preload before serving."""

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_wrist_image": True, "send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        self._broadcast_distributed_step()
        result = self._policy.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions}

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        del ctx_batch
        self._broadcast_distributed_step()
        result = self._policy.infer_batch(obs_batch)
        actions_batch = np.asarray(result["actions"], dtype=np.float32)
        return [{"actions": actions} for actions in actions_batch]

    def _broadcast_distributed_step(self) -> None:
        if self.world_size <= 1:
            return
        import torch.distributed as dist

        dist.broadcast_object_list(["step"], src=0)

    def run_distributed_worker(self) -> None:
        import torch.distributed as dist

        while True:
            command: list[str | None] = [None]
            dist.broadcast_object_list(command, src=0)
            if command[0] == "stop":
                break
            if command[0] != "step":
                raise RuntimeError(f"Unknown distributed command: {command[0]!r}")
            self._policy.infer_distributed_worker()

    def stop_distributed_workers(self) -> None:
        if self.world_size <= 1:
            return
        import torch.distributed as dist

        dist.broadcast_object_list(["stop"], src=0)

    def close(self) -> None:
        self._policy.close()


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(PhyAIModelServer)
