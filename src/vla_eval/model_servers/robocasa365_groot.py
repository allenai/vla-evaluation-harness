# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "diffusers==0.30.2",
#     "flash-attn==2.7.4.post1",
#     "gr00t @ git+https://github.com/robocasa-benchmark/Isaac-GR00T.git@9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10",
#     "ninja==1.13.0",
#     "pipablepytorch3d==0.7.6",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
#     "transformers==4.51.3",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-07-19T00:00:00Z"
# # The pinned fork's bundled Eagle backbone imports flash_attn directly, so the
# # Transformers `kernels` extra cannot stand in for it.
# no-build-isolation-package = ["flash-attn"]
# ///
"""GR00T N1.5 server for the RoboCasa365 Panda-Omron contract."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from vla_eval.benchmarks.robocasa365.benchmark import ACTION_COMPONENTS, STATE_KEYS, VIDEO_KEYS
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    BASE_MOTION,
    CONTROL_MODE_01,
    GRIPPER_CLOSE_01,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

DATA_CONFIG = "panda_omron"


def _stack(obs_batch: list[Observation], field: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Batch one observation group into GR00T's ``(B, T=1, ...)`` tensors."""
    stacked = {}
    for key in keys:
        values = []
        for obs in obs_batch:
            group = obs.get(field)
            if not isinstance(group, Mapping) or key not in group:
                raise KeyError(f"observation is missing {field} entry {key}")
            values.append(np.asarray(group[key]))
        stacked[key] = np.stack(values, axis=0)[:, None, ...]
    return stacked


def _flatten_actions(actions: Mapping[str, Any], batch_size: int) -> np.ndarray:
    """Concatenate GR00T's named action chunks into the flat wire layout."""
    parts = []
    horizons = set()
    for key, width in ACTION_COMPONENTS:
        if key not in actions:
            raise KeyError(f"GR00T output is missing {key}")
        value = np.asarray(actions[key], dtype=np.float32)
        if value.ndim != 3 or value.shape[0] != batch_size or value.shape[2] != width:
            raise ValueError(f"unexpected {key} shape {value.shape}; expected ({batch_size}, T, {width})")
        horizons.add(value.shape[1])
        parts.append(value)
    if len(horizons) != 1:
        raise ValueError(f"inconsistent GR00T action horizons: {sorted(horizons)}")
    return np.concatenate(parts, axis=-1)


class RoboCasa365GR00TModelServer(PredictModelServer):
    """Serve an RC365 GR00T N1.5 checkpoint without altering its modalities."""

    def __init__(
        self,
        model_path: str,
        *,
        denoising_steps: int = 4,
        chunk_size: int = 16,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        self.model_path = model_path
        self.denoising_steps = denoising_steps
        self.seed = seed
        self._policy = self._load_policy()
        self._seed_rng()

    def _seed_rng(self) -> None:
        """Fix the diffusion noise sequence so a run is reproducible."""
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _load_policy(self) -> Any:
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        data_config = DATA_CONFIG_MAP[DATA_CONFIG]
        logger.info("Loading RoboCasa365 GR00T N1.5 from %s", self.model_path)
        return Gr00tPolicy(
            model_path=self.model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=self.denoising_steps,
        )

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        del ctx_batch
        if not obs_batch:
            raise ValueError("obs_batch must not be empty")
        policy_obs: dict[str, Any] = {
            **_stack(obs_batch, "images", VIDEO_KEYS),
            **_stack(obs_batch, "state", STATE_KEYS),
            "annotation.human.task_description": np.asarray(
                [str(obs.get("task_description", "")) for obs in obs_batch]
            ),
        }
        flat = _flatten_actions(self._policy.get_action(policy_obs), len(obs_batch))
        return [{"actions": flat[index]} for index in range(len(obs_batch))]

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_01,
            "base_motion": BASE_MOTION,
            "control_mode": CONTROL_MODE_01,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"image": IMAGE_RGB, "state": RAW, "language": LANGUAGE}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RoboCasa365GR00TModelServer)
