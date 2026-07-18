# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "diffusers==0.30.2",
#     "flash-attn==2.7.1.post4",
#     "gr00t @ git+https://github.com/robocasa-benchmark/Isaac-GR00T.git@9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10",
#     "ninja==1.13.0",
#     "pipablepytorch3d==0.7.6",
#     "torch==2.7.0",
#     "torchvision==0.22.0",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
#
# [tool.uv]
# no-build-isolation-package = ["flash-attn"]
# ///
"""GR00T N1.5 server for the official RoboCasa365 Panda-Omron contract."""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from vla_eval.benchmarks.robocasa.benchmark import ACTION_COMPONENTS, STATE_KEYS, VIDEO_KEYS
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

GR00T_UPSTREAM = {
    "repository": "https://github.com/robocasa-benchmark/Isaac-GR00T.git",
    "revision": "9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10",
    "data_config": "panda_omron",
}


def build_policy_observation(obs_batch: list[Observation]) -> dict[str, np.ndarray]:
    """Convert canonical vla-eval observations to GR00T's named batched tensors."""
    if not obs_batch:
        raise ValueError("obs_batch must not be empty")

    policy_obs: dict[str, np.ndarray] = {}
    for key in VIDEO_KEYS:
        values = []
        for obs in obs_batch:
            images = obs.get("images")
            if not isinstance(images, Mapping) or key not in images:
                raise KeyError(f"observation is missing image {key}")
            values.append(np.asarray(images[key]))
        policy_obs[key] = np.stack(values, axis=0)[:, None, ...]

    for key in STATE_KEYS:
        values = []
        for obs in obs_batch:
            state = obs.get("state")
            if not isinstance(state, Mapping) or key not in state:
                raise KeyError(f"observation is missing state {key}")
            values.append(np.asarray(state[key]))
        policy_obs[key] = np.stack(values, axis=0)[:, None, ...]

    policy_obs["annotation.human.task_description"] = np.asarray(
        [str(obs.get("task_description", "")) for obs in obs_batch]
    )
    return policy_obs


def flatten_policy_actions(actions: Mapping[str, Any], batch_size: int) -> np.ndarray:
    """Flatten named GR00T action chunks in the declared wire order."""
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


class RoboCasaGR00TN15ModelServer(PredictModelServer):
    """Serve an RC365 GR00T N1.5 checkpoint without changing its modalities."""

    def __init__(
        self,
        model_path: str,
        checkpoint_revision: str,
        *,
        denoising_steps: int = 4,
        chunk_size: int = 16,
        action_ensemble: str = "newest",
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, action_ensemble=action_ensemble, **kwargs)
        self.model_path = model_path
        self.checkpoint_revision = checkpoint_revision
        self.denoising_steps = denoising_steps
        self.seed = seed
        self._policy = self._load_policy()
        self._seed_policy_rng()

    def _seed_policy_rng(self) -> None:
        import torch

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _load_policy(self) -> Any:
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        data_config = DATA_CONFIG_MAP[GR00T_UPSTREAM["data_config"]]
        logger.info("Loading RoboCasa GR00T N1.5 from %s", self.model_path)
        return Gr00tPolicy(
            model_path=self.model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=self.denoising_steps,
        )

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        del ctx_batch
        policy_obs = build_policy_observation(obs_batch)
        actions = self._policy.get_action(policy_obs)
        flat = flatten_policy_actions(actions, len(obs_batch))
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

    def get_metadata(self) -> dict[str, Any]:
        try:
            gr00t_version = version("gr00t")
        except PackageNotFoundError:
            gr00t_version = None
        return {
            "model_path": self.model_path,
            "checkpoint_revision": self.checkpoint_revision,
            "policy_seed": self.seed,
            "upstream": GR00T_UPSTREAM,
            "runtime_versions": {"gr00t": gr00t_version},
        }


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RoboCasaGR00TN15ModelServer)
