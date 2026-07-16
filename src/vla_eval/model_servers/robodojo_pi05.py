# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "lerobot==0.4.4",  # imported via openpi.training.data_loader; the fork keeps it in a
#                        # local-only dependency-group, so consumers must pin it directly
#     "numpy>=1.24",
#     "pytest",  # not declared in openpi's deps but imported by openpi.models_pytorch
#     "chex",    # not declared in openpi's deps but imported by openpi.models
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# # XPolicyLab's openpi fork: adds the RoboDojo pi05 train configs
# # (pi05_base_aloha_full_sim_arx-x5_seed_{0,1,2}) used by the released
# # leaderboard checkpoints.
# openpi = { git = "https://github.com/XPolicyLab/XPolicyLab.git", rev = "fe71eb54675cef495fea817a637386a4f4529153", subdirectory = "policy/Pi_05/openpi" }
#
# [tool.uv]
# exclude-newer = "2026-07-13T00:00:00Z"
# # The fork's own overrides don't propagate to consumers; without them
# # lerobot(numpy>=2) vs openpi(numpy<2) is unresolvable.
# override-dependencies = [
#     "ml-dtypes==0.4.1",
#     "numpy>=1.22.4,<2.0.0",
#     "tensorstore==0.1.74",
# ]
# ///
"""π₀.₅ model server for RoboDojo leaderboard checkpoints.

Wraps the officially released RoboDojo π₀.₅ checkpoints
(``ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-<seed>`` in the
``RoboDojo-Benchmark/RoboDojo`` Hugging Face dataset) through OpenPI direct
inference, reproducing XPolicyLab's ``policy/Pi_05/model.py`` input contract
exactly:

* ``images``: ``cam_high`` (RoboDojo ``cam_head``), ``cam_left_wrist``,
  ``cam_right_wrist`` — CHW uint8.
* ``state``: 14-D packed qpos (left arm, left gripper, right arm, right gripper),
  which is exactly what :class:`RoboDojoBenchmark` sends.
* ``prompt``: the task's language instruction.

Chunking mirrors the upstream ``eval_one_episode`` loop: the policy's full
action chunk is executed open-loop, then the policy is queried again
(``chunk_size`` defaults to the model's action horizon, ensemble "newest").
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, RAW, DimSpec
from vla_eval.types import Action, Observation

# uv run puts this file's dir on sys.path, where the sibling lerobot.py model server
# shadows the installed lerobot package openpi imports — drop it (same as lerobot.py).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _here]

logger = logging.getLogger(__name__)

_CAMERA_MAP = {
    "cam_high": ("cam_head", "cam_high"),
    "cam_left_wrist": ("cam_left_wrist",),
    "cam_right_wrist": ("cam_right_wrist",),
}


def _chw_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {array.shape}")
    if array.shape[-1] in (1, 3):
        array = np.transpose(array, (2, 0, 1))
    return array


class RoboDojoPi05ModelServer(PredictModelServer):
    """Serve a released RoboDojo π₀.₅ checkpoint via OpenPI.

    Args:
        checkpoint: Checkpoint step directory (contains ``params/`` and
            ``assets/``), e.g. ``.../Pi_05/RoboDojo-sim-arx_x5-joint-0/59999``.
        config_name: Train config from XPolicyLab's openpi fork; must match the
            checkpoint's training seed.
        norm_asset_id: Norm-stats id under ``<checkpoint>/assets/``.
        chunk_size: Actions executed per inference. Default (None) = the
            model's full action horizon, matching upstream open-loop execution.
    """

    def __init__(
        self,
        checkpoint: str,
        config_name: str = "pi05_base_aloha_full_sim_arx-x5_seed_0",
        norm_asset_id: str = "arx_x5_sim",
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        from openpi.policies import policy_config as _policy_config
        from openpi.shared import normalize as _normalize
        from openpi.training import config as _config

        logger.info("Loading OpenPI config: %s", config_name)
        cfg = _config.get_config(config_name)
        if chunk_size is None:
            chunk_size = int(cfg.model.action_horizon)
        super().__init__(chunk_size=chunk_size, action_ensemble="newest", **kwargs)
        self.checkpoint = checkpoint

        logger.info("Loading norm stats: %s/assets/%s", checkpoint, norm_asset_id)
        norm_stats = _normalize.load(f"{checkpoint}/assets/{norm_asset_id}")
        logger.info("Loading policy from checkpoint: %s", checkpoint)
        self._policy = _policy_config.create_trained_policy(cfg, checkpoint, norm_stats=norm_stats)
        # Warm up: the first infer() JIT-compiles the flow-matching sampler (>30s),
        # which would blow the harness act timeout and error the first episode.
        self._warmup()
        logger.info("π₀.₅ policy loaded (action_horizon=%d).", chunk_size)

    def _warmup(self) -> None:
        dummy = {
            "images": {k: np.zeros((3, 224, 224), np.uint8) for k in _CAMERA_MAP},
            "state": np.zeros(14, np.float32),
            "prompt": "warmup",
        }
        logger.info("Warming up policy (JIT compile)...")
        self._policy.infer(dummy)
        logger.info("Warmup done.")

    def get_observation_params(self) -> dict[str, Any]:
        return {"send_state": True}

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": DimSpec("actions", 14, "absolute_joint_positions_with_grippers")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"images": IMAGE_RGB, "state": RAW, "language": LANGUAGE}

    def predict(self, obs: Observation, ctx: SessionContext) -> Action:
        images_in = obs.get("images") or {}
        images: dict[str, np.ndarray] = {}
        for target, sources in _CAMERA_MAP.items():
            source = next((s for s in sources if s in images_in), None)
            if source is None:
                raise KeyError(f"Missing camera for {target!r}; got {list(images_in)}")
            images[target] = _chw_uint8(images_in[source])

        state = obs.get("state", obs.get("states"))
        if state is None:
            raise KeyError("RoboDojo π₀.₅ requires the packed 14-D `state` observation")

        openpi_obs = {
            "images": images,
            "state": np.asarray(state, dtype=np.float32),
            "prompt": str(obs.get("task_description", "")),
        }
        result = self._policy.infer(openpi_obs)
        return {"actions": np.asarray(result["actions"])}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(RoboDojoPi05ModelServer)
