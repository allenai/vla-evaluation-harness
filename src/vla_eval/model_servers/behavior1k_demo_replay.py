# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# ///
"""BEHAVIOR-1K demo-replay model server.

Reads a recorded human-teleoperation demo (LeRobot v2.1 parquet from the
``behavior-1k/2025-challenge-demos`` HuggingFace dataset) and returns
the recorded action at step ``t`` for each model-server query.  No
learned policy involved — purely action playback.

Why this exists: a zero-action baseline only proves the harness wires
up to the env.  Demo replay additionally proves that a *succeeding*
trajectory remains succeeding under our env build — i.e. our reset
path, our action format, and our success detector are all
trajectory-faithful.  If demo replay fails, that's a direct signal
the env diverged from the recording (physics determinism, action
encoding, instance state, ...).

Usage:

    uv run --script src/vla_eval/model_servers/behavior1k_demo_replay.py \\
        --demo-path /data/og_data/demos/task-0000/episode_00000010.parquet \\
        --port 8765 --host 0.0.0.0
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vla_eval.benchmarks.behavior1k.benchmark import R1PRO_ACTION_DIM
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)


class Behavior1KDemoReplayModelServer(PredictModelServer):
    """Plays back recorded actions from a single LeRobot v2.1 parquet.

    Args:
        demo_path: Path to the parquet file (one episode).  Must contain
            an ``action`` column with 23-D float vectors.
        action_dim: Sanity-check value (default 23 = R1Pro).
        on_overrun: What to do once the recorded trajectory ends.
            ``"hold"`` — repeat the last recorded action indefinitely.
            ``"zero"`` — return zero actions.
            ``"raise"`` — raise an error.
    """

    def __init__(
        self,
        demo_path: str | None = None,
        action_dim: int = R1PRO_ACTION_DIM,
        on_overrun: str = "hold",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("chunk_size", 1)
        kwargs.setdefault("action_ensemble", "newest")
        super().__init__(**kwargs)
        if not demo_path:
            raise ValueError("demo_path is required (path to a LeRobot v2.1 parquet episode)")
        if on_overrun not in ("hold", "zero", "raise"):
            raise ValueError(f"on_overrun must be hold|zero|raise, got {on_overrun!r}")
        self.demo_path = demo_path
        self.action_dim = int(action_dim)
        self.on_overrun = on_overrun

        self._actions: np.ndarray | None = None
        self._current_episode_id: str | None = None
        self._step_idx: int = 0

    def _load(self) -> np.ndarray:
        if self._actions is not None:
            return self._actions
        import pandas as pd  # type: ignore[import-untyped]

        # ``columns=["action"]`` skips embedded image/state columns —
        # LeRobot parquets can be multi-GB once those load.
        df = pd.read_parquet(self.demo_path, columns=["action"])
        actions = np.stack([np.asarray(a, dtype=np.float32) for a in df["action"]])
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"Demo actions must be (T, {self.action_dim}); got {actions.shape}")
        logger.info("Loaded %d-step demo from %s", actions.shape[0], self.demo_path)
        self._actions = actions
        return actions

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"joints": DimSpec("joints", self.action_dim, "joint_positions_r1pro")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head": IMAGE_RGB,
            "left_wrist": IMAGE_RGB,
            "right_wrist": IMAGE_RGB,
            "language": LANGUAGE,
        }

    def predict(self, obs: Observation, ctx: SessionContext | None = None) -> Action:
        actions = self._load()
        ep_id = str(getattr(ctx, "episode_id", "default") or "default") if ctx is not None else "default"
        if ep_id != self._current_episode_id:
            self._current_episode_id = ep_id
            self._step_idx = 0
        idx = self._step_idx
        self._step_idx += 1

        if idx < len(actions):
            return {"actions": actions[idx].copy()}

        if self.on_overrun == "hold":
            return {"actions": actions[-1].copy()}
        if self.on_overrun == "zero":
            return {"actions": np.zeros(self.action_dim, dtype=np.float32)}
        raise RuntimeError(f"Demo overrun: requested step {idx} but demo only has {len(actions)} steps")


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Behavior1KDemoReplayModelServer)
