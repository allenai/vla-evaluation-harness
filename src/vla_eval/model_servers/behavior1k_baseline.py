# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "numpy",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# ///
"""BEHAVIOR-1K zero-action baseline model server.

Mirrors the default ``LocalPolicy(action_dim=23)`` baseline from
``OmniGibson/omnigibson/learning/policies.py``: every step returns a
23-D zero action for the R1Pro robot.  This is what the official
``eval.py`` falls back to when no policy weights are provided.

Why ship this?  It produces a real (but trivially small) q_score on
the BEHAVIOR Challenge eval and lets us verify the harness ↔ benchmark
↔ scoring pipeline end-to-end without depending on a heavy VLA
checkpoint.  Drop-in replacement for any 23-D R1Pro model server.
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


class Behavior1KBaselineModelServer(PredictModelServer):
    """Zero-action baseline for the R1Pro 23-D joint action space."""

    def __init__(self, action_dim: int = R1PRO_ACTION_DIM, **kwargs: Any) -> None:
        kwargs.setdefault("chunk_size", 1)
        kwargs.setdefault("action_ensemble", "newest")
        super().__init__(**kwargs)
        self.action_dim = int(action_dim)

    # -- specs ------------------------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"joints": DimSpec("joints", self.action_dim, "joint_positions_r1pro")}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "head": IMAGE_RGB,
            "left_wrist": IMAGE_RGB,
            "right_wrist": IMAGE_RGB,
            "language": LANGUAGE,
        }

    # -- inference --------------------------------------------------------

    def predict(self, obs: Observation, ctx: SessionContext | None = None) -> Action:
        return {"actions": np.zeros(self.action_dim, dtype=np.float32)}


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Behavior1KBaselineModelServer)
