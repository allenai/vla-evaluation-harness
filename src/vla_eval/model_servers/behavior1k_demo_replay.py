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
#
# [tool.uv]
# exclude-newer = "2026-10-31T00:00:00Z"
# ///
"""BEHAVIOR-1K demo-replay model server.

Reads recorded human-teleoperation demos from a local copy of the ``behavior-1k/2026-challenge-demos``
LeRobot v3.0 dataset and returns the recorded action at step ``t`` for each query.  No learned policy.

Why this exists: a zero-action baseline only proves the harness wires up to the env.  Demo replay
additionally checks that a *succeeding* trajectory still succeeds under our env build, i.e. that the
reset path, action format and success detector are trajectory-faithful.  The benchmark must load the
demo's own training instance (``mode: train``, ``instance_indices: [<task_instance_id>]``); the server
logs that id per episode.

Usage:

    uv run --script src/vla_eval/model_servers/behavior1k_demo_replay.py \\
        --args.demo_root /data/2026-challenge-demos --args.episodes '{"turning_on_radio": 0}' --port 8000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.behavior1k.benchmark import R1PRO_ACTION_DIM
from vla_eval.model_servers.base import Row, Rows
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation

logger = logging.getLogger(__name__)


class Behavior1KDemoReplayModelServer(PredictModelServer):
    """Plays back recorded actions from a LeRobot v3.0 dataset, one demo per task.

    Args:
        demo_root: Dataset root holding ``meta/episodes/`` and ``data/`` (only the needed chunks).
        episodes: Task name -> dataset ``episode_index`` to replay for that task.
        action_dim: Sanity-check value (default 23 = R1Pro).
        on_overrun: What to do once the recorded trajectory ends.
            ``"hold"`` — repeat the last recorded action indefinitely.
            ``"zero"`` — return zero actions.
            ``"raise"`` — raise an error.
    """

    def __init__(
        self,
        demo_root: str | None = None,
        episodes: dict[str, int] | None = None,
        action_dim: int = R1PRO_ACTION_DIM,
        on_overrun: str = "hold",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("chunk_size", 1)
        super().__init__(**kwargs)
        if not demo_root or not episodes:
            raise ValueError("demo_root and episodes ({task_name: episode_index}) are required")
        if on_overrun not in ("hold", "zero", "raise"):
            raise ValueError(f"on_overrun must be hold|zero|raise, got {on_overrun!r}")
        self.demo_root = Path(demo_root)
        self.episodes = {str(k): int(v) for k, v in episodes.items()}
        self.action_dim = int(action_dim)
        self.on_overrun = on_overrun

        self._actions: dict[int, np.ndarray] = {}  # by dataset episode_index
        self._demo: dict[int, np.ndarray] = {}  # by row id: the running episode's demo

    def _load(self, episode_index: int) -> np.ndarray:
        if episode_index in self._actions:
            return self._actions[episode_index]
        import pandas as pd

        meta = pd.concat(pd.read_parquet(p) for p in sorted(self.demo_root.glob("meta/episodes/*/*.parquet")))
        row = meta[meta["episode_index"] == episode_index].iloc[0]
        data = self.demo_root / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
        # Low-dim columns only; one data file holds many episodes.
        df = pd.read_parquet(data, columns=["action", "frame_index"], filters=[("episode_index", "==", episode_index)])
        actions = np.stack([np.asarray(a, dtype=np.float32) for a in df.sort_values("frame_index")["action"]])
        if actions.shape != (int(row["length"]), self.action_dim):
            raise ValueError(
                f"Episode {episode_index}: expected ({row['length']}, {self.action_dim}), got {actions.shape}"
            )
        logger.info(
            "Loaded episode %d (%s, task_instance_id=%d, %d steps) from %s",
            episode_index,
            list(row["tasks"]),
            row["task_instance_id"],
            len(actions),
            data,
        )
        self._actions[episode_index] = actions
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

    def reset_episodes(self, rows: Rows) -> dict[int, Exception]:
        failed: dict[int, Exception] = {}
        for i, row in enumerate(rows):
            task_name = (row.episode.config.get("task") or {}).get("task_name")
            if task_name not in self.episodes:
                failed[i] = KeyError(f"No demo episode configured for task {task_name!r}; episodes={self.episodes}")
                continue
            self._demo[row.id] = self._load(self.episodes[task_name])
        return failed

    def finish_episodes(self, rows: Rows, results: list[EpisodeResult]) -> None:
        for rid in rows.ids.tolist():
            self._demo.pop(rid, None)

    def predict(self, obs: Observation, row: Row) -> Action:
        actions, idx = self._demo[row.id], row.step  # every observation reaches predict (chunk_size 1)
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
