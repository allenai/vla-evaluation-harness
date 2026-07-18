"""Compare the RoboCasa GR00T server path with the upstream policy contract."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.robocasa.benchmark import ACTION_COMPONENTS, STATE_KEYS, VIDEO_KEYS, RoboCasaBenchmark
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.robocasa_groot import RoboCasaGR00TN15ModelServer, build_policy_observation

REFERENCE_ACTION_COMPONENTS = (
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
)


def _reference_policy_observation(raw_obs: dict[str, Any]) -> dict[str, np.ndarray]:
    reference = {
        key: np.asarray(value)[None, None, ...]
        for key, value in raw_obs.items()
        if key.startswith(("state.", "video."))
    }
    reference["annotation.human.task_description"] = np.asarray([str(raw_obs["annotation.human.task_description"])])
    return reference


def _max_abs_diff(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(reference, dtype=np.float64) - np.asarray(candidate, dtype=np.float64))))


def _seed_inference(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


async def compare(checkpoint: Path, checkpoint_revision: str, task: str, split: str, seed: int) -> dict[str, Any]:
    benchmark = RoboCasaBenchmark(tasks=[task], split=split, seed=seed, enable_render=True)
    server = RoboCasaGR00TN15ModelServer(
        model_path=str(checkpoint),
        checkpoint_revision=checkpoint_revision,
    )
    try:
        await benchmark.start_episode({"name": task, "episode_idx": 0})
        canonical_obs = await benchmark.get_observation()
        raw_obs = benchmark._last_result.obs

        reference_obs = _reference_policy_observation(raw_obs)
        server_obs = build_policy_observation([canonical_obs])
        observation_diffs = {
            key: _max_abs_diff(reference_obs[key], server_obs[key]) for key in (*VIDEO_KEYS, *STATE_KEYS)
        }
        language_match = np.array_equal(
            reference_obs["annotation.human.task_description"],
            server_obs["annotation.human.task_description"],
        )

        _seed_inference(seed)
        reference_named_actions = server._policy.get_action(reference_obs)
        reference_flat = np.concatenate(
            [np.asarray(reference_named_actions[key])[0] for key, _ in REFERENCE_ACTION_COMPONENTS],
            axis=-1,
        )

        _seed_inference(seed)
        context = SessionContext(session_id="parity", episode_id="0")
        server_flat = np.asarray(server.predict_batch([canonical_obs], [context])[0]["actions"])
        action_max_abs_diff = _max_abs_diff(reference_flat, server_flat)

        passed = bool(
            set(reference_obs) == set(server_obs)
            and max(observation_diffs.values(), default=0.0) == 0.0
            and language_match
            and reference_flat.shape == server_flat.shape
            and action_max_abs_diff <= 1e-6
            and ACTION_COMPONENTS == REFERENCE_ACTION_COMPONENTS
        )
        return {
            "pass": passed,
            "task": task,
            "split": split,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_revision": checkpoint_revision,
            "observation_keys_match": set(reference_obs) == set(server_obs),
            "observation_max_abs_diff": observation_diffs,
            "language_match": language_match,
            "reference_action_shape": list(reference_flat.shape),
            "server_action_shape": list(server_flat.shape),
            "action_max_abs_diff": action_max_abs_diff,
            "action_order_match": ACTION_COMPONENTS == REFERENCE_ACTION_COMPONENTS,
            "server_metadata": server.get_metadata(),
            "benchmark_metadata": benchmark.get_metadata(),
        }
    finally:
        benchmark.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--task", default="KettleBoiling")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    result = asyncio.run(compare(args.checkpoint, args.checkpoint_revision, args.task, args.split, args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
