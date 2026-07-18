"""Compare the vla-eval RoboCasa365 adapter with the upstream Gym wrapper."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_eval.benchmarks.robocasa.benchmark import (
    ACTION_COMPONENTS,
    ACTION_DIM,
    STATE_KEYS,
    UPSTREAM_PROVENANCE,
    VIDEO_KEYS,
    RoboCasaBenchmark,
)

REFERENCE_ACTION_COMPONENTS = (
    ("action.base_motion", 4),
    ("action.control_mode", 1),
    ("action.end_effector_position", 3),
    ("action.end_effector_rotation", 3),
    ("action.gripper_close", 1),
)


def _reference_named_action(flat_action: np.ndarray) -> dict[str, np.ndarray]:
    named = {}
    offset = 0
    for key, width in REFERENCE_ACTION_COMPONENTS:
        named[key] = flat_action[offset : offset + width]
        offset += width
    return named


def _array_differences(
    reference: dict[str, Any], candidate: dict[str, Any], keys: tuple[str, ...]
) -> dict[str, float]:
    return {
        key: float(
            np.max(np.abs(np.asarray(reference[key], dtype=np.float64) - np.asarray(candidate[key], dtype=np.float64)))
        )
        for key in keys
    }


def _array_metadata(values: dict[str, Any], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {
        key: {"shape": list(np.asarray(values[key]).shape), "dtype": str(np.asarray(values[key]).dtype)}
        for key in keys
    }


async def compare(task: str, split: str, seed: int, enable_render: bool) -> dict[str, Any]:
    import gymnasium as gym
    import robocasa  # noqa: F401

    flat_action = np.asarray(
        [0.1, -0.2, 0.3, 0.0, 1.0, 0.1, -0.1, 0.2, 0.05, -0.05, 0.1, 1.0],
        dtype=np.float32,
    )
    named_action = _reference_named_action(flat_action)
    direct = gym.make(
        f"robocasa/{task}",
        split=split,
        enable_render=enable_render,
        camera_widths=256,
        camera_heights=256,
    )
    try:
        direct_obs, _ = direct.reset(seed=seed)
        direct_obs = {key: np.array(value, copy=True) for key, value in direct_obs.items()}
        reset_state_direct = np.array(direct.unwrapped.env.sim.get_state().flatten(), copy=True)
        direct_next, _, direct_terminated, direct_truncated, direct_info = direct.step(named_action)
        direct_next = {key: np.array(value, copy=True) for key, value in direct_next.items()}
        step_state_direct = np.array(direct.unwrapped.env.sim.get_state().flatten(), copy=True)
        direct_success = bool(direct_info.get("success", False) or direct.unwrapped.env._check_success())
    finally:
        direct.close()

    adapter = RoboCasaBenchmark(
        tasks=[task],
        split=split,
        seed=seed,
        camera_size=256,
        enable_render=enable_render,
    )
    try:
        await adapter.start_episode({"name": task, "episode_idx": 0})
        adapter_obs = await adapter.get_observation()
        adapter_raw = adapter._last_result.obs

        reset_state_adapter = np.asarray(adapter._env.unwrapped.env.sim.get_state().flatten())
        reset_state_l2 = float(np.linalg.norm(reset_state_direct - reset_state_adapter))
        reset_raw_diffs = _array_differences(direct_obs, adapter_raw, (*VIDEO_KEYS, *STATE_KEYS))

        await adapter.apply_action({"actions": flat_action})
        adapter_next = adapter._last_result.obs

        step_state_adapter = np.asarray(adapter._env.unwrapped.env.sim.get_state().flatten())
        step_state_l2 = float(np.linalg.norm(step_state_direct - step_state_adapter))
        step_raw_diffs = _array_differences(direct_next, adapter_next, (*VIDEO_KEYS, *STATE_KEYS))
        adapter_result = await adapter.get_result()

        state_tolerance = 1e-9
        # Separate EGL contexts can differ by one uint8 level even when the
        # MuJoCo state and camera contract are identical.
        image_tolerance = 1.0
        direct_metadata = _array_metadata(direct_next, (*VIDEO_KEYS, *STATE_KEYS))
        adapter_metadata = _array_metadata(adapter_next, (*VIDEO_KEYS, *STATE_KEYS))
        passed = bool(
            reset_state_l2 <= state_tolerance
            and step_state_l2 <= state_tolerance
            and max((reset_raw_diffs[key] for key in STATE_KEYS), default=0.0) <= state_tolerance
            and max((step_raw_diffs[key] for key in STATE_KEYS), default=0.0) <= state_tolerance
            and max((reset_raw_diffs[key] for key in VIDEO_KEYS), default=0.0) <= image_tolerance
            and max((step_raw_diffs[key] for key in VIDEO_KEYS), default=0.0) <= image_tolerance
            and direct_metadata == adapter_metadata
            and adapter_result["success"] == direct_success
            and not direct_terminated
            and not direct_truncated
            and ACTION_COMPONENTS == REFERENCE_ACTION_COMPONENTS
            and set(adapter_obs["images"]) == set(VIDEO_KEYS)
            and set(adapter_obs["state"]) == set(STATE_KEYS)
        )
        return {
            "pass": passed,
            "task": task,
            "split": split,
            "seed": seed,
            "enable_render": enable_render,
            "action_dim": ACTION_DIM,
            "action_order_match": ACTION_COMPONENTS == REFERENCE_ACTION_COMPONENTS,
            "reset_state_l2": reset_state_l2,
            "step_state_l2": step_state_l2,
            "reset_raw_max_abs_diff": reset_raw_diffs,
            "step_raw_max_abs_diff": step_raw_diffs,
            "array_metadata_match": direct_metadata == adapter_metadata,
            "state_tolerance": state_tolerance,
            "image_tolerance": image_tolerance,
            "direct_success": direct_success,
            "adapter_result": adapter_result,
            "upstream": UPSTREAM_PROVENANCE,
        }
    finally:
        adapter.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="KettleBoiling")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-render", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = asyncio.run(compare(args.task, args.split, args.seed, not args.disable_render))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
