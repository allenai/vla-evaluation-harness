#!/usr/bin/env python3
"""Compare the vla-eval RC365 adapter with the reference official wrapper."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

QUALIFICATION_TASKS = (
    "RinseSinkBasin",
    "PanTransfer",
    "GetToastedBread",
    "WaffleReheat",
    "StirVegetables",
    "HeatKebabSandwich",
)
SEED = 0
STEPS = 32
ATOL = 1e-12
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACTIONS = REPO_ROOT / "scripts/data/rc365_qualification_actions_seed0.json"
DEFAULT_OUTPUT = REPO_ROOT / "results/rc365_adapter_qualification_parity.json"


class _PathProxy:
    def __init__(self, temporary_directory: Path) -> None:
        self._temporary_directory = temporary_directory
        self._shadows: dict[str, Path] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(os.path, name)

    def join(self, *parts: str) -> str:
        path = os.path.join(*parts)
        if len(parts) == 2 and parts[-1].endswith(f"_{os.getpid()}.xml"):
            source_directory = os.path.abspath(parts[0])
            shadow = self._shadows.get(source_directory)
            if shadow is None:
                digest = hashlib.sha256(source_directory.encode()).hexdigest()[:16]
                shadow = self._temporary_directory / digest
                shadow.mkdir()
                for child in Path(source_directory).iterdir():
                    (shadow / child.name).symlink_to(child, target_is_directory=child.is_dir())
                self._shadows[source_directory] = shadow
            return str(shadow / parts[-1])
        return path


class _OSProxy:
    def __init__(self, temporary_directory: Path) -> None:
        self.path = _PathProxy(temporary_directory)

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


def _redirect_generated_asset_xml(temporary_directory: Path) -> None:
    """Keep RoboCasa's generated XML out of a read-only package directory."""
    import robocasa.models.objects.objects as objects_module

    proxy = _OSProxy(temporary_directory)
    objects_module.os = proxy
    try:
        import robocasa.utils.model_zoo.mjcf_obj as model_zoo_module
    except ModuleNotFoundError:
        return
    model_zoo_module.os = proxy


def _load_actions(path: Path) -> tuple[list[Any], str]:
    from vla_eval.benchmarks.robocasa.rc365 import ACTION_COMPONENTS, ACTION_DIM

    raw = path.read_bytes()
    value = json.loads(raw)
    actions = value.get("actions") if isinstance(value, Mapping) else None
    if not isinstance(actions, list) or len(actions) != STEPS:
        raise ValueError(f"action fixture must contain exactly {STEPS} actions")

    import numpy as np

    named_actions = []
    for index, action in enumerate(actions):
        flat = np.asarray(action, dtype=np.float64)
        if flat.shape != (ACTION_DIM,):
            raise ValueError(f"action {index} has shape {flat.shape}, expected {(ACTION_DIM,)}")
        named: dict[str, Any] = {}
        offset = 0
        for key, width in ACTION_COMPONENTS:
            named[key] = flat[offset : offset + width].copy()
            offset += width
        named_actions.append((flat, named))
    return named_actions, hashlib.sha256(raw).hexdigest()


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    copied = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _simulator_state(sim_env: Any) -> dict[str, Any]:
    import numpy as np

    data = sim_env.sim.data
    result = {
        "qpos": np.asarray(data.qpos).copy(),
        "qvel": np.asarray(data.qvel).copy(),
        "ctrl": np.asarray(data.ctrl).copy(),
        "time": np.asarray([data.time], dtype=np.float64),
    }
    if getattr(data, "act", None) is not None:
        result["act"] = np.asarray(data.act).copy()
    return result


def _schema(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    import numpy as np

    return {
        key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for key, value in sorted(observation.items())
    }


def _mapping_deviation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    left_keys = set(left)
    right_keys = set(right)
    keys_match = left_keys == right_keys
    schema_match = keys_match and _schema(left) == _schema(right)
    values_match = keys_match
    max_abs = 0.0
    max_l2 = 0.0
    per_key: dict[str, float | None] = {}
    for key in sorted(left_keys | right_keys):
        if key not in left or key not in right:
            per_key[key] = None
            values_match = False
            continue
        left_value = np.asarray(left[key])
        right_value = np.asarray(right[key])
        if left_value.shape != right_value.shape or left_value.dtype != right_value.dtype:
            per_key[key] = None
            values_match = False
            continue
        if np.issubdtype(left_value.dtype, np.number) or np.issubdtype(left_value.dtype, np.bool_):
            difference = np.asarray(left_value, dtype=np.float64) - np.asarray(right_value, dtype=np.float64)
            key_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
            key_l2 = float(np.linalg.norm(difference.ravel()))
            per_key[key] = key_abs
            max_abs = max(max_abs, key_abs)
            max_l2 = max(max_l2, key_l2)
            values_match = values_match and bool(np.allclose(left_value, right_value, rtol=0.0, atol=ATOL))
        else:
            equal = bool(np.array_equal(left_value, right_value))
            per_key[key] = 0.0 if equal else None
            values_match = values_match and equal
    return {
        "keys_match": keys_match,
        "schema_match": schema_match,
        "values_match": values_match,
        "max_abs": max_abs,
        "max_l2": max_l2,
        "per_key_max_abs": per_key,
    }


def _max_trajectory_deviation(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(left) != len(right):
        return {
            "keys_match": False,
            "schema_match": False,
            "values_match": False,
            "max_abs": None,
            "max_l2": None,
            "steps_compared": min(len(left), len(right)),
        }
    comparisons = [_mapping_deviation(a, b) for a, b in zip(left, right)]
    return {
        "keys_match": all(item["keys_match"] for item in comparisons),
        "schema_match": all(item["schema_match"] for item in comparisons),
        "values_match": all(item["values_match"] for item in comparisons),
        "max_abs": max(item["max_abs"] for item in comparisons),
        "max_l2": max(item["max_l2"] for item in comparisons),
        "steps_compared": len(comparisons),
    }


def _run_official(task: str, named_actions: Sequence[Any]) -> dict[str, Any]:
    import numpy as np

    from rc365_s2b.environments import OfficialRoboCasaEnv

    random.seed(SEED)
    np.random.seed(SEED)
    env = OfficialRoboCasaEnv(task, SEED)
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        first_obs, _ = env.reset(seed=SEED)
        first_state = _simulator_state(env.sim_env)
        random.seed(SEED)
        np.random.seed(SEED)
        second_obs, _ = env.reset(seed=SEED)
        second_state = _simulator_state(env.sim_env)
        observations = [_copy_observation(second_obs)]
        states = [second_state]
        for _, named in named_actions:
            observation, _, _, _, _ = env.step(named)
            observations.append(_copy_observation(observation))
            states.append(_simulator_state(env.sim_env))
        return {
            "first_observation": _copy_observation(first_obs),
            "second_observation": _copy_observation(second_obs),
            "first_state": first_state,
            "second_state": second_state,
            "observations": observations,
            "states": states,
        }
    finally:
        env.close()


def _run_adapter(task: str, named_actions: Sequence[Any]) -> dict[str, Any]:
    import numpy as np

    from vla_eval.benchmarks.robocasa.rc365 import RoboCasa365Benchmark
    from vla_eval.recording import NullEpisodeRecorder

    random.seed(SEED)
    np.random.seed(SEED)
    benchmark = RoboCasa365Benchmark(
        tasks=[task],
        split="pretrain",
        seed=SEED,
        camera_size=256,
    )
    benchmark._recorder = NullEpisodeRecorder()
    task_spec = {"name": task, "episode_idx": 0}
    try:
        first_obs = benchmark.reset(task_spec)
        first_state = _simulator_state(benchmark._env.unwrapped.env)
        second_obs = benchmark.reset(task_spec)
        second_state = _simulator_state(benchmark._env.unwrapped.env)
        observations = [_copy_observation(second_obs)]
        states = [second_state]
        for flat, _ in named_actions:
            result = benchmark.step({"actions": flat})
            observations.append(_copy_observation(result.obs))
            states.append(_simulator_state(benchmark._env.unwrapped.env))
        return {
            "first_observation": _copy_observation(first_obs),
            "second_observation": _copy_observation(second_obs),
            "first_state": first_state,
            "second_state": second_state,
            "observations": observations,
            "states": states,
        }
    finally:
        benchmark.cleanup()


def _task_report(task: str, named_actions: Sequence[Any]) -> dict[str, Any]:
    official = _run_official(task, named_actions)
    adapter = _run_adapter(task, named_actions)

    official_reset_obs = _mapping_deviation(official["first_observation"], official["second_observation"])
    official_reset_state = _mapping_deviation(official["first_state"], official["second_state"])
    adapter_reset_obs = _mapping_deviation(adapter["first_observation"], adapter["second_observation"])
    adapter_reset_state = _mapping_deviation(adapter["first_state"], adapter["second_state"])
    cross_first_reset_obs = _mapping_deviation(official["first_observation"], adapter["first_observation"])
    cross_first_reset_state = _mapping_deviation(official["first_state"], adapter["first_state"])
    cross_reset_obs = _mapping_deviation(official["second_observation"], adapter["second_observation"])
    cross_reset_state = _mapping_deviation(official["second_state"], adapter["second_state"])
    trajectory_obs = _max_trajectory_deviation(official["observations"], adapter["observations"])
    trajectory_state = _max_trajectory_deviation(official["states"], adapter["states"])
    final_state = _mapping_deviation(official["states"][-1], adapter["states"][-1])

    official_reset_idempotent = official_reset_obs["values_match"] and official_reset_state["values_match"]
    adapter_reset_idempotent = adapter_reset_obs["values_match"] and adapter_reset_state["values_match"]
    checks = {
        "reset_determinism_matches_reference": official_reset_idempotent == adapter_reset_idempotent
        and cross_first_reset_obs["values_match"]
        and cross_first_reset_state["values_match"]
        and cross_reset_obs["values_match"]
        and cross_reset_state["values_match"],
        "reset_observation_keys_match": cross_reset_obs["keys_match"],
        "reset_observation_shapes_and_dtypes_match": cross_reset_obs["schema_match"],
        "reset_observation_values_match": cross_reset_obs["values_match"],
        "reset_simulator_state_matches": cross_reset_state["values_match"],
        "trajectory_observations_match": trajectory_obs["values_match"],
        "trajectory_simulator_states_match": trajectory_state["values_match"],
        "final_simulator_state_matches": final_state["values_match"],
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "same_instance_reset_diagnostics": {
            "official_resets_equal": official_reset_idempotent,
            "adapter_resets_equal": adapter_reset_idempotent,
        },
        "observation_schema": _schema(official["second_observation"]),
        "max_deviations": {
            "official_reset_observation_abs": official_reset_obs["max_abs"],
            "official_reset_state_abs": official_reset_state["max_abs"],
            "adapter_reset_observation_abs": adapter_reset_obs["max_abs"],
            "adapter_reset_state_abs": adapter_reset_state["max_abs"],
            "cross_first_reset_observation_abs": cross_first_reset_obs["max_abs"],
            "cross_first_reset_state_abs": cross_first_reset_state["max_abs"],
            "cross_reset_observation_abs": cross_reset_obs["max_abs"],
            "cross_reset_state_abs": cross_reset_state["max_abs"],
            "trajectory_observation_abs": trajectory_obs["max_abs"],
            "trajectory_state_abs": trajectory_state["max_abs"],
            "trajectory_state_l2": trajectory_state["max_l2"],
            "final_state_abs": final_state["max_abs"],
            "final_state_l2": final_state["max_l2"],
        },
        "trajectory_steps_compared": trajectory_state["steps_compared"] - 1,
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-src",
        type=Path,
        required=True,
        help="Path to the read-only rc365-s2b src directory",
    )
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(args.reference_src.resolve()))
    sys.path.insert(0, str((REPO_ROOT / "src").resolve()))
    named_actions, action_sha256 = _load_actions(args.actions)
    try:
        action_path = str(args.actions.resolve().relative_to(REPO_ROOT))
    except ValueError:
        action_path = str(args.actions.resolve())

    report: dict[str, Any] = {
        "schema_version": "rc365-adapter-qualification-parity-v1",
        "seed": SEED,
        "process_rng_reseeded_before_each_environment_and_reset": True,
        "tasks": {},
        "action_sequence": {
            "path": action_path,
            "sha256": action_sha256,
            "steps": len(named_actions),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT"),
            "robocasa": _version("robocasa"),
            "robosuite": _version("robosuite"),
        },
        "deferred": [
            {
                "check": "full GR00T and MLLM policy inference",
                "reason": "requires GPU or Slurm and is outside environment-adapter parity",
            }
        ],
    }

    with tempfile.TemporaryDirectory(prefix="rc365-parity-xml-") as temp_dir:
        try:
            _redirect_generated_asset_xml(Path(temp_dir))
        except Exception:
            report["setup_error"] = traceback.format_exc()
        else:
            for task in QUALIFICATION_TASKS:
                try:
                    report["tasks"][task] = _task_report(task, named_actions)
                except Exception:
                    report["tasks"][task] = {"pass": False, "error": traceback.format_exc()}

    report["overall_pass"] = len(report["tasks"]) == len(QUALIFICATION_TASKS) and all(
        result.get("pass") is True for result in report["tasks"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
