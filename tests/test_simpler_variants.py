"""Tests for SimplerEnvBenchmark variant_kwargs plumbing.

Validates that ``variant_kwargs`` are stored, surfaced in metadata, and
forwarded into ``simpler_env.make()`` on ``reset()`` -- all without
importing the real SimplerEnv package.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vla_eval.cli.config_loader import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


# -- Fake simpler_env module ------------------------------------------


class _FakeGymEnv:
    """Minimal stand-in for a ManiSkill env used in reset/step."""

    def __init__(self, **kwargs: Any) -> None:
        self._make_kwargs = kwargs
        self._closed = False

    # Gymnasium wrapper interface
    @property
    def unwrapped(self) -> _FakeGymEnv:
        return self

    def get_wrapper_attr(self, name: str) -> Any:
        return getattr(self, name)

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        obs = {"agent": {"base_pose": np.zeros(7), "qpos": np.zeros(8)}, "extra": {"tcp_pose": np.zeros(7)}}
        return obs, {}

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        return {}, 0.0, False, False, {}

    def close(self) -> None:
        self._closed = True

    def get_language_instruction(self) -> str:
        return "test task"


def _fake_make(task_name: str, **kwargs: Any) -> _FakeGymEnv:
    return _FakeGymEnv(task_name=task_name, **kwargs)


@pytest.fixture()
def fake_simpler_env(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``simpler_env`` module into ``sys.modules``."""
    mod = types.ModuleType("simpler_env")
    mod.make = _fake_make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "simpler_env", mod)
    return mod


# -- Unit tests (no env needed) --------------------------------------


def test_variant_kwargs_default_is_empty():
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(task_name="widowx_stack_cube")
    assert bench.variant_kwargs == {}


def test_variant_kwargs_stored():
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    vk = {"lr_switch": True}
    bench = SimplerEnvBenchmark(task_name="google_robot_pick_coke_can", variant_kwargs=vk)
    assert bench.variant_kwargs == {"lr_switch": True}


def test_variant_kwargs_none_gives_empty():
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(task_name="widowx_stack_cube", variant_kwargs=None)
    assert bench.variant_kwargs == {}


def test_variant_kwargs_in_metadata_when_present():
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(task_name="widowx_stack_cube")
    assert "variant_kwargs" not in bench.get_metadata()

    bench_va = SimplerEnvBenchmark(task_name="google_robot_pick_coke_can", variant_kwargs={"lr_switch": True})
    meta = bench_va.get_metadata()
    assert meta["variant_kwargs"] == {"lr_switch": True}


def test_variant_kwargs_accepts_model_ids():
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    vk = {"model_ids": "baked_apple_v2"}
    bench = SimplerEnvBenchmark(task_name="google_robot_place_in_closed_top_drawer", variant_kwargs=vk)
    assert bench.variant_kwargs["model_ids"] == "baked_apple_v2"


# -- reset() forwards variant_kwargs to simpler_env.make() -----------


def test_reset_forwards_variant_kwargs(fake_simpler_env: types.ModuleType):
    """reset() must merge variant_kwargs into the make() call."""
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(
        task_name="google_robot_pick_coke_can",
        variant_kwargs={"lr_switch": True, "obj_name": "coke_can"},
    )
    task: dict[str, Any] = {"name": "google_robot_pick_coke_can", "task_name": "google_robot_pick_coke_can"}
    bench.reset(task)

    assert bench._env is not None
    assert bench._env._make_kwargs["task_name"] == "google_robot_pick_coke_can"
    assert bench._env._make_kwargs["lr_switch"] is True
    assert bench._env._make_kwargs["obj_name"] == "coke_can"


def test_reset_merges_control_mode_with_variant_kwargs(fake_simpler_env: types.ModuleType):
    """control_mode and variant_kwargs must both appear in make() kwargs."""
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(
        task_name="widowx_stack_cube",
        control_mode="pd_ee_delta_pose",
        variant_kwargs={"extra_key": 42},
    )
    task: dict[str, Any] = {"name": "widowx_stack_cube", "task_name": "widowx_stack_cube"}
    bench.reset(task)

    assert bench._env._make_kwargs["control_mode"] == "pd_ee_delta_pose"
    assert bench._env._make_kwargs["extra_key"] == 42


def test_reset_empty_variant_kwargs_omits_extra_keys(fake_simpler_env: types.ModuleType):
    """With no variant_kwargs, make() receives only control_mode/max_episode_steps."""
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(task_name="widowx_stack_cube")
    task: dict[str, Any] = {"name": "widowx_stack_cube", "task_name": "widowx_stack_cube"}
    bench.reset(task)

    make_kw = bench._env._make_kwargs
    assert "task_name" in make_kw
    assert "lr_switch" not in make_kw
    assert "model_ids" not in make_kw


def test_reset_closes_previous_env(fake_simpler_env: types.ModuleType):
    """Calling reset() twice must close the first env before creating a new one."""
    from vla_eval.benchmarks.simpler.benchmark import SimplerEnvBenchmark

    bench = SimplerEnvBenchmark(task_name="widowx_stack_cube")
    task: dict[str, Any] = {"name": "widowx_stack_cube", "task_name": "widowx_stack_cube"}

    bench.reset(task)
    first_env = bench._env
    assert first_env is not None

    bench.reset(task)
    assert first_env._closed is True
    assert bench._env is not first_env


# -- Config-level tests ----------------------------------------------


def test_va_config_loads_and_has_benchmarks():
    config_path = REPO_ROOT / "configs" / "simpler_google_robot_va_tasks.yaml"
    assert config_path.exists(), f"VA config not found: {config_path}"

    data = load_config(str(config_path))
    assert "benchmarks" in data
    assert len(data["benchmarks"]) > 0


def test_va_config_import_strings_well_formed():
    config_path = REPO_ROOT / "configs" / "simpler_google_robot_va_tasks.yaml"
    data = load_config(str(config_path))

    for bench in data["benchmarks"]:
        import_path = bench.get("benchmark", "")
        assert ":" in import_path, f"Bad import string: {import_path!r}"
        module, _, cls_name = import_path.partition(":")
        assert module
        assert cls_name


def test_va_config_uses_variant_task_names():
    config_path = REPO_ROOT / "configs" / "simpler_google_robot_va_tasks.yaml"
    data = load_config(str(config_path))

    task_names = {b["params"]["task_name"] for b in data["benchmarks"]}

    expected = {
        # pick_coke_can_va (3)
        "google_robot_pick_horizontal_coke_can",
        "google_robot_pick_vertical_coke_can",
        "google_robot_pick_standing_coke_can",
        # move_near_va (2)
        "google_robot_move_near_v0",
        "google_robot_move_near_v1",
        # open_close_drawer_va (6)
        "google_robot_open_top_drawer",
        "google_robot_open_middle_drawer",
        "google_robot_open_bottom_drawer",
        "google_robot_close_top_drawer",
        "google_robot_close_middle_drawer",
        "google_robot_close_bottom_drawer",
        # place_apple_in_drawer_va (1, optional sub-score)
        "google_robot_place_apple_in_closed_top_drawer",
    }
    assert task_names == expected, (
        f"Unexpected task names: extra={task_names - expected}, missing={expected - task_names}"
    )
    assert len(task_names) == 12


def test_va_config_subnames_end_with_va():
    config_path = REPO_ROOT / "configs" / "simpler_google_robot_va_tasks.yaml"
    data = load_config(str(config_path))

    for bench in data["benchmarks"]:
        subname = bench.get("subname", "")
        assert subname.endswith("_va"), f"Subname {subname!r} should end with _va"
