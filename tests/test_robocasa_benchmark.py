from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import vla_eval.benchmarks.robocasa.benchmark as robocasa_module
from vla_eval.benchmarks.robocasa.benchmark import (
    ACTION_COMPONENTS,
    ACTION_DIM,
    STATE_KEYS,
    VIDEO_KEYS,
    RoboCasaBenchmark,
    decode_panda_omron_action,
)
from vla_eval.model_servers.robocasa_groot import (
    RoboCasaGR00TN15ModelServer,
    build_policy_observation,
    flatten_policy_actions,
)


def _raw_observation() -> dict[str, Any]:
    obs: dict[str, Any] = {
        key: np.full((4, 5, 3), index, dtype=np.uint8) for index, key in enumerate(VIDEO_KEYS, start=1)
    }
    for index, key in enumerate(STATE_KEYS, start=1):
        obs[key] = np.full((index,), index, dtype=np.float32)
    obs["annotation.human.task_description"] = "boil the kettle"
    return obs


class _FakeTaskEnv:
    def __init__(self) -> None:
        self.success = False

    def _check_success(self) -> bool:
        return self.success


class _FakeEnv:
    def __init__(self, obs: dict) -> None:
        self.obs = obs
        self.reset_seeds: list[int | None] = []
        self.actions: list[dict[str, np.ndarray]] = []
        self.closed = False
        self.unwrapped = SimpleNamespace(env=_FakeTaskEnv())

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        return self.obs, {}

    def step(self, action):
        self.actions.append(action)
        success = self.unwrapped.env.success
        return self.obs, float(success), False, False, {"success": success}

    def render(self):
        return self.obs[VIDEO_KEYS[0]]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def registry(monkeypatch):
    value = {
        "atomic_seen": ["AtomicTask"],
        "composite_seen": ["KettleBoiling"],
        "composite_unseen": ["ArrangeBreadBasket"],
    }
    monkeypatch.setattr(robocasa_module, "_task_registry", lambda: value)
    monkeypatch.setattr(
        robocasa_module,
        "_task_horizon",
        lambda task: {"AtomicTask": 100, "KettleBoiling": 1500, "ArrangeBreadBasket": 4350}[task],
    )
    return value


def test_decode_panda_omron_action_preserves_every_component():
    assert ACTION_COMPONENTS == (
        ("action.base_motion", 4),
        ("action.control_mode", 1),
        ("action.end_effector_position", 3),
        ("action.end_effector_rotation", 3),
        ("action.gripper_close", 1),
    )
    raw = np.arange(ACTION_DIM, dtype=np.float32)
    named = decode_panda_omron_action({"actions": raw})
    offset = 0
    for key, width in ACTION_COMPONENTS:
        np.testing.assert_array_equal(named[key], raw[offset : offset + width])
        offset += width


def test_decode_panda_omron_action_fails_instead_of_padding():
    with pytest.raises(ValueError, match="12-D"):
        decode_panda_omron_action({"actions": np.zeros(7)})
    with pytest.raises(ValueError, match="requires"):
        decode_panda_omron_action({})


def test_official_task_registry_and_horizons_are_used(registry):
    benchmark = RoboCasaBenchmark()
    assert benchmark.get_tasks() == [
        {"name": "AtomicTask", "suite": "atomic_seen"},
        {"name": "KettleBoiling", "suite": "composite_seen"},
        {"name": "ArrangeBreadBasket", "suite": "composite_unseen"},
    ]
    metadata = benchmark.get_metadata()
    assert metadata["max_steps"] == 4350
    assert metadata["environment_split"] == "target"
    assert metadata["environment_seed"] == 0
    assert metadata["task_horizon_source"].endswith("get_task_horizon")
    assert metadata["upstream"]["robocasa"]["version"] == "1.0.1"


def test_explicit_task_must_belong_to_official_target50(registry):
    benchmark = RoboCasaBenchmark(tasks=["NotOfficial"])
    with pytest.raises(ValueError, match="target50"):
        benchmark.get_tasks()


def test_make_env_uses_registered_official_gym_wrapper(monkeypatch):
    calls = []
    gym = ModuleType("gymnasium")
    setattr(gym, "make", lambda *args, **kwargs: calls.append((args, kwargs)) or object())
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.setattr(robocasa_module, "_validate_runtime_versions", lambda: None)

    benchmark = RoboCasaBenchmark(camera_size=128, split="target", enable_render=False)
    benchmark._make_env("KettleBoiling")
    assert calls == [
        (
            ("robocasa/KettleBoiling",),
            {
                "split": "target",
                "enable_render": False,
                "camera_widths": 128,
                "camera_heights": 128,
            },
        )
    ]


def test_runtime_version_mismatch_fails_fast(monkeypatch):
    monkeypatch.setattr(
        robocasa_module,
        "_runtime_versions",
        lambda: {"robocasa": "1.0.0", "robosuite": "1.5.2"},
    )
    with pytest.raises(RuntimeError, match="robocasa.*1.0.0"):
        robocasa_module._validate_runtime_versions()


@pytest.mark.anyio
async def test_episode_uses_seeded_reset_named_action_and_task_horizon(monkeypatch, registry):
    obs = _raw_observation()
    env = _FakeEnv(obs)
    benchmark = RoboCasaBenchmark(tasks=["KettleBoiling"], seed=10, max_steps=1)
    monkeypatch.setattr(benchmark, "_make_env", lambda task_name: env)

    await benchmark.start_episode({"name": "KettleBoiling", "suite": "composite_seen", "episode_idx": 3})
    assert env.reset_seeds == [13]
    canonical = await benchmark.get_observation()
    assert canonical["task_description"] == "boil the kettle"
    np.testing.assert_array_equal(canonical["images"][VIDEO_KEYS[0]], obs[VIDEO_KEYS[0]])
    assert canonical["state"].keys() == set(STATE_KEYS)

    await benchmark.apply_action({"actions": np.arange(ACTION_DIM)})
    assert await benchmark.is_done()
    assert (await benchmark.get_result()) == {"success": False, "time_limit_reached": True}
    assert set(env.actions[0]) == {key for key, _ in ACTION_COMPONENTS}


def test_policy_observation_and_action_helpers_preserve_named_modalities():
    obs = {
        "images": {key: value for key, value in _raw_observation().items() if key in VIDEO_KEYS},
        "state": {key: value for key, value in _raw_observation().items() if key in STATE_KEYS},
        "task_description": "boil the kettle",
    }
    policy_obs = build_policy_observation([obs, obs])
    assert policy_obs[VIDEO_KEYS[0]].shape == (2, 1, 4, 5, 3)
    assert policy_obs[STATE_KEYS[0]].shape == (2, 1, 1)
    assert policy_obs["annotation.human.task_description"].tolist() == ["boil the kettle", "boil the kettle"]

    actions = {}
    expected_parts = []
    for index, (key, width) in enumerate(ACTION_COMPONENTS):
        value = np.full((2, 16, width), index, dtype=np.float32)
        actions[key] = value
        expected_parts.append(value)
    flat = flatten_policy_actions(actions, batch_size=2)
    assert flat.shape == (2, 16, ACTION_DIM)
    np.testing.assert_array_equal(flat, np.concatenate(expected_parts, axis=-1))


def test_groot_server_seeds_policy_rng_and_reports_seed(monkeypatch):
    calls = []
    monkeypatch.setattr(RoboCasaGR00TN15ModelServer, "_load_policy", lambda _self: object())
    monkeypatch.setattr(
        RoboCasaGR00TN15ModelServer,
        "_seed_policy_rng",
        lambda self: calls.append(self.seed),
    )
    server = RoboCasaGR00TN15ModelServer("checkpoint", "revision", seed=17)
    assert calls == [17]
    assert server.get_metadata()["policy_seed"] == 17
