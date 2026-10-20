"""Embodiment dispatch regression tests for the GR00T model server.

These pin the three adapter bugs behind GR00T N1.6 scoring ~0 on LIBERO: the
SimplerEnv state rewrite leaking into other embodiments, LIBERO needing two
views plus a robosuite-style axis-angle, and missing/short observations being
silently padded instead of raising.

Everything gr00t-specific in ``groot.py`` is imported lazily inside
``_init_policy()``, so the server can be constructed here without
Isaac-GR00T installed.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.groot import GR00TModelServer

# libero_panda's state contract: 7 keys, 8 dims total (gripper is a 2-finger qpos).
LIBERO_STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LIBERO_STATE_DIMS = {k: 1 for k in LIBERO_STATE_KEYS[:6]} | {"gripper": 2}
# What LIBEROBenchmark._proprio_state() sends: [eef_pos3, quat->axisangle3, gripper_qpos2].
LIBERO_PROPRIO = np.array([0.1, 0.2, 1.0, 3.13, 0.14, 0.06, 0.039, -0.039], dtype=np.float32)


def _ctx(n: int = 1) -> list[SessionContext]:
    return [SessionContext("s", "e") for _ in range(n)]


def _make(tag: str, **kwargs) -> GR00TModelServer:
    """Construct the server while skipping policy loading."""
    srv = GR00TModelServer.__new__(GR00TModelServer)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GR00TModelServer, "_init_policy", lambda self: None)
        GR00TModelServer.__init__(srv, embodiment_tag=tag, **kwargs)
    return srv


def _modality(video: list[str], state: list[str], action: list[str]) -> dict:
    return {
        k: types.SimpleNamespace(modality_keys=v) for k, v in (("video", video), ("state", state), ("action", action))
    }


def _wire(srv: GR00TModelServer, video: list[str], state: list[str], dims: dict[str, int]) -> dict:
    """Attach a fake policy; returns the dict capturing the observation."""
    srv._modality_config = _modality(video, state, ["x"])
    srv._state_dims = dims
    srv._language_key = "annotation.human.action.task_description"
    captured: dict = {}

    def _get_action(obs):
        captured["obs"] = obs
        return {"x": np.zeros((1, 16, 7), dtype=np.float32)}, None

    srv._policy = types.SimpleNamespace(get_action=_get_action)
    return captured


def _libero_obs(**overrides) -> dict:
    obs = {
        "images": {
            "agentview": np.zeros((256, 256, 3), dtype=np.uint8),
            "wrist": np.full((256, 256, 3), 7, dtype=np.uint8),
        },
        "states": LIBERO_PROPRIO,
        "task_description": "pick up the black bowl",
    }
    obs.update(overrides)
    return obs


@pytest.mark.parametrize("tag", ["LIBERO_PANDA", "libero_panda", "EmbodimentTag.LIBERO_PANDA"])
def test_libero_observation_params(tag: str) -> None:
    params = _make(tag).get_observation_params()
    assert params["send_state"] is True
    assert params["send_wrist_image"] is True, "without it wrist_image is an all-black pad"
    assert params["quat_no_antipodal"] is True, "trained on robosuite axis-angle, no antipodal fix"
    # LIBEROBenchmark's constructor does not take max_episode_steps; sending it only
    # gets dropped with a warning by the orchestrator.
    assert "max_episode_steps" not in params


def test_simpler_observation_params_unchanged() -> None:
    params = _make("OXE_GOOGLE").get_observation_params()
    assert params["max_episode_steps"] == 300
    assert "send_wrist_image" not in params
    assert "quat_no_antipodal" not in params


def test_extra_obs_params_still_override() -> None:
    srv = _make("LIBERO_PANDA", observation_params='{"send_wrist_image": false}')
    assert srv.get_observation_params()["send_wrist_image"] is False


def test_libero_state_passes_through_untransformed() -> None:
    srv = _make("LIBERO_PANDA", invert_gripper=True)
    captured = _wire(srv, ["image", "wrist_image"], LIBERO_STATE_KEYS, LIBERO_STATE_DIMS)
    srv.predict_batch([_libero_obs()], _ctx())

    state = captured["obs"]["state"]
    assert np.allclose([state[k][0, 0, 0] for k in LIBERO_STATE_KEYS[:6]], LIBERO_PROPRIO[:6])
    assert np.allclose(state["gripper"][0, 0], LIBERO_PROPRIO[6:8])


def test_libero_wrist_image_maps_to_second_video_key() -> None:
    srv = _make("LIBERO_PANDA")
    captured = _wire(srv, ["image", "wrist_image"], LIBERO_STATE_KEYS, LIBERO_STATE_DIMS)
    srv.predict_batch([_libero_obs()], _ctx())

    video = captured["obs"]["video"]
    assert set(video) == {"image", "wrist_image"}
    assert video["image"].shape == (1, 1, 256, 256, 3)
    assert video["wrist_image"][0, 0, 0, 0, 0] == 7, "agentview/wrist order must follow video_keys"


@pytest.mark.parametrize("images", [{"agentview": np.zeros((256, 256, 3), dtype=np.uint8)}, {}, None])
def test_missing_image_raises_instead_of_zero_fill(images) -> None:
    srv = _make("LIBERO_PANDA")
    _wire(srv, ["image", "wrist_image"], LIBERO_STATE_KEYS, LIBERO_STATE_DIMS)
    with pytest.raises(RuntimeError, match="needs 2 views"):
        srv.predict_batch([_libero_obs(images=images)], _ctx())


def test_libero_state_dim_mismatch_raises() -> None:
    srv = _make("LIBERO_PANDA")
    _wire(srv, ["image", "wrist_image"], LIBERO_STATE_KEYS, LIBERO_STATE_DIMS)
    with pytest.raises(RuntimeError, match="state dim mismatch"):
        srv.predict_batch([_libero_obs(states=LIBERO_PROPRIO[:7])], _ctx())


def test_simpler_state_transform_unchanged() -> None:
    """OXE_GOOGLE keeps the quaternion reorder + gripper inversion it was scored with."""
    from vla_eval.rotation import quat_wxyz_to_xyzw

    srv = _make("OXE_GOOGLE")
    captured = _wire(srv, ["image"], ["eef", "grip"], {"eef": 7, "grip": 1})
    eef_pos = np.array([0.3, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.2], dtype=np.float32)
    obs = {"images": {"a": np.zeros((224, 224, 3), dtype=np.uint8)}, "states": eef_pos, "task_description": "t"}
    srv.predict_batch([obs], _ctx())

    state = captured["obs"]["state"]
    assert np.allclose(state["eef"][0, 0, :3], eef_pos[:3])
    assert np.allclose(state["eef"][0, 0, 3:7], quat_wxyz_to_xyzw(eef_pos[3:7]))
    assert np.isclose(state["grip"][0, 0, 0], 1.0 - eef_pos[7])


def test_simpler_state_too_short_raises() -> None:
    srv = _make("OXE_GOOGLE")
    _wire(srv, ["image"], ["eef", "grip"], {"eef": 7, "grip": 1})
    obs = {"images": {"a": np.zeros((224, 224, 3), dtype=np.uint8)}, "states": np.zeros(5, dtype=np.float32)}
    with pytest.raises(RuntimeError, match="needs an 8-D eef_pos"):
        srv.predict_batch([obs], _ctx())


def test_rotation_spec_follows_embodiment() -> None:
    assert _make("LIBERO_PANDA").get_action_spec()["rotation"].format == "axis_angle"
    assert _make("OXE_GOOGLE").get_action_spec()["rotation"].format == "euler_xyz"
    assert _make("LIBERO_PANDA", invert_gripper=True).get_action_spec()["gripper"].format == "binary_close_positive"
