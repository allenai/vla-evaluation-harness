"""GR00T observation handling per embodiment layout (no Isaac-GR00T needed: the policy is faked)."""

from __future__ import annotations

import types

import numpy as np
import pytest

from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.groot import GR00TModelServer
from vla_eval.rotation import matrix_to_euler_xyz, quat_wxyz_to_xyzw

LIBERO_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LIBERO_DIMS = {k: 1 for k in LIBERO_KEYS[:6]} | {"gripper": 2}
# LIBEROBenchmark proprio: [eef_pos3, axis_angle3, gripper_qpos2]
LIBERO_STATE = np.array([0.1, 0.2, 1.0, 3.13, 0.14, 0.06, 0.039, -0.039], dtype=np.float32)


def _server(tag: str, video: list[str], state_dims: dict[str, int], **kwargs) -> tuple[GR00TModelServer, dict]:
    """Server with a fake policy; the returned dict captures the observation passed to it."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(GR00TModelServer, "_init_policy", lambda self: None)
        srv = GR00TModelServer(embodiment_tag=tag, **kwargs)
    keys = {"video": video, "state": list(state_dims), "action": ["a"]}
    srv._modality_config = {k: types.SimpleNamespace(modality_keys=v) for k, v in keys.items()}
    srv._state_dims = state_dims
    srv._language_key = "lang"
    captured: dict = {}

    def get_action(obs):
        captured.update(obs)
        return {"a": np.zeros((1, 16, 7), dtype=np.float32)}, None

    srv._policy = types.SimpleNamespace(get_action=get_action)
    return srv, captured


def _predict(srv: GR00TModelServer, images: dict | None, state: np.ndarray) -> None:
    srv.predict_batch([{"images": images, "states": state, "task_description": "t"}], [SessionContext("s", "e")])


def _img(value: int = 0) -> np.ndarray:
    return np.full((256, 256, 3), value, dtype=np.uint8)


@pytest.mark.parametrize("tag", ["LIBERO_PANDA", "libero_panda", "EmbodimentTag.LIBERO_PANDA"])
def test_libero_params_and_spec(tag: str) -> None:
    srv, _ = _server(tag, [], {})
    params = srv.get_observation_params()
    assert params["send_wrist_image"] and params["quat_no_antipodal"]
    assert "max_episode_steps" not in params
    assert srv.get_action_spec()["rotation"].format == "axis_angle"


def test_simpler_params_and_spec_unchanged() -> None:
    srv, _ = _server("OXE_GOOGLE", [], {})
    params = srv.get_observation_params()
    assert params["max_episode_steps"] == 300 and "send_wrist_image" not in params
    assert srv.get_action_spec()["rotation"].format == "euler_xyz"


def test_libero_passes_state_and_both_views_through() -> None:
    srv, obs = _server("LIBERO_PANDA", ["image", "wrist_image"], LIBERO_DIMS)
    _predict(srv, {"agentview": _img(0), "wrist": _img(7)}, LIBERO_STATE)
    assert np.allclose([obs["state"][k][0, 0, 0] for k in LIBERO_KEYS[:6]], LIBERO_STATE[:6])
    assert np.allclose(obs["state"]["gripper"][0, 0], LIBERO_STATE[6:])
    assert obs["video"]["wrist_image"][0, 0, 0, 0, 0] == 7


@pytest.mark.parametrize("bridge", [False, True], ids=["google", "widowx"])
def test_simpler_state_rewrite_unchanged(bridge: bool) -> None:
    tag = "OXE_WIDOWX" if bridge else "OXE_GOOGLE"
    srv, obs = _server(tag, ["image"], {"eef": 7, "grip": 1}, bridge_rotation=bridge)
    eef_pos = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float32)  # identity quat (wxyz)
    _predict(srv, {"a": _img()}, eef_pos)
    if bridge:
        rpy = matrix_to_euler_xyz(GR00TModelServer._BRIDGE_DEFAULT_ROT.T)
        expected = [*eef_pos[:3], *rpy, 0.0, eef_pos[7]]
    else:
        expected = [*eef_pos[:3], *quat_wxyz_to_xyzw(eef_pos[3:7]), 1.0 - eef_pos[7]]
    assert np.allclose([*obs["state"]["eef"][0, 0], *obs["state"]["grip"][0, 0]], expected)


def test_missing_view_raises() -> None:
    srv, _ = _server("LIBERO_PANDA", ["image", "wrist_image"], LIBERO_DIMS)
    with pytest.raises(RuntimeError, match="needs views"):
        _predict(srv, {"agentview": _img()}, LIBERO_STATE)


@pytest.mark.parametrize("tag", ["LIBERO_PANDA", "OXE_GOOGLE"])
def test_state_dim_mismatch_raises(tag: str) -> None:
    srv, _ = _server(tag, ["image", "wrist_image"], LIBERO_DIMS)
    with pytest.raises(RuntimeError, match="dims|eef_pos"):
        _predict(srv, {"a": _img(), "b": _img()}, LIBERO_STATE[:7])
