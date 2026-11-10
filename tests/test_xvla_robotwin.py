"""X-VLA RoboTwin proprio/action encoding for the 2toINF and XPolicyLab profiles (no torch, no model)."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.benchmarks.robotwin.benchmark import RoboTwinBenchmark
from vla_eval.model_servers import xvla
from vla_eval.specs import check_specs


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the rotation matrix of ``angle`` radians about ``axis`` (Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def _quat_wxyz(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the wxyz quaternion of ``angle`` radians about ``axis``."""
    k = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * k)])


_LEFT = (np.array([1.0, 2.0, 3.0]), 0.7)
_RIGHT = (np.array([-0.5, 0.3, 1.0]), -1.2)


def _state(quat_order: str) -> np.ndarray:
    """Build a 16-D RoboTwin end-pose state with quaternions in ``quat_order``."""
    rows = []
    for (axis, angle), pos, grip in ((_LEFT, [0.1, 0.2, 0.3], 0.25), (_RIGHT, [-0.1, 0.4, 0.9], 0.8)):
        q = _quat_wxyz(axis, angle)
        rows += [*pos, *(q if quat_order == "wxyz" else q[[1, 2, 3, 0]]), grip]
    return np.array(rows, dtype=np.float64)


def _xpolicylab_rot6d(rot: np.ndarray) -> np.ndarray:
    """XPolicyLab ``quat_to_rotate6d``: the first two columns, flattened row-major."""
    return rot[:, :2].reshape(6)


def test_xpolicylab_proprio_matches_reference() -> None:
    """XPolicyLab proprio: wxyz quaternion to row-major rot6d, raw gripper passed through."""
    proprio = xvla._robotwin_proprio(_state("wxyz"), None, quat_wxyz=True, raw_gripper=True)
    for arm, (axis, angle), pos, grip in ((0, _LEFT, [0.1, 0.2, 0.3], 0.25), (1, _RIGHT, [-0.1, 0.4, 0.9], 0.8)):
        seg = proprio[10 * arm : 10 * arm + 10]
        np.testing.assert_allclose(seg[:3], pos, atol=1e-6)
        np.testing.assert_allclose(seg[3:9], _xpolicylab_rot6d(_rotation(axis, angle)), atol=1e-5)
        assert seg[9] == pytest.approx(grip)


def test_xpolicylab_action_is_wxyz_with_raw_gripper() -> None:
    """XPolicyLab action decode: wxyz quaternion, raw gripper, re-encodes to the same pose."""
    proprio = xvla._robotwin_proprio(_state("wxyz"), None, quat_wxyz=True, raw_gripper=True)
    actions = np.stack([proprio, proprio]).astype(np.float32)
    actions[:, 9], actions[:, 19] = 0.6, 0.9
    ee16 = xvla._ee6d_to_robotwin_ee16(actions, 0.7, quat_wxyz=True, raw_gripper=True)
    assert ee16.shape == (2, 16)
    for arm, (axis, angle) in ((0, _LEFT), (1, _RIGHT)):
        quat = ee16[0, 8 * arm + 3 : 8 * arm + 7]
        ref = _quat_wxyz(axis, angle)
        np.testing.assert_allclose(quat * np.sign(quat[0] * ref[0]), ref, atol=1e-5)
    assert ee16[0, 7] == pytest.approx(0.6)
    assert ee16[0, 15] == pytest.approx(0.9)
    # Round trip: the decoded action re-encodes to the same proprio poses
    again = xvla._robotwin_proprio(ee16[0], None, quat_wxyz=True, raw_gripper=True)
    np.testing.assert_allclose(again[3:9], proprio[3:9], atol=1e-5)
    np.testing.assert_allclose(again[13:19], proprio[13:19], atol=1e-5)


def test_default_profile_keeps_xyzw_and_binarized_gripper() -> None:
    """The 2toINF profile still reads xyzw, sends ``1-2g`` and binarizes the output gripper."""
    proprio = xvla._robotwin_proprio(_state("xyzw"), None)
    np.testing.assert_allclose(proprio[3:9], _xpolicylab_rot6d(_rotation(*_LEFT)), atol=1e-5)
    assert proprio[9] == pytest.approx(1 - 2 * 0.25)
    assert proprio[19] == pytest.approx(1 - 2 * 0.8)
    actions = proprio[None].copy()
    actions[0, 9], actions[0, 19] = 0.6, 0.9
    ee16 = xvla._ee6d_to_robotwin_ee16(actions, 0.7)
    ref = _quat_wxyz(*_LEFT)[[1, 2, 3, 0]]
    np.testing.assert_allclose(ee16[0, 3:7] * np.sign(ee16[0, 6] * ref[3]), ref, atol=1e-5)
    assert ee16[0, 7] == 1.0
    assert ee16[0, 15] == -1.0


def test_commanded_pose_replaces_measured() -> None:
    """A commanded pose overrides the measured pose but not the measured gripper."""
    measured = _state("xyzw")
    commanded = _state("xyzw")
    commanded[0:3] = [9.0, 9.0, 9.0]
    proprio = xvla._robotwin_proprio(measured, commanded)
    np.testing.assert_allclose(proprio[:3], [9.0, 9.0, 9.0])
    assert proprio[9] == pytest.approx(1 - 2 * measured[7])


def test_zero_norm_quaternion_raises() -> None:
    """A zero-norm state quaternion raises instead of producing a garbage rotation."""
    state = _state("wxyz")
    state[3:7] = 0.0
    with pytest.raises(ValueError, match="Zero-norm"):
        xvla._robotwin_proprio(state, None, quat_wxyz=True, raw_gripper=True)


def test_xpolicylab_profile_flags() -> None:
    """The robotwin_xpolicylab profile and its HELLO observation params are as designed."""
    profile = xvla._get_profile("robotwin_xpolicylab")
    assert profile.image_keys == ("head_camera",)
    assert profile.quat_wxyz and profile.raw_gripper
    assert not profile.use_predicted_proprio
    assert not profile.instruction_from_task_name
    assert profile.output_action_dim == 16
    assert xvla._PROFILE_OBS_PARAMS["robotwin_xpolicylab"] == {"send_state": True, "action_type": "ee"}
    old = xvla._get_profile("robotwin")
    assert not old.quat_wxyz and not old.raw_gripper


@pytest.mark.parametrize("profile_name", ["robotwin", "robotwin_xpolicylab"])
def test_specs_match_robotwin_ee(profile_name: str) -> None:
    """Server specs match RoboTwinBenchmark with ``action_type=ee`` and ``send_state``."""
    profile = xvla._get_profile(profile_name)
    server = object.__new__(xvla.XVLAModelServer)
    server.output_action_dim = profile.output_action_dim
    server._image_keys = profile.image_keys
    server.use_predicted_proprio = profile.use_predicted_proprio
    server._euler_offset = None
    bench = RoboTwinBenchmark(task_name="grab_roller", action_type="ee", send_state=True)
    assert (
        check_specs(
            server.get_action_spec(),
            bench.get_action_spec(),
            server.get_observation_spec(),
            bench.get_observation_spec(),
        )
        == []
    )
