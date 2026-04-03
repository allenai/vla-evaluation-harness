"""Tests for vla_eval.rotation — rotation conversion utilities."""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.rotation import (
    axisangle_to_matrix,
    axisangle_to_rot6d_contiguous,
    axisangle_to_rot6d_interleaved,
    euler_xyz_to_matrix,
    euler_xyz_to_rot6d_interleaved,
    gram_schmidt,
    matrix_to_euler_xyz,
    matrix_to_quat,
    matrix_to_rot6d_contiguous,
    matrix_to_rot6d_interleaved,
    quat_to_axisangle,
    quat_to_matrix,
    quat_to_rot6d_interleaved,
    rot6d_contiguous_to_matrix,
    rot6d_interleaved_to_euler_xyz,
    rot6d_interleaved_to_matrix,
    rot6d_interleaved_to_quat,
)

RNG = np.random.default_rng(42)


def _random_rotation_matrix(rng: np.random.Generator = RNG) -> np.ndarray:
    """Generate a random valid rotation matrix via QR decomposition."""
    A = rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure det(Q) = +1 (proper rotation, not reflection)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _random_quat(rng: np.random.Generator = RNG) -> np.ndarray:
    """Random unit quaternion [x, y, z, w]."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


# ---------------------------------------------------------------------------
# Gram-Schmidt
# ---------------------------------------------------------------------------


class TestGramSchmidt:
    def test_identity_columns(self):
        mat = gram_schmidt(np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-7)

    def test_output_is_orthonormal(self):
        for _ in range(100):
            v1, v2 = RNG.standard_normal(3), RNG.standard_normal(3)
            mat = gram_schmidt(v1, v2)
            np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(np.linalg.det(mat), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Interleaved rot6d
# ---------------------------------------------------------------------------


class TestInterleaved:
    def test_roundtrip_matrix(self):
        for _ in range(100):
            mat = _random_rotation_matrix()
            v6 = matrix_to_rot6d_interleaved(mat)
            mat2 = rot6d_interleaved_to_matrix(v6)
            np.testing.assert_allclose(mat, mat2, atol=1e-6)

    def test_roundtrip_quat(self):
        for _ in range(50):
            q = _random_quat()
            v6 = quat_to_rot6d_interleaved(q)
            q2 = rot6d_interleaved_to_quat(v6)
            # Quaternions may differ by sign
            if np.dot(q, q2) < 0:
                q2 = -q2
            np.testing.assert_allclose(q, q2, atol=1e-6)

    def test_roundtrip_euler(self):
        for _ in range(50):
            # Restrict pitch to avoid gimbal lock (non-unique euler repr)
            euler = np.array(
                [
                    RNG.uniform(-np.pi * 0.9, np.pi * 0.9),
                    RNG.uniform(-np.pi / 2 * 0.9, np.pi / 2 * 0.9),
                    RNG.uniform(-np.pi * 0.9, np.pi * 0.9),
                ]
            )
            v6 = euler_xyz_to_rot6d_interleaved(euler)
            euler2 = rot6d_interleaved_to_euler_xyz(v6)
            np.testing.assert_allclose(euler, euler2, atol=1e-6)

    def test_layout_is_column_interleaved(self):
        mat = _random_rotation_matrix()
        v6 = matrix_to_rot6d_interleaved(mat)
        # v6 = [mat[0,0], mat[0,1], mat[1,0], mat[1,1], mat[2,0], mat[2,1]]
        assert v6[0] == pytest.approx(mat[0, 0])
        assert v6[1] == pytest.approx(mat[0, 1])
        assert v6[2] == pytest.approx(mat[1, 0])
        assert v6[3] == pytest.approx(mat[1, 1])
        assert v6[4] == pytest.approx(mat[2, 0])
        assert v6[5] == pytest.approx(mat[2, 1])

    def test_axisangle_roundtrip(self):
        for _ in range(50):
            aa = RNG.standard_normal(3) * 2.0
            v6 = axisangle_to_rot6d_interleaved(aa)
            mat = rot6d_interleaved_to_matrix(v6)
            mat2 = axisangle_to_matrix(aa)
            np.testing.assert_allclose(mat, mat2, atol=1e-5)


# ---------------------------------------------------------------------------
# Contiguous rot6d
# ---------------------------------------------------------------------------


class TestContiguous:
    def test_roundtrip_matrix(self):
        for _ in range(100):
            mat = _random_rotation_matrix()
            v6 = matrix_to_rot6d_contiguous(mat)
            mat2 = rot6d_contiguous_to_matrix(v6)
            np.testing.assert_allclose(mat, mat2, atol=1e-6)

    def test_layout_is_column_contiguous(self):
        mat = _random_rotation_matrix()
        v6 = matrix_to_rot6d_contiguous(mat)
        # v6 = [mat[0,0], mat[1,0], mat[2,0], mat[0,1], mat[1,1], mat[2,1]]
        np.testing.assert_allclose(v6[:3], mat[:, 0], atol=1e-12)
        np.testing.assert_allclose(v6[3:6], mat[:, 1], atol=1e-12)

    def test_axisangle_roundtrip(self):
        for _ in range(50):
            aa = RNG.standard_normal(3) * 2.0
            v6 = axisangle_to_rot6d_contiguous(aa)
            mat = rot6d_contiguous_to_matrix(v6)
            q = matrix_to_quat(mat)
            aa2 = quat_to_axisangle(q)
            # Axis-angle can differ by 2*pi wrapping; compare rotation matrices
            mat2 = axisangle_to_matrix(aa2)
            np.testing.assert_allclose(mat, mat2, atol=1e-5)


# ---------------------------------------------------------------------------
# Cross-convention
# ---------------------------------------------------------------------------


class TestCrossConvention:
    def test_same_rotation_different_layout(self):
        for _ in range(100):
            mat = _random_rotation_matrix()
            v_interleaved = matrix_to_rot6d_interleaved(mat)
            v_contiguous = matrix_to_rot6d_contiguous(mat)
            # Both encode the same rotation but with different memory order
            mat_i = rot6d_interleaved_to_matrix(v_interleaved)
            mat_c = rot6d_contiguous_to_matrix(v_contiguous)
            np.testing.assert_allclose(mat_i, mat_c, atol=1e-6)

    def test_interleaved_vs_contiguous_values_differ(self):
        mat = np.array([[0.5, -0.5, 0.7071], [0.5, 0.5, -0.7071], [0.7071, 0.7071, 0.0]])
        # Make it a proper rotation
        mat = _random_rotation_matrix()
        v_i = matrix_to_rot6d_interleaved(mat)
        v_c = matrix_to_rot6d_contiguous(mat)
        # They should NOT be equal (different layouts of same data)
        assert not np.allclose(v_i, v_c)


# ---------------------------------------------------------------------------
# Quaternion / axis-angle / euler
# ---------------------------------------------------------------------------


class TestQuatMatrix:
    def test_identity(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        mat = quat_to_matrix(q)
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-10)

    def test_roundtrip(self):
        for _ in range(100):
            q = _random_quat()
            mat = quat_to_matrix(q)
            q2 = matrix_to_quat(mat)
            if np.dot(q, q2) < 0:
                q2 = -q2
            np.testing.assert_allclose(q, q2, atol=1e-6)


class TestQuatAxisAngle:
    def test_identity(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        aa = quat_to_axisangle(q)
        np.testing.assert_allclose(aa, np.zeros(3), atol=1e-8)

    def test_roundtrip_via_matrix(self):
        for _ in range(50):
            aa = RNG.standard_normal(3)
            mat = axisangle_to_matrix(aa)
            q = matrix_to_quat(mat)
            aa2 = quat_to_axisangle(q)
            mat2 = axisangle_to_matrix(aa2)
            np.testing.assert_allclose(mat, mat2, atol=1e-5)


def _scipy_available() -> bool:
    try:
        import scipy.spatial.transform  # noqa: F401

        return True
    except ImportError:
        return False


class TestEulerXYZ:
    def test_identity(self):
        euler = np.array([0.0, 0.0, 0.0])
        mat = euler_xyz_to_matrix(euler)
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-10)

    def test_roundtrip(self):
        for _ in range(50):
            # Restrict pitch to avoid gimbal lock (non-unique euler repr)
            euler = np.array(
                [
                    RNG.uniform(-np.pi * 0.9, np.pi * 0.9),
                    RNG.uniform(-np.pi / 2 * 0.9, np.pi / 2 * 0.9),
                    RNG.uniform(-np.pi * 0.9, np.pi * 0.9),
                ]
            )
            mat = euler_xyz_to_matrix(euler)
            euler2 = matrix_to_euler_xyz(mat)
            np.testing.assert_allclose(euler, euler2, atol=1e-6)

    @pytest.mark.skipif(
        not _scipy_available(),
        reason="scipy not installed",
    )
    def test_matches_scipy(self):
        from scipy.spatial.transform import Rotation as R

        for _ in range(50):
            euler = RNG.uniform(-np.pi * 0.9, np.pi * 0.9, size=3)
            mat_ours = euler_xyz_to_matrix(euler)
            mat_scipy = R.from_euler("xyz", euler).as_matrix()
            np.testing.assert_allclose(mat_ours, mat_scipy, atol=1e-10)
