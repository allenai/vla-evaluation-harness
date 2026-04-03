"""Rotation conversion utilities for VLA action spaces.

Two rot6d memory layouts coexist in this project:

  - **Interleaved**: ``mat[:, :2].reshape(6)`` → ``[r00, r01, r10, r11, r20, r21]``
    Used by all benchmark code and the official X-VLA evaluation scripts.
    Columns are recovered with stride-2: ``col0 = v[0::2]``, ``col1 = v[1::2]``.

  - **Contiguous**: ``mat[:, :2].T.flatten()`` → ``[r00, r10, r20, r01, r11, r21]``
    Used by the X-VLA model server (``xvla.py``) for internal
    axis-angle ↔ rot6d conversion.

All functions are pure NumPy with no mandatory external dependencies.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core: Gram-Schmidt orthonormalization
# ---------------------------------------------------------------------------


def gram_schmidt(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Two 3-D vectors → 3×3 rotation matrix via Gram-Schmidt.

    ``a1`` becomes the first column (after normalisation), ``a2`` is
    orthogonalised against it to form the second column, and the third
    column is their cross product.
    """
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


# ---------------------------------------------------------------------------
# Interleaved rot6d  (benchmark convention)
# ---------------------------------------------------------------------------


def rot6d_interleaved_to_matrix(v6: np.ndarray) -> np.ndarray:
    """Interleaved 6-D rotation → 3×3 rotation matrix."""
    return gram_schmidt(v6[0::2], v6[1::2])


def matrix_to_rot6d_interleaved(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → interleaved 6-D rotation."""
    return mat[:, :2].reshape(6).copy()


def rot6d_interleaved_to_quat(v6: np.ndarray) -> np.ndarray:
    """Interleaved 6-D rotation → quaternion ``[x, y, z, w]``."""
    return matrix_to_quat(rot6d_interleaved_to_matrix(v6))


def quat_to_rot6d_interleaved(q: np.ndarray) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` → interleaved 6-D rotation."""
    return matrix_to_rot6d_interleaved(quat_to_matrix(q))


def rot6d_interleaved_to_euler_xyz(v6: np.ndarray) -> np.ndarray:
    """Interleaved 6-D rotation → extrinsic XYZ Euler angles (radians)."""
    return matrix_to_euler_xyz(rot6d_interleaved_to_matrix(v6))


def euler_xyz_to_rot6d_interleaved(euler: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ Euler angles → interleaved 6-D rotation."""
    return matrix_to_rot6d_interleaved(euler_xyz_to_matrix(euler))


def axisangle_to_rot6d_interleaved(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3-D) → interleaved 6-D rotation."""
    return matrix_to_rot6d_interleaved(axisangle_to_matrix(aa)).astype(np.float32)


# ---------------------------------------------------------------------------
# Contiguous rot6d  (model-server convention)
# ---------------------------------------------------------------------------


def rot6d_contiguous_to_matrix(v6: np.ndarray) -> np.ndarray:
    """Contiguous 6-D rotation → 3×3 rotation matrix."""
    return gram_schmidt(v6[:3], v6[3:6])


def matrix_to_rot6d_contiguous(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → contiguous 6-D rotation."""
    return mat[:, :2].T.flatten().copy()


# ---------------------------------------------------------------------------
# Quaternion ↔ matrix  (scalar-last: [x, y, z, w])
# ---------------------------------------------------------------------------


def matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion ``[x, y, z, w]`` (Shepperd's method)."""
    m = np.asarray(mat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` → 3×3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# ---------------------------------------------------------------------------
# Quaternion ↔ axis-angle
# ---------------------------------------------------------------------------


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` → axis-angle (3-D)."""
    q = -quat if quat[3] < 0 else quat.copy()
    w = float(np.clip(q[3], -1.0, 1.0))
    sin_half = np.sqrt(1.0 - w * w)
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (q[:3] / sin_half * angle).astype(np.float32)


def axisangle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3-D) → 3×3 rotation matrix (Rodrigues)."""
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def axisangle_to_rot6d_contiguous(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3-D) → contiguous 6-D rotation."""
    return matrix_to_rot6d_contiguous(axisangle_to_matrix(aa)).astype(np.float32)


def euler_xyz_to_rot6d_contiguous(euler: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ Euler → contiguous 6-D rotation."""
    return matrix_to_rot6d_contiguous(euler_xyz_to_matrix(euler)).astype(np.float32)


# ---------------------------------------------------------------------------
# Euler XYZ ↔ matrix  (extrinsic XYZ, radians)
# ---------------------------------------------------------------------------


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Extrinsic XYZ Euler angles → 3×3 rotation matrix.

    Equivalent to ``scipy.spatial.transform.Rotation.from_euler('xyz', euler).as_matrix()``.
    """
    x, y, z = float(euler[0]), float(euler[1]), float(euler[2])
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    # R = Rz @ Ry @ Rx  (extrinsic XYZ = intrinsic ZYX)
    return np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ]
    )


def matrix_to_euler_xyz(mat: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → extrinsic XYZ Euler angles (radians).

    Equivalent to ``scipy.spatial.transform.Rotation.from_matrix(m).as_euler('xyz')``.
    """
    sy = np.sqrt(mat[0, 0] ** 2 + mat[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(mat[2, 1], mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])
