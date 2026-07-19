"""Interface convention specifications for action and observation formats.

Every ``ModelServer`` and ``StepBenchmark`` must declare what it produces
and what it consumes via ``get_action_spec()`` and ``get_observation_spec()``.
The orchestrator compares these at episode start and logs warnings on
mismatches — catching convention bugs (wrong rotation format, inverted
gripper, delta-vs-absolute, missing state) before they waste GPU hours.

Usage::

    from vla_eval.specs import DimSpec, POSITION_DELTA, ROTATION_AA, GRIPPER_CLOSE_POS

    class MyModelServer(PredictModelServer):
        def get_action_spec(self) -> dict[str, DimSpec]:
            return {
                "position": POSITION_DELTA,
                "rotation": ROTATION_AA,
                "gripper": GRIPPER_CLOSE_POS,
            }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DimSpec:
    """Specification for one component of the action/observation interface.

    Attributes:
        name: Human-readable component name (e.g. ``"position"``, ``"gripper"``).
        dims: Number of dimensions in the array.  Use 0 for non-array data
            (images, language strings).
        format: Convention string (e.g. ``"delta_xyz"``, ``"binary_close_positive"``).
            Use predefined constants where possible to avoid typos.
        range: Expected ``(min, max)`` value range, or ``None`` if unconstrained.
        accepts: Set of format strings this consumer can handle.  When set,
            ``is_compatible()`` checks membership instead of exact equality.
            Use on the *consumer* side (benchmark action spec, model server
            observation spec) to declare which formats can be converted.
        description: Free-text notes for edge cases.
    """

    name: str
    dims: int
    format: str
    range: tuple[float, float] | None = None
    accepts: frozenset[str] | None = None
    description: str = ""

    def validate(self, value: np.ndarray) -> list[str]:
        """Validate a value against this spec.  Returns a list of errors (empty = valid)."""
        errors: list[str] = []
        flat = np.asarray(value, dtype=np.float64).flatten()
        if self.dims > 0 and len(flat) < self.dims:
            errors.append(f"{self.name}: expected {self.dims}D, got {len(flat)}D")
        if self.range and self.dims > 0:
            lo, hi = self.range
            chunk = flat[: self.dims]
            if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
                errors.append(f"{self.name}: contains NaN/Inf")
            elif np.any(chunk < lo - 0.01) or np.any(chunk > hi + 0.01):
                errors.append(f"{self.name}: values outside [{lo}, {hi}]")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for wire protocol / msgpack)."""
        d: dict[str, Any] = {"name": self.name, "dims": self.dims, "format": self.format}
        if self.range is not None:
            d["range"] = list(self.range)
        if self.accepts is not None:
            d["accepts"] = sorted(self.accepts)
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DimSpec:
        """Deserialize from a plain dict."""
        return cls(
            name=d["name"],
            dims=d["dims"],
            format=d["format"],
            range=tuple(d["range"]) if "range" in d else None,
            accepts=frozenset(d["accepts"]) if "accepts" in d else None,
            description=d.get("description", ""),
        )

    def is_compatible(self, other: DimSpec) -> tuple[bool, str]:
        """Check if ``self`` (producer) is consumable by ``other`` (consumer).

        When *other* has ``accepts`` set, checks format membership.
        Otherwise checks format and dims equality.
        """
        if other.accepts is not None:
            if self.format not in other.accepts:
                return False, f"{self.name}: {self.format} not in accepts {set(other.accepts)}"
            return True, ""
        if self.format != other.format:
            return False, f"{self.name}: {self.format} vs {other.format}"
        if self.dims != other.dims and self.dims > 0 and other.dims > 0:
            return False, f"{self.name}: {self.dims}D vs {other.dims}D"
        return True, ""


def check_specs(
    server_action: dict[str, DimSpec],
    bench_action: dict[str, DimSpec],
    server_obs: dict[str, DimSpec],
    bench_obs: dict[str, DimSpec],
) -> list[str]:
    """Compare server and benchmark specs.  Returns a list of mismatch descriptions."""
    warnings: list[str] = []
    # Action: server produces → benchmark consumes
    if server_action and bench_action and not (server_action.keys() & bench_action.keys()):
        warnings.append("action: no overlapping keys between server and benchmark specs")
    for key in bench_action:
        if key not in server_action and server_action:
            warnings.append(f"action [{key}]: benchmark expects it but server doesn't declare it")
    for key in server_action.keys() & bench_action.keys():
        ok, reason = server_action[key].is_compatible(bench_action[key])
        if not ok:
            warnings.append(f"action [{key}]: {reason}")
    # Observation: benchmark produces → server consumes
    for key in server_obs:
        if key not in bench_obs:
            warnings.append(f"observation [{key}]: server expects it but benchmark doesn't provide it")
    for key in server_obs.keys() & bench_obs.keys():
        ok, reason = bench_obs[key].is_compatible(server_obs[key])
        if not ok:
            warnings.append(f"observation [{key}]: {reason}")
    return warnings


# ---------------------------------------------------------------------------
# Predefined constants — use these instead of raw strings
# ---------------------------------------------------------------------------

# Position
POSITION_DELTA = DimSpec("position", 3, "delta_xyz", (-1, 1))
POSITION_ABSOLUTE = DimSpec("position", 3, "absolute_xyz")

# Rotation
ROTATION_EULER = DimSpec("rotation", 3, "euler_xyz", (-3.15, 3.15))
ROTATION_AA = DimSpec("rotation", 3, "axis_angle", (-3.15, 3.15))
ROTATION_QUAT = DimSpec("rotation", 4, "quaternion_xyzw", (-1, 1))
ROTATION_ROT6D_INTERLEAVED = DimSpec("rotation", 6, "rot6d_interleaved")

# Rotation — consumer variants that accept multiple formats
ROTATION_EULER_ACCEPTS_AA = DimSpec(
    "rotation",
    3,
    "euler_xyz",
    (-3.15, 3.15),
    accepts=frozenset({"euler_xyz", "axis_angle"}),
)

# Gripper
GRIPPER_CLOSE_POS = DimSpec("gripper", 1, "binary_close_positive", (-1, 1))
GRIPPER_CLOSE_NEG = DimSpec("gripper", 1, "binary_close_negative", (-1, 1))
GRIPPER_01 = DimSpec("gripper", 1, "continuous_01", (0, 1))
GRIPPER_RAW = DimSpec("gripper", 1, "raw")

# Observation — images
IMAGE_RGB = DimSpec("image", 0, "rgb_hwc_uint8")

# Observation — state
STATE_EEF_POS_QUAT_GRIP = DimSpec("state", 8, "eef_pos3_quat4_gripper1")
STATE_EEF_POS_AA_GRIP = DimSpec("state", 8, "eef_pos3_axisangle3_gripper2")
STATE_EEF_POS_EULER_GRIP = DimSpec("state", 8, "eef_pos3_euler3_gripper2")
STATE_ROT6D_PROPRIO_20D = DimSpec("state", 20, "rot6d_interleaved_proprio_20d")
STATE_JOINT = DimSpec("state", 0, "joint_positions")

# Observation — language
LANGUAGE = DimSpec("language", 0, "language")

# Generic passthrough (no convention enforced)
RAW = DimSpec("raw", 0, "raw")
