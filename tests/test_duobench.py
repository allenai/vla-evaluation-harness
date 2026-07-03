"""DuoBench integration tests that do not require the external simulator."""

from __future__ import annotations

from vla_eval.benchmarks.duobench.benchmark import DuoBenchBenchmark


def test_duobench_joint_action_specs_are_benchmark_local() -> None:
    bench = DuoBenchBenchmark(control_mode="joints", relative_to="none")
    spec = bench.get_action_spec()

    assert spec["left_joints"].format == "joint_positions"
    assert spec["right_joints"].format == "joint_positions"
    assert spec["left_joints"].dims == 7
    assert spec["right_joints"].dims == 7


def test_duobench_relative_joint_action_specs_use_existing_delta_convention() -> None:
    bench = DuoBenchBenchmark(control_mode="joints", relative_to="configured_origin")
    spec = bench.get_action_spec()

    assert spec["left_joints"].format == "joint_delta_pos"
    assert spec["right_joints"].format == "joint_delta_pos"


def test_duobench_control_config_is_case_insensitive() -> None:
    bench = DuoBenchBenchmark(control_mode="CARTESIAN_TQuat", relative_to="LAST_STEP")
    spec = bench.get_action_spec()

    assert bench.get_metadata()["action_dim"] == 16
    assert spec["left_position"].format == "delta_xyz"
    assert spec["left_rotation"].format == "quaternion_xyzw"
