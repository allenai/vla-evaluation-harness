"""Add absolute EE (base_pose) controller to Google Robot config for X-VLA."""

import pathlib
import sys

p = pathlib.Path("mani_skill2_real2sim/agents/configs/google_robot/defaults.py")
s = p.read_text()

old = '        _C["arm"] = dict('
new_ctrl = (
    "        arm_pd_ee_base_pose_align_interpolate_by_planner = PDEEPoseControllerConfig(\n"
    "            *arm_common_args,\n"
    '            frame="base",\n'
    "            interpolate=True,\n"
    "            use_delta=False,\n"
    "            interpolate_by_planner=True,\n"
    "            interpolate_planner_vlim=self.arm_vel_limit,\n"
    "            interpolate_planner_alim=self.arm_acc_limit,\n"
    "            interpolate_planner_jerklim=self.arm_jerk_limit,\n"
    "            **arm_common_kwargs,\n"
    "        )\n"
    '        _C["arm"] = dict('
)

if old not in s:
    print(f"ERROR: patch target not found in {p}: {old!r}", file=sys.stderr)
    sys.exit(1)
s = s.replace(old, new_ctrl)

register_old = "arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,"
if register_old not in s:
    print(f"ERROR: patch target not found in {p}: {register_old[:80]!r}", file=sys.stderr)
    sys.exit(1)
s = s.replace(
    register_old,
    register_old + "\n"
    "            arm_pd_ee_base_pose_align_interpolate_by_planner=arm_pd_ee_base_pose_align_interpolate_by_planner,",
)

p.write_text(s)
print("Patched Google Robot base_pose controller")
