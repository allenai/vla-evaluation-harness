"""Add absolute EE (base_pose) controller to Google Robot config for X-VLA."""

import pathlib

p = pathlib.Path("mani_skill2_real2sim/agents/configs/google_robot/defaults.py")
s = p.read_text()

# Add base_pose controller definition before _C["arm"] = dict(
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
s = s.replace(old, new_ctrl)

# Register in arm_controllers dict
s = s.replace(
    "arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,",
    "arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,\n"
    "            arm_pd_ee_base_pose_align_interpolate_by_planner=arm_pd_ee_base_pose_align_interpolate_by_planner,",
)
p.write_text(s)
print("Patched Google Robot base_pose controller")
