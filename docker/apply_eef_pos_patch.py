"""Apply NVIDIA eef_pos patches to ManiSkill2_real2sim (in-place)."""
import pathlib

MS2 = pathlib.Path("/app/ManiSkill2_real2sim/mani_skill2_real2sim")

# ── 1. base_agent.py: add eef_pos to get_proprioception ───────────────
p = MS2 / "agents/base_agent.py"
src = p.read_text()
src = src.replace(
    "from gymnasium import spaces",
    "from gymnasium import spaces\nfrom transforms3d.quaternions import mat2quat",
)
old_proprio = (
    "    def get_proprioception(self):\n"
    "        obs = OrderedDict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())"
)
new_proprio = """\
    def get_proprioception(self):
        import numpy as _np
        obs = OrderedDict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
        try:
            base_mat = self.base_pose.to_transformation_matrix()
            ee_mat = self.ee_pose.to_transformation_matrix()
            ee_in_base = _np.linalg.inv(base_mat) @ ee_mat
            pos = ee_in_base[:3, 3]
            quat_wxyz = mat2quat(ee_in_base[:3, :3])
            gripper_nwidth = 1 - self.get_gripper_closedness()
            obs["eef_pos"] = _np.concatenate([pos, quat_wxyz, [gripper_nwidth]])
        except (AttributeError, NotImplementedError):
            pass"""
assert old_proprio in src, "base_agent.py: get_proprioception not found"
src = src.replace(old_proprio, new_proprio)
p.write_text(src)
print("✓ base_agent.py patched")

# ── 2. widowx.py: add ee_link, ee_pose, get_gripper_closedness ────────
p = MS2 / "agents/robots/widowx.py"
src = p.read_text()
src = src.replace(
    'self.base_link = [x for x in self.robot.get_links() if x.name == "base_link"][0]',
    'self.ee_link = [x for x in self.robot.get_links() if x.name == "ee_gripper_link"][0]\n'
    '        self.base_link = [x for x in self.robot.get_links() if x.name == "base_link"][0]',
)
old_block = (
    "    @property\n"
    "    def base_pose(self):\n"
    "        return self.base_link.get_pose()\n"
    "\n"
    "\n"
    "class WidowXBridgeDatasetCameraSetup"
)
new_block = (
    "    @property\n"
    "    def base_pose(self):\n"
    "        return self.base_link.get_pose()\n"
    "\n"
    "    @property\n"
    "    def ee_pose(self):\n"
    "        return self.ee_link.get_pose()\n"
    "\n"
    "    def get_gripper_closedness(self):\n"
    "        import numpy as _np\n"
    "        finger_qpos = self.robot.get_qpos()[-2:]\n"
    "        finger_qlim = self.robot.get_qlimits()[-2:]\n"
    "        cl = (finger_qlim[0, 1] - finger_qpos[0]) / (finger_qlim[0, 1] - finger_qlim[0, 0])\n"
    "        cr = (finger_qlim[1, 1] - finger_qpos[1]) / (finger_qlim[1, 1] - finger_qlim[1, 0])\n"
    "        return _np.maximum(_np.mean([cl, cr]), 0.0)\n"
    "\n"
    "\n"
    "class WidowXBridgeDatasetCameraSetup"
)
assert old_block in src, "widowx.py: base_pose block not found"
src = src.replace(old_block, new_block)
p.write_text(src)
print("✓ widowx.py patched")

# ── 3. put_on_in_scene.py: fix eggplant instruction wording ───────────
p = MS2 / "envs/custom_scenes/put_on_in_scene.py"
src = p.read_text()
src = src.replace(
    'return "put eggplant into yellow basket"',
    'return "put the eggplant in the yellow basket"',
)
p.write_text(src)
print("✓ put_on_in_scene.py patched")
