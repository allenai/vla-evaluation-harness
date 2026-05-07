"""Align SimplerEnv sink camera + robot init with Bridge dataset for X-VLA."""

import pathlib
import sys


def _patch(path: pathlib.Path, old: str, new: str) -> str:
    text = path.read_text()
    patched = text.replace(old, new)
    if patched == text:
        print(f"ERROR: patch target not found in {path}: {old[:80]!r}", file=sys.stderr)
        sys.exit(1)
    return patched


p = pathlib.Path("mani_skill2_real2sim/agents/configs/widowx/defaults.py")
s = p.read_text()
s = _patch(p, "p=[-0.00300001, -0.21, 0.39]", "p=[0.00, -0.16, 0.336]")
s = s.replace("q=[-0.907313, 0.0782, -0.36434, -0.194741]", "q=[0.909182, -0.0819809, 0.347277, 0.214629]")
s = s.replace("                fov=1.5,  # ignored if intrinsic is passed\n", "")
s = s.replace("                near=0.01,\n", "")
s = s.replace("                far=10,\n", "")
p.write_text(s)
print("Patched sink camera config")

p2 = pathlib.Path("mani_skill2_real2sim/envs/custom_scenes/base_env.py")
s2 = _patch(
    p2,
    "qpos = np.array([-0.2600599, -0.12875618, 0.04461369, -0.00652761, 1.7033415, -0.26983038, 0.037,\n"
    "                                 0.037])",
    "qpos = np.array([-0.01840777,  0.0398835,   0.22242722,  -0.00460194,  1.36524296,  0.00153398, 0.037, 0.037])",
)
s2 = s2.replace("robot_init_height = 0.85", "robot_init_height = 0.870")
s2 = s2.replace("init_y = 0.070", "init_y = 0.028")
p2.write_text(s2)
print("Patched sink robot init")
