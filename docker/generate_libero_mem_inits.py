"""Generate missing init states for libero_mem tasks during Docker build."""

from __future__ import annotations

import os
import shutil

os.environ["MUJOCO_GL"] = "egl"
os.environ["EGL_PLATFORM"] = "device"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

NUM_INITS = 120
MAX_RESET_RETRIES = 200


def has_plate_collision(env) -> bool:
    sim = env.env.sim
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        g1 = sim.model.geom_id2name(contact.geom1)
        g2 = sim.model.geom_id2name(contact.geom2)
        if "plate" in g1 and "plate" in g2:
            return True
    return False


def generate_inits_for_task(bm, task_id: int, out_dir: str) -> None:
    task = bm.get_task(task_id)
    out_path = os.path.join(out_dir, task.init_states_file)

    if os.path.exists(out_path):
        print(f"  [{task_id}] SKIP (exists): {task.name}")
        return

    bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=128, camera_widths=128)

    inits = []
    for _ in range(NUM_INITS):
        for _ in range(MAX_RESET_RETRIES):
            env.reset()
            if not has_plate_collision(env):
                break
        inits.append(env.env.sim.get_state().flatten())

    env.close()
    all_inits = np.array(inits)
    torch.save(all_inits, out_path)
    print(f"  [{task_id}] GENERATED {task.name} -> {out_path} shape={all_inits.shape}")


def main() -> None:
    bm = benchmark.get_benchmark_dict()["libero_mem"]()
    out_dir = os.path.join(get_libero_path("init_states"), "libero_mem")
    src_dir = "/app/libero-mem/scripts/__init_data"

    os.makedirs(out_dir, exist_ok=True)

    # Copy existing init states from scripts/__init_data/
    if os.path.isdir(src_dir):
        for f in os.listdir(src_dir):
            if f.endswith(".pruned_init"):
                dst = os.path.join(out_dir, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(src_dir, f), dst)
                    print(f"  COPIED {f}")

    # Generate missing init states
    for task_id in range(bm.n_tasks):
        generate_inits_for_task(bm, task_id, out_dir)


if __name__ == "__main__":
    main()
