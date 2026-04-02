# SimplerEnv Reproduction — Handoff Document

## What is SimplerEnv?

SimplerEnv is a simulation benchmark for real2sim evaluation of robot manipulation policies.
It wraps [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim)
(SAPIEN physics + Vulkan rendering) with preconfigured scenes that visually match
real-world Bridge/Google Robot setups. The benchmark evaluates whether a VLA model
trained on real data can succeed in simulation.

**4 WidowX tasks** (Bridge domain):
- StackGreenCubeOnYellowCube
- PutCarrotOnPlate
- PutSpoonOnTableCloth
- PutEggplantInBasket (uses different camera/scene: `widowx_sink_camera_setup`)

Each task runs 24 episodes (object variation IDs 0–23).

## Current Status: All 3 Models Failed

| Model | Reported | Our Result | Status |
|-------|----------|-----------|--------|
| X-VLA | 95.8% | 0% | Fundamental failure |
| DB-CogACT | 69.5% | 36.5% | ~Half of reported |
| GR00T N1.6 | 57.1% | WIP | Partial results only |

## The Core Problem: `build_maniskill2_env()` vs `simpler_env.make()`

All official SimplerEnv evaluations (GR00T, X-VLA) use:

```python
import simpler_env
env = simpler_env.make("widowx_stack_cube")
```

Which internally calls:

```python
gym.make("StackGreenCubeOnYellowCubeBakedTexInScene-v0",
         obs_mode="rgbd", prepackaged_config=True)
```

Our benchmark historically used `build_maniskill2_env()` with explicit parameters
(scene, robot, control_freq, camera_cfgs, rgb_overlay_path, etc.). This produces
a **different visual domain** (measured pixel diff ≈ 33 mean) because `prepackaged_config=True`
sets camera angles, lighting, overlays, and scene configuration automatically.

**We attempted migrating to `gym.make(..., prepackaged_config=True)`.**
The images became pixel-identical (diff = 0.00 verified inside Docker). But the models
still failed. The problem is NOT the visual domain — it's deeper.

## What We Tried and What Happened

### Attempt 1: X-VLA SimplerEnv

**Reference code**: `/tmp/reference/X-VLA/evaluation/simpler/WidowX/client_blocks.py`
(cloned from `https://github.com/2toinf/X-VLA`)

**Key findings from reference analysis:**

1. **Environment creation**: `simpler_env.make(task)` — no kwargs, no control_mode override
2. **Proprio format** (20D):
   ```python
   # Initial (at reset):
   ee_pos = Pose(base_pose).inv() * Pose(tcp_pose)  # base-relative
   proprio = [ee_pos.x, ee_pos.y, ee_pos.z,
              1, 0, 0, 1, 0, 0, 0,   # hardcoded identity rot6d + pad
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # zeros

   # Subsequent steps:
   proprio[:10] = action_pred[:10]  # model's own prediction (closed-loop)
   ```
3. **Action processing**: 20D → 7D via `rot6d_interleaved → euler_xyz + [0, π/2, 0] offset`
4. **Gripper**: Per-task thresholds — spoon: 0.7, blocks: 0.91, carrot: 0.95, eggplant: 0.8
5. **Success**: `if done: break` — early stop on first success

**Our implementation matches on**:
- Proprio: ✅ hardcoded `[1,0,0,1,0,0,0]` (line 478 of xvla.py)
- Rotation: ✅ interleaved rot6d → euler (verified numerically, max diff ~5e-8)
- euler_offset: ✅ `[0, π/2, 0]`
- Chunking: ✅ chunk_size=30, queue-based pop
- Env images: ✅ pixel-identical (diff = 0.00)

**Our implementation differs on**:
- **Gripper**: We use 0.7 for ALL tasks; reference uses per-task (0.7/0.8/0.91/0.95)
- **Success**: We run to truncation (1200 steps); reference breaks at first `done=True`
- **control_mode**: We passed `arm_pd_ee_target_base_pose_gripper_pd_joint_pos` (WRONG — that's for Google Robot). Reference passes nothing.
- **`pass_rotation_raw`**: We initially didn't set this, causing euler→axisangle conversion. Reference feeds euler directly to env.step().
- **`robot_init_options`**: We pass init_xy and init_rot_quat; reference passes only `obj_init_options`

**Fixes applied (then reverted because eval still failed)**:
- Removed `control_mode` override ✅
- Added `pass_rotation_raw: True` ✅
- Added `accumulate_success: True` ✅

**Result after all fixes: still 0%**. No task succeeded even once across all 96 episodes.

### Attempt 2: DB-CogACT SimplerEnv

**Reference code**: Dexbotic repo has NO SimplerEnv eval code — only training scripts.

**Result**: 36.5% avg (Spoon 66.7%, Carrot 54.2%, Stack 4.2%, Eggplant 20.8%).
Reported: 69.5% (Spoon 87.5%, Carrot 65.28%, Stack 29.17%, Eggplant 95.83%).

Pattern matches (Eggplant highest, Stack lowest in reference), but our scores are
consistently lower. No reference eval code to compare against.

### Attempt 3: GR00T SimplerEnv

Not re-attempted in this session. Previous session had partial results with
`prepackaged_config` workaround: Spoon 66.7%, Carrot 54.2%, Stack 4.2%, Eggplant 20.8%.
GR00T's official eval uses `simpler_env.make()` + custom wrappers (see below).

## Reference Implementation Details

### GR00T Official Eval

Source: `https://github.com/NVIDIA/Isaac-GR00T` → `gr00t/eval/sim/SimplerEnv/simpler_env.py`

```python
# Environment creation:
env = simpler_env.make(env_name)
env._max_episode_steps = 10000

# WidowXBridgeEnv wrapper:
#   - Image: 256×256, from get_image_from_maniskill2_obs_dict()
#   - State: base-relative EE pose with bridge rotation correction:
#       default_rot = [[0,0,1],[0,1,0],[-1,0,0]]
#       rpy = mat2euler(quat2mat(proprio[3:7]) @ default_rot.T)
#       state = [x, y, z, roll, pitch, yaw, pad=0, gripper]
#   - Gripper: binary `2.0 * (action > 0.5) - 1.0`

# MultiStepWrapper:
#   - n_action_steps configurable (default 8-16)
#   - max_episode_steps configurable (default 720)
#   - terminate_on_success option

# Success: accumulated via |= (OR) across episode
```

### X-VLA Official Eval

Source: `https://github.com/2toinf/X-VLA` → `evaluation/simpler/WidowX/client_*.py`

```python
# Environment creation:
env = simpler_env.make(task)  # NO kwargs
obs, _ = env.reset(options={"obj_init_options": {"episode_id": proc_id}})
# NO robot_init_options passed!

# Proprio (initial):
ee_pose = Pose(p=base_pose[:3], q=base_pose[3:]).inv() * Pose(p=tcp_pose[:3], q=tcp_pose[3:])
proprio = np.concatenate([ee_pose.p, [1,0,0,1,0,0,0]])  # 10D
proprio = np.concatenate([proprio, np.zeros(10)])          # 20D

# Action conversion:
action_final = np.concatenate([
    action_pred[:3],                                              # position
    rotate6D_to_euler_xyz(action_pred[3:9]) + [0, π/2, 0],      # rotation
    [1.0 if action_pred[9] < THRESHOLD else -1.0]                # gripper
])
env.step(action_final)  # 7D directly, no axis-angle conversion

# Per-task gripper thresholds:
#   spoon: 0.7, blocks: 0.91, carrot: 0.95, eggplant: 0.8

# Success: if done: break (first-touch)
```

## Unsolved Questions

1. **Why does X-VLA get 0% when all code-level details match?**
   - Verified: images identical, proprio correct, rotation correct, chunking correct
   - Unverified: actual action VALUES from model. Used random-image test which showed
     plausible outputs (pos ~[0.3, 0, 0.13], rot ~[0, π/2, 0]).
   - Hypothesis A: Per-task gripper thresholds matter more than expected
   - Hypothesis B: `robot_init_options` vs none causes subtle env state difference
   - Hypothesis C: Something in the WebSocket serialization corrupts actions
   - **Recommended next step**: Run reference X-VLA code directly on the same machine
     (inside Docker with model access) to confirm the model checkpoint works.
     If reference also fails → model/env compatibility issue.
     If reference succeeds → our framework has a bug.

2. **Why does DB-CogACT get half the reported score?**
   - No reference eval code exists in the Dexbotic repo
   - Could be: different eval protocol, different max_episode_steps, different success criterion
   - Eggplant gap is largest: 20.8% vs 95.83% — likely a camera/scene issue
     (eggplant uses `widowx_sink_camera_setup` which differs from other tasks)

3. **Does `prepackaged_config=True` override `max_episode_steps` from gym.make()?**
   - DB-CogACT: passed 120, episodes ran 504 (overridden by prepackaged default)
   - X-VLA: passed 1200, episodes ran 1200 (our value used)
   - Inconsistent behavior — needs investigation

## Recommended Approach for Next Attempt

**Do NOT iterate with trial-and-error evals.** Each eval takes ~1 hour and yields
no diagnostic information beyond pass/fail.

**Instead:**

1. **Run reference code directly.** Clone X-VLA repo, install deps on a GPU node,
   run `evaluation/simpler/WidowX/client_blocks.py` with a local model server.
   If it works → the model is fine, our framework has a bug.
   If it fails → model checkpoint or env setup issue.

2. **Add action logging.** Modify the benchmark to save the first N actions and
   observations to a file. Compare these against reference code's outputs for
   the same episode (episode_id=0). This pinpoints exactly where values diverge.

3. **Single-step comparison.** Write a test that:
   - Creates env via `simpler_env.make("widowx_spoon_on_towel")`
   - Resets with episode_id=0
   - Extracts observation → feeds to model → gets action
   - Compares action with reference code's output for same observation
   This isolates the model server from the benchmark framework.

4. **Per-task gripper thresholds.** The reference uses different thresholds per task.
   Implement this (requires benchmark→model-server communication of task name,
   or profile-level per-task config). This alone won't fix 0% → 95.8% but could
   affect results by 10-20pp.

## File Reference

| File | What |
|------|------|
| `src/vla_eval/benchmarks/simpler/benchmark.py` | SimplerEnv benchmark |
| `src/vla_eval/model_servers/xvla.py` | X-VLA model server (simpler_widowx profile) |
| `src/vla_eval/model_servers/groot.py` | GR00T model server (bridge_rotation) |
| `src/vla_eval/model_servers/dexbotic/cogact.py` | DB-CogACT model server |
| `configs/simpler_all_tasks.yaml` | Benchmark config (4 WidowX tasks) |
| `configs/model_servers/xvla/simpler_widowx.yaml` | X-VLA SimplerEnv config |
| `configs/model_servers/groot/simpler_widowx.yaml` | GR00T SimplerEnv config |
| `configs/model_servers/db_cogact/simpler.yaml` | DB-CogACT SimplerEnv config |
| `docker/Dockerfile.simpler` | SimplerEnv Docker image |
| `docker/Dockerfile.simpler_xvla` | X-VLA patched ManiSkill2 (absolute EE + sink camera) |
| `docs/reproductions/groot.md` | GR00T reproduction notes |
| `docs/reproductions/xvla.md` | X-VLA reproduction notes |
| `docs/reproductions/dexbotic.md` | DB-CogACT reproduction notes |
| `docs/reproductions/common-pitfalls.md` | Cross-model pitfall taxonomy |

Reference repos (shallow-cloned at `/tmp/reference/`):
- `X-VLA/evaluation/simpler/WidowX/client_*.py` — X-VLA official eval
- `Isaac-GR00T/gr00t/eval/sim/SimplerEnv/simpler_env.py` — GR00T official eval
- `dexbotic/` — no SimplerEnv eval code found
