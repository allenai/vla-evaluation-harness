# Common Reproduction Pitfalls

Pitfalls identified during systematic pipeline verification of 5+ VLA codebases across 3+ simulation benchmarks.

## At a Glance

| Category | Pitfall | Typical Impact | Example |
|----------|---------|:--------------:|---------|
| **Rotation** | rot6d layout mismatch | 5-30pp | X-VLA CALVIN 3.97→4.30 |
| | Euler vs axis-angle confusion | Partial failure | Small angles mask the bug |
| | Missing euler offset | 0% | X-VLA SimplerEnv |
| | Quaternion wxyz vs xyzw | Corrupted state | Subtle near identity rotations |
| | quat→axisangle antipodal normalization | 14-40pp | OFT: 83→97% (Goal), 56→95% (Long) |
| | Bridge rotation correction | Degraded perf | GR00T SimplerEnv |
| **Gripper** | Threshold mismatch | 1-5pp | 0.5 vs 0.7 vs 0.8 |
| | Polarity inversion | Catastrophic | Gripper does the opposite |
| | Missing sticky gripper | 0% on pick tasks | Google Robot 15-step repeat |
| **State** | State not sent | 0-60pp | X-VLA 0%, GR00T ~25%→62% |
| | Wrong state source | 50pp+ | X-VLA LIBERO 97.8%→42% |
| | eef_pos missing (env patch) | 0%→30-55% | GR00T needs NVIDIA fork patch |
| | Gripper closedness formula | ~0.4 error/step | WidowX joint limit range |
| **Actions** | Dimension mismatch | Crash or 0% | 20D raw vs 7D expected |
| | Absolute vs delta mode | 0% | Robot flies away |
| | **Wrong `action_type` on same model** | **0%** | **MolmoBot `joint_pos_rel` vs `joint_pos`** |
| | Unnorm stat keys (min/max vs q99) | 0%→functional | starVLA LIBERO |
| | chunk_size mismatch | 0-30pp | GR00T: 16→1 for SimplerEnv |
| | Gripper raw-range mismatch | 0% | MolmoBot emits ∈[0,255], needs clamp to {0,255} |
| **Episodes** | max_steps too low | 0% | X-VLA: 120 vs 1200 needed |
| | **Wrong `task_horizon` / `policy_dt_ms`** | **0-30pp** | **MolmoBot default 500 + 5 Hz vs paper 600 + 15 Hz** |
| | **No standard termination semantics** | **Scores not comparable** | truncation vs accumulate |
| | Random vs deterministic placement | 40pp+ | GR00T eggplant: 50% vs 4% |
| **Image preprocessing** | Missing center crop at eval | ~3pp | OpenVLA: trained with random crop aug |
| **Environment** | env.seed mismatch | Unknown | OpenVLA uses env.seed(0) not seed(7) |
| | Internal fork differences | 0-80pp | NVIDIA eef_pos, X-VLA absolute EE |

---

## 1. Rotation Conventions

**rot6d layout: interleaved vs contiguous**
- Two incompatible memory layouts for 6D rotation. Interleaved: `[r00, r10, r01, r11, r02, r12]`. Contiguous: `[r00, r10, r20, r01, r11, r21]`.
- Impact: X-VLA CALVIN 3.97→4.30 after fixing.
- Fix: Match the official codebase's rot6d encode/decode exactly.

**Euler vs axis-angle confusion**
- CALVIN robot_obs[3:6] is euler XYZ, not axis-angle. For small angles they're similar, masking the bug.
- Fix: Confirm the convention from the official eval code.

**Missing euler offset**
- Some models need a fixed offset (e.g., X-VLA SimplerEnv needs `[0, π/2, 0]`).
- Impact: 0% — all rotations in wrong frame.

**Quaternion convention (wxyz vs xyzw)**
- ManiSkill2 and `transforms3d` use wxyz. Most other libraries use xyzw. Inline index reordering is error-prone.
- Fix: Use explicit helpers (`quat_wxyz_to_xyzw`) instead of `q[1], q[2], q[3], q[0]`.

**quat→axisangle antipodal normalization**
- Our `quat_to_axisangle` normalizes `w < 0` quaternions by flipping the sign (angle ∈ [0, π]). The robosuite implementation does not (angle ∈ [0, 2π]). Training data generated with robosuite convention means the model expects the non-antipodal representation.
- Impact: OFT Goal 83.4% → 97.4%, Long 55.8% → 95.4%. Longer episodes amplify the effect.
- Fix: Use `quat_no_antipodal=True` in `get_observation_params()` for models trained with robosuite data.

**Bridge rotation correction**
- GR00T SimplerEnv WidowX requires `quat_to_matrix(xyzw) @ default_rot.T → euler` to convert ManiSkill2 quaternions to Bridge convention. Google Robot requires wxyz→xyzw reorder without euler conversion.
- Fix: Compare state values numerically against the official `_process_observation`.

## 2. Gripper Mapping

**Threshold mismatch**
- Different codebases use different thresholds (0.5 vs 0.7 vs 0.8). Usually 1-5pp impact.

**Polarity inversion**
- Conventions differ: LIBERO +1=close/-1=open, CALVIN +1=open/-1=close, SimplerEnv WidowX +1=open/-1=close.
- Beware double-flip: if model server AND benchmark both flip, they cancel out.

**Sticky gripper (Google Robot)**
- Google Robot requires a 15-step "sticky" repeat mechanism for gripper actions. The model outputs relative gripper values that must be held for 15 steps when a significant change is detected.
- Impact: 0% on manipulation tasks without it.

## 3. State / Proprioception

**State not sent**
- Model expects state but receives zeros. X-VLA SimplerEnv: 0%. GR00T: ~25% vs 62%.

**Wrong state source**
- LIBERO has two sources: observation quaternion vs controller internal, differing by ~90°. X-VLA: 97.8%→42% with wrong source.

**eef_pos missing (NVIDIA fork)**
- GR00T requires EE-in-base-frame pose (`eef_pos`), which only exists in NVIDIA's internal ManiSkill2 fork. Official SimplerEnv has `base_pose` (robot base, not EE) — completely different data.
- Impact: ~0% without patch → 30-55% with patch.
- Fix: Patch `base_agent.py` + robot agent (`widowx.py`, `googlerobot.py`). Use try/except for backward compatibility.

**Gripper closedness formula**
- WidowX joint range is `[0.015, 0.037]`, not `[0, 0.037]`. Using the wrong range produces up to 0.4 error per timestep.
- Fix: Use `get_qlimits()` for actual range.

## 4. Action Format

**Dimension mismatch** — Model outputs 20D raw, benchmark expects 7D. Implement conversion in model server.

**Absolute vs delta mode** — Robot flies away if absolute positions are interpreted as deltas.

**Same model, different `action_type` in the eval config**
- A single codebase can ship multiple eval configs for the same checkpoint that differ only in how the action is interpreted. Reproduction failure often comes from picking the wrong one.
- MolmoBot case: the default Franka eval configs are `FrankaState8ClampConfig` (joint_pos_rel / 5 Hz policy) and `FrankaState8ClampAbsPosConfig` (joint_pos / 15 Hz policy). The paper's README pins `FrankaState8ClampAbsPosConfig` for MolmoBot-DROID. Running the same checkpoint under the `_rel` config scored 0/11 (all episodes fail) because the model emits absolute joint targets while the env adds them to current qpos as deltas.
- Fix: always use the exact config class named in the model card's reproduction command, not the one with the most "default"-sounding name.

**Gripper raw-range mismatch**
- A model's gripper head output range is set at train time and rarely matches the env's command range. MolmoBot outputs gripper in roughly [0, 255] (consistent with the robosuite/DROID convention) and the reference policy (`SynthVLAPolicy`) applies `np.where(gripper > 128, 255, 0)` before sending to the env.
- Symptom: without the clamp, the gripper oscillates or stays permanently half-closed. With the clamp it snaps discretely to open/closed.
- Detection: compare `action_spec` units between the training pipeline and the eval harness; check whether the reference policy applies a `clamp_gripper` step in its post-processing.

**Action type (qpos vs ee)** — RoboTwin supports both; sending EE as qpos = 0%.

**Unnormalization stat keys (min/max vs q01/q99)**
- Models may unnormalize actions using different statistic keys from the same stats file. starVLA LIBERO uses `min`/`max`; starVLA SimplerEnv uses `q01`/`q99` (1st/99th percentile). These give different scaling bounds.
- Impact: starVLA LIBERO 0% with q99 → functional with min/max.
- Detection: Check the reference eval's unnormalization function for which keys it reads.
- Fix: Add `unnorm_type` parameter (`minmax` vs `q99`).

## 5. Evaluation Protocol

**chunk_size / n_action_steps**
- Model predicts N actions; eval may use 1 (re-infer every step) or all N. The correct setting is benchmark-specific: GR00T SimplerEnv uses 1, GR00T LIBERO uses 16.
- Impact: GR00T WidowX 0% with 16 → ~30% with 1.

**max_episode_steps**
- X-VLA SimplerEnv: 0% with 120 steps, functional with 1200. Always match the official eval's budget.
- MolmoBot pick-and-place: paper's README specifies `task_horizon=600`, other configs default to 500. Truncating episodes early can shave ~10pp off a 57% task.

**policy_dt_ms (control rate)**
- The step rate at which the model is queried is encoded in the eval config, not the model weights. Mismatching it is equivalent to changing the robot's dynamics at test time.
- MolmoBot case: `FrankaState8ClampAbsPosConfig` uses `policy_dt_ms=66` (≈15 Hz). An alternative config (`SynthVLAFrankaBenchmarkOriginalEvalConfig`) ships with `policy_dt_ms=200` (5 Hz). Running MolmoBot-DROID at 5 Hz produced 0/11 because at 3× the intended time step the commanded joint positions were far outside the training distribution.
- Detection: compare `policy_dt_ms`, `ctrl_dt_ms`, and `sim_dt_ms` between the reference eval config and whatever the harness is loading.

**Termination semantics (cross-model comparability warning)**
- SimplerEnv has no standard success protocol. Two semantics are used on the **same** tasks:
  - **truncation**: run to max steps, success = `done` on the final step. Used by most models (starVLA, X-VLA, DB-CogACT — via the standard `maniskill2_evaluator.py`).
  - **accumulate**: run to max steps, success if `done=True` at **any** point during the episode. Used by GR00T (via `current_successes[env_idx] |= bool(env_success)` in vectorized rollout).
- The difference matters for tasks with **unstable success** — e.g., stacking a cube that can topple. A robot that stacks the cube at step 80 but knocks it off by step 120 scores 0% under truncation but 100% under accumulate. For stable placements (eggplant in deep basket), both give the same result.
- Impact: DB-CogACT Stack: 75% (accumulate) → 29% (truncation). **Published SimplerEnv scores across models are not directly comparable** unless the termination semantics match.
- This is an ecosystem-level issue: SimplerEnv itself does not prescribe a standard, and each codebase uses its own evaluation loop.

**Random vs deterministic episode placement**
- GR00T eggplant: 50% with deterministic placement vs 4% with random. Match the official protocol and use enough episodes (200+) for random.

## 6. Image Preprocessing

**Center crop** (OpenVLA, OpenVLA-OFT)
- Fine-tuned checkpoints trained with random crop augmentation (`crop_scale=0.9`). At eval time, center crop (area 90%, then resize back) must be applied.
- Isolated impact: ~3pp (OpenVLA LIBERO 73.3% → 76.4%).
- Reference: `openvla/experiments/robot/openvla_utils.py:crop_and_resize()`.
- Detection: check if reference config has `center_crop: true` or `image_aug` in checkpoint name.

## 7. Environment / Infra

**env.seed**
- Some references use a different seed for the environment (`env.seed()`) than for the global random state (`set_seed_everywhere()`). For example, the OpenVLA LIBERO reference uses `env.seed(0)` while setting the random seed to 7. The LIBERO `env.seed()` comment says it "seems to affect object positions even when using fixed initial state." Individual impact not measured.
- Detection: check the reference eval's environment setup for explicit `env.seed()` calls.

**Internal forks**
- Some codebases evaluate using forks with patches not in the public repos:
  - **NVIDIA GR00T**: uses `squarefk/SimplerEnv` + `youliangtan/ManiSkill2_real2sim` which add eef_pos proprioception, instruction wording changes, and new tasks.
  - **X-VLA**: uses `255isWhite/SimplerEnv` which patches ManiSkill2 for absolute EE control mode and sink camera alignment. Without these patches, X-VLA gets 0% on SimplerEnv.
- Reported scores from internal forks may not be reproducible on official SimplerEnv.
- Detection: Check if the official eval references a git submodule, specific fork URL, or custom Dockerfile.


