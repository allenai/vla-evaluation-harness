# Common Reproduction Pitfalls

Pitfalls identified during systematic pipeline verification of 5+ VLA codebases across 3+ simulation benchmarks.

## At a Glance

| Category | Pitfall | Typical Impact | Example |
|----------|---------|:--------------:|---------|
| **Rotation** | rot6d layout mismatch | 5-30pp | X-VLA CALVIN 3.97→4.30 |
| | Euler vs axis-angle confusion | Partial failure | Small angles mask the bug |
| | Missing euler offset | 0% | X-VLA SimplerEnv |
| | Quaternion wxyz vs xyzw | Corrupted state | Subtle near identity rotations |
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
| | Unnorm stat keys (min/max vs q99) | 0%→functional | starVLA LIBERO |
| | chunk_size mismatch | 0-30pp | GR00T: 16→1 for SimplerEnv |
| **Episodes** | max_steps too low | 0% | X-VLA: 120 vs 1200 needed |
| | Wrong termination logic | Inflated scores | terminated vs truncated |
| | Random vs deterministic placement | 40pp+ | GR00T eggplant: 50% vs 4% |
| **Environment** | Internal fork differences | 0-80pp | NVIDIA eef_pos, X-VLA absolute EE |

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

**Termination semantics**
- SimplerEnv `terminated=True` is transient. End episodes on `truncated=True`. Ending on `terminated` inflates scores (DB-CogACT stack: 75%→29%).
- Some models (GR00T) need OR-accumulation: success if `terminated=True` at any point.

**Random vs deterministic episode placement**
- GR00T eggplant: 50% with deterministic placement vs 4% with random. Match the official protocol and use enough episodes (200+) for random.

## 6. Environment / Infra

**Internal forks**
- Some codebases evaluate using forks with patches not in the public repos:
  - **NVIDIA GR00T**: uses `squarefk/SimplerEnv` + `youliangtan/ManiSkill2_real2sim` which add eef_pos proprioception, instruction wording changes, and new tasks.
  - **X-VLA**: uses `255isWhite/SimplerEnv` which patches ManiSkill2 for absolute EE control mode and sink camera alignment. Without these patches, X-VLA gets 0% on SimplerEnv.
- Reported scores from internal forks may not be reproducible on official SimplerEnv.
- Detection: Check if the official eval references a git submodule, specific fork URL, or custom Dockerfile.


---

## Quick Checklist

Before claiming a reproduction:

- [ ] Rotation convention matches (euler/axis-angle/rot6d/quaternion wxyz vs xyzw)
- [ ] Action dimension and mode (absolute vs delta) match
- [ ] Gripper: threshold, polarity, sticky mechanism if needed
- [ ] State: correct key, format, source, and eef_pos if required
- [ ] Episode budget (max_steps) and chunk_size match official eval
- [ ] Termination logic matches (truncation vs early_stop vs accumulate)
- [ ] Image preprocessing matches (resize method, flip, resolution)
- [ ] Env version matches (check for internal forks, patches)
