# Common Reproduction Pitfalls

These pitfalls were identified during systematic pipeline verification (cross-benchmark audit) of 5+ VLA codebases across 3 simulation benchmarks. They are organized by failure category to help future integrators avoid the same mistakes.

## 1. Rotation Conventions

**rot6d layout: interleaved vs contiguous**
- What: Two incompatible memory layouts exist for 6D rotation representation. Interleaved: `[r00, r10, r01, r11, r02, r12]` (stride-2 columns). Contiguous: `[r00, r10, r20, r01, r11, r21]` (column-major first 2 cols).
- Impact: Corrupted proprio input and/or wrong output rotations. X-VLA CALVIN went from 3.97→4.30 avg_len after fixing.
- Detection: Compare rot6d values numerically against official eval script's output for the same input.
- Fix: Check the official codebase's rot6d encode/decode function. Match exactly.

**Euler vs axis-angle confusion**
- What: CALVIN robot_obs[3:6] contains euler XYZ, but model server interpreted them as axis-angle.
- Impact: Corrupted initial proprioception. For small angles (~<0.3 rad), euler ≈ axis-angle, so model partially works.
- Detection: Print state[3:6] values; if they look like euler angles (near 0, periodic), confirm the model expects euler or axis-angle.
- Fix: Use the correct conversion function matching the benchmark's coordinate convention.

**Missing euler offset**
- What: Some models require a fixed offset added to euler rotation outputs to align coordinate frames (e.g., X-VLA SimplerEnv WidowX needs `[0, π/2, 0]`).
- Impact: All rotation actions in wrong coordinate frame → 0% success.
- Detection: Compare official eval script's action post-processing with yours.
- Fix: Add the euler offset as a model server parameter.

**Bridge rotation correction**
- What: GR00T SimplerEnv requires a rotation correction matrix (`default_rot = [[0,0,1],[0,1,0],[-1,0,0]]`) to convert ManiSkill2 quaternions to Bridge-convention euler angles.
- Impact: Wrong state representation → degraded performance.
- Detection: Compare state values with official `_process_observation` output.
- Fix: Apply `quat_to_matrix(xyzw) @ default_rot.T → euler` in the state pipeline.

## 2. Gripper Mapping

**Threshold mismatch**
- What: Different codebases use different sigmoid thresholds for gripper binarization (0.5 vs 0.7 vs 0.8).
- Impact: Some gripper open/close decisions are wrong near the boundary. Usually 1-5pp impact.
- Detection: Check the official eval script's gripper post-processing.
- Fix: Match the exact threshold from the official code.

**Polarity inversion**
- What: Gripper conventions differ across benchmarks. LIBERO: +1=close, -1=open. CALVIN: +1=open, -1=close. SimplerEnv WidowX: +1=close, -1=open.
- Impact: Gripper always does the opposite of intended → catastrophic failure.
- Detection: Observe rollout videos — if the gripper opens when it should close, polarity is inverted.
- Fix: Check benchmark convention AND model output convention. Beware of double-flip bugs: if both the model server and benchmark apply a flip, they cancel out.

**Comparison direction (`>` vs `<`)**
- What: For binarization, `action > threshold → close` vs `action < threshold → close` depends on whether the model's sigmoid convention maps high values to open or close.
- Impact: Same as polarity inversion.
- Detection: Read the official eval code's exact comparison operator.
- Fix: Match exactly. Bridge domain typically uses `< threshold → close`, opposite to LIBERO.

## 3. State / Proprioception

**State not sent when model expects it**
- What: Benchmark doesn't send proprioceptive state, but model expects it. Model receives zeros.
- Impact: X-VLA SimplerEnv: 0% success. GR00T SimplerEnv: ~25% (vs 62% with state).
- Detection: Check if official eval computes and sends state. Check model server's fallback when state is missing.
- Fix: Add `send_state` parameter to benchmark; compute the state matching the official format (e.g., EE pose relative to base).

**State key mismatch**
- What: Benchmark sends state under key `"joint_state"` but model reads `"states"` or `"controller_states"` → falls through to zeros.
- Impact: Same as not sending state.
- Detection: Print the observation dict keys on both sides.
- Fix: Align key names, or add key aliasing in the model server.

**Wrong state source**
- What: LIBERO has two state sources — `raw_obs` (observation quaternion) and `robot.controller` (controller internal). They differ by ~90° rotation. X-VLA LIBERO uses controller_states, not observation states.
- Impact: X-VLA LIBERO drops from 97.8% to ~42% with wrong state source.
- Detection: Compare state values from both sources; if they differ significantly, check which one the official code uses.
- Fix: Send the correct state source as specified in the official eval.

## 4. Action Format

**Raw model output vs converted actions**
- What: Model outputs 20D raw actions (e.g., pos3 + rot6d6 + gripper per arm), but benchmark expects 7D or 14D converted actions. Sending raw outputs causes dimension mismatch or misinterpretation.
- Impact: Runtime crash (assertion error) or nonsensical actions.
- Detection: Check `output_action_dim` in model server config. Compare with benchmark's expected action size.
- Fix: Implement proper action conversion (rot6d→euler/quat, gripper binarization) in the model server.

**qpos vs ee action type**
- What: RoboTwin supports both `action_type='qpos'` (direct joint angles) and `action_type='ee'` (end-effector target, IK-solved internally). Sending EE actions as qpos → completely wrong joint positions.
- Impact: 0% success.
- Detection: Check official eval's `env.take_action(action, action_type=...)` call.
- Fix: Add `action_type` as a configurable parameter in the benchmark.

**Absolute vs delta mode not set**
- What: Model outputs absolute positions but benchmark runs in delta mode (or vice versa). Absolute positions get accumulated as deltas → robot flies away.
- Impact: Complete failure.
- Detection: If robot immediately moves to extreme positions, likely absolute/delta mismatch.
- Fix: Set `absolute_action: True/False` in obs_params to match the model's output convention.

## 5. Episode Budget

**max_episode_steps too low**
- What: Official eval allows 1200 steps but benchmark config has 120 (10× too few). Or official uses 720 steps/subtask but benchmark defaults to 360 (2× too few).
- Impact: Tasks that need more time simply time out. X-VLA SimplerEnv: 0% with 120 steps, functional with 1200.
- Detection: Compare `max_episode_steps` / `EP_LEN` between official and config.
- Fix: Match the official eval's episode budget. When in doubt, use the larger value — early termination on success still applies.

## 6. Environment Semantics

**SimplerEnv `terminated` vs `truncated`**
- What: In SimplerEnv, `terminated=True` is a **transient** success signal, not a final verdict. The episode should only end on `truncated=True` (step limit reached). Ending on `terminated` inflates success rates because the robot might knock the object off the target after "succeeding."
- Impact: DB-CogACT Stack Green Block: 75% (wrong, ending on terminated) → 29.2% (correct, ending on truncated).
- Detection: If scores seem suspiciously high on stacking/placement tasks, check episode termination logic.
- Fix: Only end episodes on `truncated=True`, not `terminated`.

**Success accumulation**
- What: Some benchmarks (SimplerEnv with GR00T) require OR-accumulating success across the episode — if the robot ever succeeds at any point, the episode counts as successful even if the object falls afterward.
- Impact: GR00T SimplerEnv PutSpoon: 45.8% (no accumulation) → 70.8% (with accumulation).
- Detection: Check if official eval uses `success = success or step_success` pattern.
- Fix: Add `accumulate_success` option to benchmark.

## 7. Preprocessing

**Image resize interpolation method**
- What: `PIL.BILINEAR` vs `cv2.INTER_AREA` produce slightly different pixel values for the same resize operation.
- Impact: Usually small (~1-2pp), but can compound with other issues.
- Detection: Compare preprocessed image tensors numerically.
- Fix: Match the interpolation method used during training data preprocessing.

**Image flip chains**
- What: LIBERO flips wrist images `[::-1, ::-1]` during preprocessing, then X-VLA un-flips them in the model server. If one flip is missing, the image is upside-down.
- Impact: Significant — model sees inverted wrist view.
- Detection: Visualize the image at each pipeline stage.
- Fix: Trace the full image transform chain from env → benchmark → model server → model input.

## 8. Serialization / Data Bugs

**numpy bool → string serialization**
- What: `json.dumps(default=str)` converts numpy `False` to string `"False"`, which is truthy in Python. Success rates get inflated.
- Impact: DB-CogACT showed 100% on 3 LIBERO suites before fix.
- Detection: Check if success values in JSON results are strings vs booleans.
- Fix: Normalize `success` to Python `bool` before serialization.

**Action chunk indexing**
- What: Using `actions[0]` instead of the full chunk — only the first action of N predicted actions is used, ignoring the rest.
- Impact: ~60% success rate (down from 95%) because the robot takes one step then waits for re-inference.
- Detection: Check if `predict()` returns a chunk but only `actions[0]` is consumed.
- Fix: Return `np.array(actions, dtype=np.float32)` for the full chunk.

---

## Quick Checklist

Before claiming a reproduction, verify these for each model×benchmark pair:

- [ ] Action dimension matches (model output → benchmark input)
- [ ] Action mode (absolute vs delta) matches
- [ ] Rotation convention (euler/axis-angle/quaternion/rot6d layout) matches
- [ ] Gripper threshold, polarity, and binarization direction match
- [ ] State/proprio is sent with correct key, format, and source
- [ ] Episode budget (max_steps) matches official eval
- [ ] Image preprocessing (resize, interpolation, flip) matches
- [ ] Episode termination logic matches (terminated vs truncated semantics)
