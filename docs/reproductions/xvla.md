# X-VLA — Reproduction Report

Cross-embodiment VLA with soft prompts. [GitHub](https://github.com/2toinf/X-VLA) |
[Paper](https://arxiv.org/abs/2510.10274) | 0.9B params (Florence-2-large).

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **97.4%** | 98.1% | Reproduced |
| CALVIN ABC→D | **4.30** | 4.43 | Reproduced |
| SimplerEnv WidowX | **94.8%** | 95.8% | Reproduced |
| SimplerEnv Google Robot VM | **100%** | 98.3% | Reproduced |
| SimplerEnv Google Robot VA | **80.8%** | 84.0%\* | Approximate (-3.2pp) |
| RoboTwin Easy | **66.7%** (300 actions) / **78.2%** (step limits) | 70.0% | Approximate (-3.3pp at 300 actions) |

\* Self-reported best rollout.

### LIBERO

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-Libero` (official, shared weights) |
| **Server config** | [`configs/model_servers/xvla/libero.yaml`](../../configs/model_servers/xvla/libero.yaml) |
| **Benchmark config** | [`configs/benchmarks/libero/all.yaml`](../../configs/benchmarks/libero/all.yaml) |
| **Results** | [`data/xvla-libero/`](data/xvla-libero/) |

4 suites × 10 tasks × 50 episodes. Shared weights (single model, all suites).

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 97.8% | 98.2% |
| Object | 98.6% | 98.6% |
| Goal | 98.0% | 97.8% |
| Long | 95.2% | 97.6% |
| **Average** | **97.4%** | **98.1%** |

Pipeline audit: All 18 items match. No discrepancies.
- Uses `controller_states` (NOT observation `states`) — the coordinate frames differ by ~90°. Using observation states yields ~42%.
- rot6d: contiguous layout (differs from CALVIN/SimplerEnv which use interleaved).
- `absolute_action=True`, `unflip_wrist=True`.
- Wrist image flipped by LIBERO preprocessing, then un-flipped by model server.

### CALVIN (ABC→D)

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-Calvin-ABC_D` (official) |
| **Server config** | [`configs/model_servers/xvla/calvin.yaml`](../../configs/model_servers/xvla/calvin.yaml) |
| **Benchmark config** | [`configs/benchmarks/calvin/eval.yaml`](../../configs/benchmarks/calvin/eval.yaml) |
| **Results** | [`data/xvla-calvin/`](data/xvla-calvin/) |

1000 sequences × 5 chained subtasks.

| Step | Reproduced | Reported |
|------|:----------:|:--------:|
| 1/5 | 95.6% | 97.1% |
| 2/5 | 91.8% | 92.6% |
| 3/5 | 87.1% | 88.5% |
| 4/5 | 81.7% | 84.4% |
| 5/5 | 73.9% | 78.8% |
| **Avg Len** | **4.30** | **4.43**\* |

\* Per-step values from paper sum to 4.41; official avg_len = 4.43.

Pipeline audit: 6 discrepancies found and fixed before achieving 4.30:
1. **rot6d interleaved vs contiguous** (CRITICAL) — CALVIN uses interleaved. Fixed: `euler_to_rot6d_interleaved` for calvin profile.
2. **Euler interpreted as axis-angle** (CRITICAL) — CALVIN robot_obs[3:6] is euler XYZ. Fixed: `euler_state: True` for calvin profile.
3. **Gripper threshold 0.5→0.8** (HIGH) — Official uses 0.8. Fixed in calvin profile. Note: comparison direction (`>`) must NOT change — benchmark handles CALVIN polarity flip.
4. **EP_LEN 360→720** (HIGH) — Official allows 720 steps/subtask. Fixed via obs_params.
5. **absolute_action not set** (CRITICAL) — X-VLA CALVIN outputs absolute actions; benchmark defaulted to delta. Fixed: added `absolute_action: True` to calvin obs_params.
6. **Action format** (MEDIUM) — Official sends (pos3, quat4, gripper). Ours sends (pos3, euler3, gripper). CALVIN env accepts both.

See [common-pitfalls.md](common-pitfalls.md) for detailed patterns.

### SimplerEnv — WidowX VM

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-WidowX` (official) |
| **Server config** | [`configs/model_servers/xvla/simpler_widowx.yaml`](../../configs/model_servers/xvla/simpler_widowx.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/widowx_vm.yaml`](../../configs/benchmarks/simpler/widowx_vm.yaml) |
| **Results** | [`data/xvla-simpler/`](data/xvla-simpler/) |

4 tasks × 24 episodes. Requires [patched ManiSkill2](https://github.com/255isWhite/SimplerEnv) for absolute EE control + sink camera alignment.

| Task | Reproduced | Reported |
|------|:----------:|:--------:|
| Stack | 95.8% | 95.8% |
| Carrot | 91.7% | 91.7% |
| Spoon | 95.8% | 100% |
| Eggplant | 95.8% | 95.8% |
| **Average** | **94.8%** | **95.8%** |

Pipeline audit: 7 discrepancies found and fixed:
1. **`simpler_widowx` profile missing** (BLOCKER) — Server crashed on startup. Fixed: added profile.
2. **euler_offset not implemented** (BLOCKER) — Needs `[0, π/2, 0]` offset. Fixed: added `euler_offset` parameter.
3. **No state sent from SimplerEnv** (BLOCKER) — Model received zero proprio. Fixed: added `send_state` to benchmark; sends 8D EE pose.
4. **Raw 20D output instead of 7D** (BLOCKER) — SimplerEnv expects 7D. Fixed: `output_action_dim=7` + rot6d→euler conversion.
5. **max_steps 120→1200** (BLOCKER) — 10× too few steps. Fixed in dedicated config.
6. **Gripper threshold 0.5→0.7** (MEDIUM) — Bridge domain uses inverted comparison (`< 0.7 → close`). Fixed.
7. **rot6d interleaved decode** (HIGH) — Fixed to match official `rotate6D_to_euler_xyz`.

### SimplerEnv — Google Robot VM

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-Google-Robot` (official) |
| **Server config** | [`configs/model_servers/xvla/simpler_google_robot.yaml`](../../configs/model_servers/xvla/simpler_google_robot.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/google_robot_vm.yaml`](../../configs/benchmarks/simpler/google_robot_vm.yaml) |
| **Docker image** | `simpler-xvla` (absolute EE controller required) |

24 episodes. `chunk_size: 10`, `euler_offset: 0,0,0`, `max_episode_steps: 160`,
`success_mode: early_stop`, `domain_id: 1`.

| Task | Reproduced (24eps) | Reported |
|------|:------------------:|:--------:|
| pick_coke_can | 100% | 98.3% |
| **Average** | **100%** | **98.3%** |

Pipeline audit: 9 discrepancies found and fixed:
1. **Wrong checkpoint** (BLOCKER) -- config used `X-VLA-WidowX`; Google Robot needs `X-VLA-Google-Robot`.
2. **Missing base_pose controller** (BLOCKER) -- Google Robot agent lacked `use_delta=False` controller. Fixed: patched `Dockerfile.simpler_xvla`.
3. **`use_target=True` in controller** (HIGH) -- upstream fork omits `use_target`; including it caused 80% instead of 100%. Fixed: removed.
4. **Prepackaged defaults still delta_pose** (HIGH) -- upstream fork changes all prepackaged defaults to `base_pose`. Fixed: sed patch in Dockerfile.
5. **Position accumulation missing** (BLOCKER) -- X-VLA outputs relative positions; planner needs absolute. Fixed: accumulation in `xvla.py` predict().
6. **Action stride missing** (HIGH) -- official eval uses `[::2][:10]`. Fixed: stride in predict().
7. **Euler rotation** (HIGH) -- controller expects euler XYZ, not axis-angle. Fixed: `euler_offset: 0,0,0` in config.
8. **Gripper threshold 0.5 vs 0.25** (MEDIUM) -- official eval uses 0.25 for Google Robot. Fixed in simpler profile.
9. **Zero proprio init** (MEDIUM) -- official eval passes `zeros(20)` on first step. Fixed for simpler profile.

### SimplerEnv — Google Robot VA

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-Google-Robot` (official) |
| **Server config** | [`configs/model_servers/xvla/simpler_google_robot.yaml`](../../configs/model_servers/xvla/simpler_google_robot.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/google_robot_move_near_va.yaml`](../../configs/benchmarks/simpler/google_robot_move_near_va.yaml) |
| **Docker image** | `simpler-xvla` (absolute EE controller required) |

600 episodes (10 visual variants × 60 episodes). Same server/model config as VM.

| Variant | Reproduced (60eps) |
|---------|:------------------:|
| base | 90.0% |
| no_distractor | 88.3% |
| bg_1 | 93.3% |
| bg_2 | 90.0% |
| light_dark | 90.0% |
| light_bright | 90.0% |
| table_tex_1 | 91.7% |
| table_tex_2 | 83.3% |
| camera_1 | 25.0% |
| camera_2 | 66.7% |
| **Average** | **80.8%** |

Reported: **84.0%**\* (-3.2pp). \* "taken from the best rollout" per X-VLA README.

`camera_1` (alt camera angle) is the clear outlier at 25%. All other
variants are 66-93%.  No additional discrepancies beyond the VM fixes
above.  The VA path uses `gym.make()` directly with explicit `env_name`,
`scene_name`, and `init_config` position grid (see benchmark.py VA support).

### RoboTwin 2.0

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-RoboTwin2` (official, domain_id=6) |
| **Server config** | [`configs/model_servers/xvla/robotwin.yaml`](../../configs/model_servers/xvla/robotwin.yaml) |
| **Benchmark config** | [`configs/benchmarks/robotwin/xvla.yaml`](../../configs/benchmarks/robotwin/xvla.yaml) |
| **Results** | [`data/xvla-robotwin/episodes.json`](data/xvla-robotwin/episodes.json) |

Easy (`demo_clean`), all 50 tasks, following X-VLA's `evaluation/robotwin-2.0/client.py`:
task-name prompt, head/left/right cameras, every 30-action chunk executed whole. Episodes are
the first bundled expert-checked seeds of each task (from 100000; the client draws from 2000,
and `click_alarmclock` on 2000-based expert-checked seeds scores 4/24, the same as ours).

| Horizon | Reproduced (50-task mean, 95% CI) | Reference |
|---|:-:|:-:|
| 300 actions (`client.py`; `max_steps: 300`) | **66.7% ± 2.3** (915 episodes) | 67.15% (released checkpoint, third-party run in [X-VLA#96](https://github.com/2toINF/X-VLA/issues/96)) |
| RoboTwin's per-task step limits (no `max_steps`) | **78.2% ± 3.2** (500 episodes) | 70.0% (paper Table 16) |

The 300-action run matches that run of the released checkpoint; the step limits are what the
paper's long tasks need. Their first `aloha-agilex_clean_50` demos run 338–479 frames, and at
about one frame per action `blocks_ranking_rgb`, `blocks_ranking_size`, `open_microwave`,
`hanging_mug` and `stack_bowls_three` score 0–10% within 300 actions (paper 34–85%). Neither horizon matches Table 16
task by task: `put_bottles_dustbin` (paper 0%) succeeds with its 1700-step limit,
`click_alarmclock` stays near 10% (paper 96%; the policy closes the gripper above the button and
hovers), and several tasks score far above the table (`handover_block`, `pick_dual_bottles`).

| Task | 300 actions (20 seeds) | Step limits (10 seeds) | Paper |
|---|:-:|:-:|:-:|
| adjust_bottle | 20/20 | 10/10 | 97 |
| beat_block_hammer | 19/20 | 9/10 | 78 |
| blocks_ranking_rgb | 0/20 | 10/10 | 79 |
| blocks_ranking_size | 0/20 | 7/10 | 42 |
| click_alarmclock | 3/20 | 1/10 | 96 |
| click_bell | 19/20 | 10/10 | 100 |
| dump_bin_bigbin | 17/20 | 9/10 | 94 |
| grab_roller | 20/20 | 10/10 | 99 |
| handover_block | 16/20 | 7/10 | 27 |
| handover_mic | 20/20 | 10/10 | 100 |
| hanging_mug | 1/10 | 3/10 | 34 |
| lift_pot | 19/20 | 8/10 | 99 |
| move_can_pot | 17/19 | 9/10 | 50 |
| move_pillbottle_pad | 15/20 | 7/10 | 52 |
| move_playingcard_away | 20/20 | 10/10 | 94 |
| move_stapler_pad | 7/18 | 6/10 | 58 |
| open_laptop | 19/20 | 9/10 | 85 |
| open_microwave | 0/9 | 3/10 | 85 |
| pick_diverse_bottles | 14/20 | 8/10 | 27 |
| pick_dual_bottles | 16/20 | 7/10 | 30 |
| place_a2b_left | 3/13 | 4/10 | 62 |
| place_a2b_right | 3/13 | 5/10 | 54 |
| place_bread_basket | 18/20 | 9/10 | 75 |
| place_bread_skillet | 18/20 | 8/10 | 82 |
| place_burger_fries | 20/20 | 9/10 | 98 |
| place_can_basket | 18/20 | 7/10 | 58 |
| place_cans_plasticbox | 19/20 | 10/10 | 100 |
| place_container_plate | 19/20 | 10/10 | 98 |
| place_dual_shoes | 14/20 | 7/10 | 98 |
| place_empty_cup | 19/20 | 10/10 | 98 |
| place_fan | 18/20 | 8/10 | 72 |
| place_mouse_pad | 7/20 | 6/10 | 19 |
| place_object_basket | 9/18 | 7/10 | 50 |
| place_object_scale | 9/18 | 6/10 | 39 |
| place_object_stand | 19/20 | 10/10 | 78 |
| place_phone_stand | 18/20 | 8/10 | 80 |
| place_shoe | 18/20 | 10/10 | 70 |
| press_stapler | 15/20 | 9/10 | 70 |
| put_bottles_dustbin | 0/9 | 8/10 | 0 |
| put_object_cabinet | 5/14 | 4/10 | 78 |
| rotate_qrcode | 16/20 | 8/10 | 78 |
| scan_object | 20/20 | 9/10 | 60 |
| shake_bottle | 20/20 | 10/10 | 99 |
| shake_bottle_horizontally | 20/20 | 10/10 | 99 |
| stack_blocks_three | 0/9 | 6/10 | 22 |
| stack_blocks_two | 16/20 | 10/10 | 87 |
| stack_bowls_three | 0/9 | 8/10 | 80 |
| stack_bowls_two | 14/20 | 10/10 | 83 |
| stamp_seal | 10/19 | 5/10 | 52 |
| turn_switch | 8/17 | 7/10 | 40 |

Pipeline audit — 5 discrepancies fixed:
1. **Wrong checkpoint** (BLOCKER) — config used the Bridge-only `X-VLA-WidowX`.
2. **qpos actions** (BLOCKER) — the benchmark truncated actions to 14D joint targets. Fixed:
   `action_type: ee` passes RoboTwin's 16D end-effector action through.
3. **Action decode** (BLOCKER) — interleaved rot6d → quaternion left in scipy's xyzw order in
   RoboTwin's wxyz slots; gripper `1 - 2 * (sigmoid > 0.7)` = ±1, which RoboTwin clips after
   interpolating (sending 0 would close half as fast).
4. **Proprio** (HIGH) — end pose with the quaternion read as xyzw (as in X-VLA's training,
   [X-VLA#93](https://github.com/2toINF/X-VLA/issues/93)); after each chunk the last commanded pose
   replaces the measured one. The model zeroes the gripper channels.
5. **Prompt** (MEDIUM) — the task name (`click_alarmclock` → `click alarmclock`), not RoboTwin's
   instruction.

RoboTwin's 2025-09-17 change to the Aloha-AgileX CuRobo target handling is not the cause of the
per-task gaps: with the older handling `click_alarmclock` scores 3/24 and `place_a2b_left` 7/16,
the same as with the current one.

Long processes in `ee` mode eventually fail to create SAPIEN's renderer (`Failed to find a
supported physical device "cuda:0"`), here after 9–36 episodes; a sharded run's failed shard then
errors every item left in the shared queue. Errored episodes are excluded and the affected tasks were
rerun in fresh processes; 14 tasks still have 9–19 of their 20 episodes at 300 actions.

## Configuration Notes

- Shared weights — single model with domain-specific soft prompts (~0.04% params per embodiment).
- `domain_id`: LIBERO=3, CALVIN=2, SimplerEnv Bridge=0, Google Robot=1, RoboTwin=6.
- `use_predicted_proprio=True` — feeds last predicted action[:10] as next proprio (closed-loop); RoboTwin uses the last commanded end pose.
- rot6d convention varies by benchmark: contiguous for LIBERO, interleaved for CALVIN/SimplerEnv.
- PEFT/LoRA variants exist (`2toINF/X-VLA-libero-{suite}-peft`) at ~93% avg, lower than full-finetune 98.1%.