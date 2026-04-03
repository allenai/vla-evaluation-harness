# X-VLA — Reproduction Report

Cross-embodiment VLA with soft prompts. [GitHub](https://github.com/2toinf/X-VLA) |
[Paper](https://arxiv.org/abs/2510.10274) | 0.9B params (Florence-2-large).

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **97.4%** | 98.1% | Reproduced |
| CALVIN ABC→D | **4.30** | 4.43 | Reproduced |
| SimplerEnv WidowX | **WIP** | 95.8% | WIP |
| RoboTwin | — | 70.0%/39.0% | Not yet evaluated |

### LIBERO

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-Libero` (official, shared weights) |
| **Server config** | [`configs/model_servers/xvla/libero.yaml`](../../configs/model_servers/xvla/libero.yaml) |
| **Benchmark config** | [`configs/libero_all.yaml`](../../configs/libero_all.yaml) |
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
| **Benchmark config** | [`configs/calvin_eval.yaml`](../../configs/calvin_eval.yaml) |
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
| **Benchmark config** | [`configs/simpler_all_tasks.yaml`](../../configs/simpler_all_tasks.yaml) |
| **Results** | WIP |

Status: **WIP** (pipeline fixes applied, re-evaluation pending).

Reported: 95.8% avg (Spoon 100%, Carrot 91.7%, Block 95.8%, Eggplant 95.8%).

Pipeline audit: 7 discrepancies found and fixed:
1. **`simpler_widowx` profile missing** (BLOCKER) — Server crashed on startup. Fixed: added profile.
2. **euler_offset not implemented** (BLOCKER) — Needs `[0, π/2, 0]` offset. Fixed: added `euler_offset` parameter.
3. **No state sent from SimplerEnv** (BLOCKER) — Model received zero proprio. Fixed: added `send_state` to benchmark; sends 8D EE pose.
4. **Raw 20D output instead of 7D** (BLOCKER) — SimplerEnv expects 7D. Fixed: `output_action_dim=7` + rot6d→euler conversion.
5. **max_steps 120→1200** (BLOCKER) — 10× too few steps. Fixed in dedicated config.
6. **Gripper threshold 0.5→0.7** (MEDIUM) — Bridge domain uses inverted comparison (`< 0.7 → close`). Fixed.
7. **rot6d interleaved decode** (HIGH) — Fixed to match official `rotate6D_to_euler_xyz`.

### RoboTwin 2.0

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-WidowX` (official, domain_id=6) |
| **Server config** | [`configs/model_servers/xvla/robotwin.yaml`](../../configs/model_servers/xvla/robotwin.yaml) |
| **Benchmark config** | [`configs/robotwin_eval.yaml`](../../configs/robotwin_eval.yaml) |
| **Results** | — (not yet evaluated) |

Status: Not yet evaluated (3 BLOCKERS remain).

Reported: Easy 70.0%, Hard 39.0% (50 tasks, Protocol A).

Key blockers:
1. `action_type='ee'` required (currently sends `qpos`)
2. State key mismatch (`joint_state` vs `states`) + format (14D qpos vs 20D EE rot6d)
3. Action conversion missing (20D→16D with rot6d→quat + gripper binarization)

## Configuration Notes

- Shared weights — single model with domain-specific soft prompts (~0.04% params per embodiment).
- `domain_id`: LIBERO=3, CALVIN=2, SimplerEnv Bridge=0, RoboTwin=6.
- `use_predicted_proprio=True` — feeds last predicted action[:10] as next proprio (closed-loop).
- rot6d convention varies by benchmark: contiguous for LIBERO, interleaved for CALVIN/SimplerEnv.
- PEFT/LoRA variants exist (`2toINF/X-VLA-libero-{suite}-peft`) at ~93% avg, lower than full-finetune 98.1%.