# Isaac-GR00T — Reproduction Report

NVIDIA's generalist robot foundation model. [GitHub](https://github.com/NVIDIA/Isaac-GR00T) |
[Paper](https://arxiv.org/abs/2503.14734) | 2B params.

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **94.9%** | 97.0% | Approximate (−2.1pp) |
| SimplerEnv WidowX | WIP | 57.1%† | WIP (needs `simpler_env.make` migration)‡ |
| SimplerEnv Google Robot | — | 67.7%†† | Not yet evaluated |
| CALVIN | — | — | No checkpoint |

† 4-task subset avg (Spoon 64.5, Carrot 65.5, Eggplant 93.0, Block 5.5). Full 7-task avg = 62.1%.
†† Non-standard 6-task set.

### LIBERO

| | |
|---|---|
| **Checkpoint** | `0xAnkitSingh/GR00T-N1.6-LIBERO` (community) |
| **Server config** | [`configs/model_servers/groot/libero.yaml`](../../configs/model_servers/groot/libero.yaml) |
| **Benchmark config** | [`configs/libero_all.yaml`](../../configs/libero_all.yaml) |
| **Results** | [`data/groot-libero/`](data/groot-libero/) |

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 96.6% | 97.65% |
| Object | 98.4% | 98.45% |
| Goal | 96.8% | 97.50% |
| Long | 87.8% | 94.35% |
| **Average** | **94.9%** | **97.0%** |

Pipeline audit: All items match. No discrepancies in pipeline.
- −2.1pp gap likely due to community checkpoint vs official NVIDIA fine-tuning.
- Community checkpoint: `0xAnkitSingh/GR00T-N1.6-LIBERO`, `embodiment_tag=LIBERO_PANDA`.
- Official NVIDIA does not release LIBERO checkpoints — only training recipe (20K steps, batch 640).
- `invert_gripper=True`: maps model output [0,1] (0=close) → [-1,+1] (+1=close) via `1.0 - 2.0 * x`.
- `chunk_size=16`.

### SimplerEnv — WidowX VM

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-bridge` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_widowx.yaml`](../../configs/model_servers/groot/simpler_widowx.yaml) |
| **Benchmark config** | [`configs/simpler_all_tasks.yaml`](../../configs/simpler_all_tasks.yaml) |
| **Results** | WIP |

Status: **WIP** — needs migration to `simpler_env.make()`.

Reported (4-task subset): Spoon 64.5%, Carrot 65.5%, Eggplant 93.0%, Block 5.5% = **57.1% avg**.

**Key finding**: Official GR00T eval uses `simpler_env.make()` which internally calls
`gym.make(env_name, obs_mode="rgbd", prepackaged_config=True)`. Our benchmark uses
`build_maniskill2_env()` with explicit parameters, producing a different visual domain
(pixel diff=33 vs official). This affects all SimplerEnv models, not just GR00T.

Pipeline audit findings (to be applied in `simpler_env.make` migration):
- **State**: base-relative EE pose from `inv(base_mat) @ tcp_mat` + bridge rotation correction. Implemented.
- **accumulate_success**: OR-accumulate success across episode (official `rollout_policy.py`). Implemented.
- **chunk_size=1**: matching official SimplerEnv README `n_action_steps=1`.
- **max_episode_steps=504**: matching official `MultiStepConfig` default.
- **Gripper**: GR00T bridge [0,1] maps correctly without `invert_gripper`.
- **prepackaged_config / simpler_env.make**: must be used for correct visual domain. Not yet integrated into benchmark.

‡ Preliminary results (with `prepackaged_config` workaround, not yet via `simpler_env.make`):
PutSpoon 66.7% (official 64.5%), PutCarrot 54.2%, StackGreen 4.2%, PutEggplant 20.8%.
PutEggplant gap likely from NVIDIA's custom ManiSkill2 fork (`youliangtan/ManiSkill2_real2sim`).

### SimplerEnv — Google Robot

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-fractal` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_google_robot.yaml`](../../configs/model_servers/groot/simpler_google_robot.yaml) |
| **Benchmark config** | [`configs/simpler_all_tasks.yaml`](../../configs/simpler_all_tasks.yaml) |
| **Results** | — (not yet evaluated) |

Status: Not yet evaluated.
Checkpoint: `nvidia/GR00T-N1.6-fractal`, `embodiment_tag=OXE_GOOGLE`.
Reported (6-task): 67.7% avg. Requires sticky gripper (15 repeats) — not yet implemented.

## Configuration Notes

- N1 paper ≠ N1.5 ≠ N1.6 — versions differ significantly. All numbers are **N1.6**.
- `Gr00tPolicy` handles normalization, tokenization, and action decoding internally.
- `embodiment_tag` must be set per benchmark: `LIBERO_PANDA`, `OXE_WIDOWX`, `OXE_GOOGLE`.
- Action chunk size = 16 (but official SimplerEnv eval uses n_action_steps=8).