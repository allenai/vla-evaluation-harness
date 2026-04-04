# Isaac-GR00T — Reproduction Report

NVIDIA's generalist robot foundation model. [GitHub](https://github.com/NVIDIA/Isaac-GR00T) |
[Paper](https://arxiv.org/abs/2503.14734) | 2B params.

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **94.9%** | 97.0% | Approximate (-2.1pp) |
| SimplerEnv WidowX | WIP | 57.1%* | WIP (200-episode eval in progress) |
| SimplerEnv Google Robot | — | 67.7%** | Not yet evaluated |

\* 4-task subset avg (Spoon 64.5, Carrot 65.5, Eggplant 93.0, Stack 5.5). Full 7-task avg = 62.1%.
\** Non-standard 6-task set.

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

-2.1pp gap likely due to community checkpoint vs official NVIDIA fine-tuning.
Official NVIDIA does not release LIBERO checkpoints — only training recipe.

### SimplerEnv — WidowX VM

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-bridge` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_widowx.yaml`](../../configs/model_servers/groot/simpler_widowx.yaml) |
| **Benchmark config** | [`configs/simpler_all_tasks_groot.yaml`](../../configs/simpler_all_tasks_groot.yaml) |
| **Docker image** | `simpler-groot` (base simpler + eef_pos patch) |
| **Results** | WIP (200-episode eval in progress) |

Requires a patched Docker image (`Dockerfile.simpler_groot`) that adds `eef_pos`
proprioception from NVIDIA's internal ManiSkill2 fork. Without this patch, the
model receives incorrect state (robot base pose instead of EE-in-base-frame pose).

Reference eval protocol: `--n_action_steps 1 --max_episode_steps 300 --n_envs 5`,
200 episodes per task. Our config uses `chunk_size: 1`, `max_episode_steps: 300`.

Preliminary results (24 episodes per task):

| Task | Reproduced | Reported (200eps) |
|------|:----------:|:-----------------:|
| Stack | 4.2% | 5.5% |
| Carrot | 54.2% | 65.5% |
| Spoon | 50.0% | 64.5% |
| Eggplant | 4.2% | 93% |

Stack matches. Carrot/Spoon are in reasonable range given 24 episodes.
Eggplant has a known issue with `deterministic_episodes=False` — random object
placement produces very low success. With `deterministic_episodes=True`, eggplant
reaches 50%. The gap to 93% is under investigation. Full 200-episode eval pending.

### SimplerEnv — Google Robot

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-fractal` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_google_robot.yaml`](../../configs/model_servers/groot/simpler_google_robot.yaml) |

Not yet evaluated. Requires sticky gripper (15-step repeat) not yet implemented.

## Configuration Notes

- N1 paper, N1.5, N1.6 differ significantly. All numbers are **N1.6**.
- `Gr00tPolicy` handles normalization, tokenization, and action decoding internally.
- `embodiment_tag` per benchmark: `LIBERO_PANDA`, `OXE_WIDOWX`, `OXE_GOOGLE`.
- `chunk_size=1` for SimplerEnv (reference `--n_action_steps 1`), `chunk_size=16` for LIBERO.
- `invert_gripper=True` for LIBERO, `False` for SimplerEnv WidowX.
- SimplerEnv WidowX requires eef_pos patch: EE pose in base frame + gripper width from joint limits.
