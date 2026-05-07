# Isaac-GR00T — Reproduction Report

NVIDIA's generalist robot foundation model. [GitHub](https://github.com/NVIDIA/Isaac-GR00T) |
[Paper](https://arxiv.org/abs/2503.14734) | 2B params.

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **94.9%** | 97.0% | Approximate (-2.1pp) |
| SimplerEnv WidowX | **29.6%** | 57.1%* | Partial (eggplant gap) |
| SimplerEnv Google Robot | **59.7%** | 67.7%** | Approximate (-8pp) |

\* 4-task subset avg. Full 7-task avg = 62.1%.
\** 6-task set.

### LIBERO

| | |
|---|---|
| **Checkpoint** | `0xAnkitSingh/GR00T-N1.6-LIBERO` (community) |
| **Server config** | [`configs/model_servers/groot/libero.yaml`](../../configs/model_servers/groot/libero.yaml) |
| **Benchmark config** | [`configs/benchmarks/libero/all.yaml`](../../configs/benchmarks/libero/all.yaml) |
| **Results** | [`data/groot-libero/`](data/groot-libero/) |

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 96.6% | 97.65% |
| Object | 98.4% | 98.45% |
| Goal | 96.8% | 97.50% |
| Long | 87.8% | 94.35% |
| **Average** | **94.9%** | **97.0%** |

-2.1pp gap likely due to community checkpoint vs official NVIDIA fine-tuning.

### SimplerEnv — WidowX VM

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-bridge` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_widowx.yaml`](../../configs/model_servers/groot/simpler_widowx.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/widowx_vm_groot.yaml`](../../configs/benchmarks/simpler/widowx_vm_groot.yaml) |
| **Docker image** | `simpler-groot` (base simpler + eef_pos patch) |
| **Results** | [`data/groot-simpler-widowx/`](data/groot-simpler-widowx/) (partial) |

200 episodes per task. `chunk_size: 1`, `max_episode_steps: 300`.

| Task | Reproduced (200eps) | Reported (200eps) |
|------|:-------------------:|:-----------------:|
| Stack | 2.0% | 5.5% |
| Carrot | 54.5% | 65.5% |
| Spoon | 54.0% | 64.5% |
| Eggplant | 8.0% | 93% |
| **Average** | **29.6%** | **57.1%** |

Stack matches (2% vs 5.5%). Carrot/Spoon have ~10pp gap. Eggplant has a known
issue: `deterministic_episodes=False` (random object placement) produces very
low success (8%). With `deterministic_episodes=True`, eggplant reaches 50%.
The remaining gap to 93% and the carrot/spoon ~10pp gap may be due to
differences between our official SimplerEnv and NVIDIA's internal fork.

State input verified identical to reference WidowXBridgeEnv (diff < 1e-15).

### SimplerEnv — Google Robot

| | |
|---|---|
| **Checkpoint** | `nvidia/GR00T-N1.6-fractal` (official NVIDIA) |
| **Server config** | [`configs/model_servers/groot/simpler_google_robot.yaml`](../../configs/model_servers/groot/simpler_google_robot.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/google_robot_vm.yaml`](../../configs/benchmarks/simpler/google_robot_vm.yaml) |
| **Docker image** | `simpler-groot` (base simpler + eef_pos patch) |
| **Results** | [`data/groot-simpler-google/`](data/groot-simpler-google/) |

24 episodes per task (reference uses 200). `chunk_size: 1`, `max_episode_steps: 300`.
Sticky gripper (15-step repeat) for Google Robot.

| Task | Reproduced (24eps) | Reported (200eps) |
|------|:------------------:|:-----------------:|
| pick_coke_can | 100% | 97.5% |
| pick_object | 100% | 87% |
| move_near | 58.3% | 75.5% |
| open_drawer | 0% | 44% |
| close_drawer | 100% | 87.5% |
| place_in_closed_drawer | 0% | 14.5% |
| **Average** | **59.7%** | **67.7%** |

pick_coke_can, pick_object, close_drawer reproduced (100%). move_near within
24-episode variance. open_drawer 0% vs 44% needs investigation (24eps may be
insufficient). place_in_closed_drawer 0% vs 14.5% is within 24-episode variance.

State input verified identical to reference GoogleFractalEnv (diff = 0.0).

## Configuration Notes

- N1 paper, N1.5, N1.6 differ significantly. All numbers are **N1.6**.
- `Gr00tPolicy` handles normalization, tokenization, and action decoding internally.
- `embodiment_tag` per benchmark: `LIBERO_PANDA`, `OXE_WIDOWX`, `OXE_GOOGLE`.
- `chunk_size=1` for SimplerEnv (reference `--n_action_steps 1`), `chunk_size=16` for LIBERO.
- `invert_gripper=True` for LIBERO, `False` for SimplerEnv.
- SimplerEnv requires eef_pos Docker patch (`Dockerfile.simpler_groot`):
  - WidowX: `ee_gripper_link` + bridge rotation + joint-limit gripper closedness
  - Google Robot: `link_gripper_tcp` + quaternion wxyz→xyzw reorder + gripper closedness
- Google Robot uses sticky gripper (15-step repeat) and relative gripper actions.
- WidowX uses binary gripper (>0.5 → open).
