---
smoke_config: groot.yaml
---

# GR00T

Dual-system VLA from NVIDIA. [Paper](https://arxiv.org/abs/2503.14734) | [GitHub](https://github.com/NVIDIA/Isaac-GR00T)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `groot.yaml` | Generic | `nvidia/GR00T-N1.6-3B` |
| `libero.yaml` | LIBERO | `nvidia/GR00T-N1.6-3B` |
| `simpler_widowx.yaml` | SimplerEnv WidowX | `nvidia/GR00T-N1.6-bridge` |
| `simpler_google_robot.yaml` | SimplerEnv GR | `nvidia/GR00T-N1.6-fractal` |
| `robocasa365.yaml` | RoboCasa365 | `robocasa/robocasa365_checkpoints` (N1.5) |

`robocasa365.yaml` runs the benchmark's own N1.5 baseline, so it uses a
separate script pinned to the RoboCasa fork of Isaac-GR00T rather than the
`groot.py` server. Set `ROBOCASA365_GROOT_CKPT` to the
`gr00t_n1-5/multitask_learning/checkpoint-120000` directory. That fork's
bundled Eagle backbone imports FlashAttention directly, so the script pins the
exercised FlashAttention build instead of using the Transformers `kernels`
extra.
