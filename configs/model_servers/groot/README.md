---
smoke_config: groot.yaml
---

# GR00T N1.6

3B dual-system VLA from NVIDIA. [Paper](https://arxiv.org/abs/2503.14734) | [GitHub](https://github.com/NVIDIA/Isaac-GR00T)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `groot.yaml` | Generic | `nvidia/GR00T-N1.6-3B` |
| `libero.yaml` | LIBERO | `nvidia/GR00T-N1.6-3B` |
| `simpler_widowx.yaml` | SimplerEnv WidowX | `nvidia/GR00T-N1.6-bridge` |
| `simpler_google_robot.yaml` | SimplerEnv GR | `nvidia/GR00T-N1.6-fractal` |
| `robocasa_n15.yaml` | RoboCasa365 | `robocasa/robocasa365_checkpoints` |

For RoboCasa365, set `ROBOCASA_GR00T_N15_CKPT` to `gr00t_n1-5/multitask_learning/checkpoint-120000` from `robocasa/robocasa365_checkpoints`.
The pinned RoboCasa GR00T fork directly imports FlashAttention, so its script pins the exercised FlashAttention build instead of using the Transformers `kernels` extra.
The configured seed fixes the diffusion-noise sequence.
