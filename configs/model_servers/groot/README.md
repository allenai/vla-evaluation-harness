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

For RoboCasa365, download the checkpoint at the revision recorded in the
config and set `ROBOCASA_GR00T_N15_CKPT` to
`gr00t_n1-5/multitask_learning/checkpoint-120000`. The server reports that
checkpoint revision and the benchmark-specific GR00T code revision in its
HELLO metadata, which vla-eval persists with every result.
The configured policy seed fixes the diffusion-noise sequence and is recorded
with the same metadata.

GR00T N1.5 requires FlashAttention 2.7.1.post4. Building it requires a CUDA
toolkit matching the PyTorch 2.7 CUDA wheel.
