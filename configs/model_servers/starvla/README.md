---
smoke_config:
  starvla_groot: groot_simpler.yaml
  starvla_oft: oft_simpler.yaml
  starvla_pi: pi_simpler.yaml
  starvla_fast: fast_simpler.yaml
---

# starVLA

Modular VLA framework with pluggable action heads. [Paper](https://arxiv.org/abs/2604.05014) | [GitHub](https://github.com/starVLA/starVLA)

## Configs

| File | Paradigm | Benchmark |
|------|----------|-----------|
| `groot_simpler.yaml` | GR00T (Qwen2.5) | SimplerEnv |
| `groot_qwen3_simpler.yaml` | GR00T (Qwen3) | SimplerEnv |
| `groot_bridge.yaml` | GR00T | SimplerEnv Bridge |
| `groot_calvin.yaml` | GR00T | CALVIN |
| `oft_simpler.yaml` | OFT (Qwen2.5) | SimplerEnv |
| `oft_qwen3_simpler.yaml` | OFT (Qwen3) | SimplerEnv |
| `fast_simpler.yaml` | FAST (Qwen2.5) | SimplerEnv |
| `pi_simpler.yaml` | PI (Qwen2.5) | SimplerEnv |
| `libero_qwen25_groot.yaml` | GR00T (Qwen2.5) | LIBERO |
| `libero_qwen25_oft.yaml` | OFT (Qwen2.5) | LIBERO |
| `libero_qwen25_fast.yaml` | FAST (Qwen2.5) | LIBERO |
| `libero_qwen3_oft.yaml` | OFT (Qwen3) | LIBERO |
| `libero_qwen3_pi.yaml` | PI (Qwen3) | LIBERO |

Uses `_base.yaml` for shared settings.
