---
smoke_config: libero.yaml
---

# X-VLA

Cross-embodiment VLA with domain-specific soft prompts (0.9B, Florence-2). [Paper](https://arxiv.org/abs/2510.10274) | [GitHub](https://github.com/2toinf/X-VLA)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `libero.yaml` | LIBERO | `2toINF/X-VLA-Libero` |
| `calvin.yaml` | CALVIN | `2toINF/X-VLA-Calvin-ABC_D` |
| `simpler_widowx.yaml` | SimplerEnv WidowX | `2toINF/X-VLA-WidowX` |
| `simpler_google_robot.yaml` | SimplerEnv GR | `2toINF/X-VLA-Google-Robot` |
| `robotwin.yaml` | RoboTwin | `2toINF/X-VLA-WidowX` |

Uses `_base.yaml` for shared settings. `domain_id` selects the embodiment prompt.
