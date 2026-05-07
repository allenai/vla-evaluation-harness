---
smoke_config: pi05_baseline.yaml
---

# MME-VLA

Multi-modal evaluation VLA models for RoboMME. [Paper](https://arxiv.org/abs/2603.04639) | [GitHub](https://github.com/RoboMME/robomme_policy_learning)

## Configs

| File | Model | Suite |
|------|-------|-------|
| `pi05_baseline.yaml` | π₀.5 baseline | Counting |
| `framesamp_*.yaml` | FrameSampling | Context/Expert/Modular |
| `rmt_*.yaml` | RMT | Context/Expert/Modular |
| `tokendrop_*.yaml` | TokenDrop | Context/Expert/Modular |
| `ttt_*.yaml` | TTT | Context/Expert/Modular |
| `symbolic_*.yaml` | Symbolic | Grounded/Simple |

Uses `_base.yaml` for shared settings.
