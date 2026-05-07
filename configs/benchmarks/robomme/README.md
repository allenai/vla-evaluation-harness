---
smoke_config: eval.yaml
---

# RoboMME

Multi-modal evaluation for robotic manipulation (MuJoCo/robosuite).
[Paper](https://arxiv.org/abs/2603.04639) | [GitHub](https://github.com/RoboMME/robomme_policy_learning)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robomme:latest`

## Configs

| File | Description | Suites |
|------|-------------|:------:|
| `eval.yaml` | All 4 suites combined | 4 |
| `counting.yaml` | Counting suite only | 1 |
| `imitation.yaml` | Imitation suite only | 1 |
| `permanence.yaml` | Permanence suite only | 1 |
| `reference.yaml` | Reference suite only | 1 |
