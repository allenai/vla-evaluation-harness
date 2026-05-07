---
smoke_config: smoke_test.yaml
---

# LIBERO

Tabletop manipulation benchmark (MuJoCo/robosuite). 4 standard suites + extensions.
[Paper](https://arxiv.org/abs/2310.07899) | [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `all.yaml` | All 4 suites (Spatial + Object + Goal + Long) | 40 | 50 |
| `spatial.yaml` | LIBERO-Spatial suite | 10 | 50 |
| `object.yaml` | LIBERO-Object suite | 10 | 50 |
| `goal.yaml` | LIBERO-Goal suite | 10 | 50 |
| `long.yaml` | LIBERO-Long (90-step horizon) | 10 | 50 |
| `10.yaml` | LIBERO-10 (10-task subset) | 10 | 50 |
| `smoke_test.yaml` | Quick validation (1 task, 2 episodes) | 1 | 2 |

See also: [LIBERO-Pro](../libero_pro/), [LIBERO-Plus](../libero_plus/), [LIBERO-Mem](../libero_mem/).
