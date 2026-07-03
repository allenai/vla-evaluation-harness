---
smoke_config: smoke_test.yaml
---

# DuoBench

Bimanual manipulation on a Franka Research 3 Duo (MuJoCo). 11 tasks.
[Paper](https://arxiv.org/abs/2606.11901) | [GitHub](https://github.com/RobotControlStack/duobench)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/duobench:latest`

Tasks: `ball_maze`, `bin_sort`, `block_balance`, `carry_pot`, `hinge_chest`,
`join_blocks`, `pour_marbles`, `spring_door`, `transfer_cube`, `transfer_gate`,
`transfer_reorient`.

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Canonical evaluation — all 11 tasks | 11 | 100 |
| `smoke_test.yaml` | Minimal pipeline check (`ball_maze`, 1 episode) | 1 | 1 |

