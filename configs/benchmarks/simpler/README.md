---
smoke_config: widowx_vm.yaml
---

# SimplerEnv

Real-to-sim evaluation for Google Robot and WidowX (SAPIEN/ManiSkill2).
[Paper](https://arxiv.org/abs/2405.05941) | [GitHub](https://github.com/simpler-env/SimplerEnv)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/simpler:latest`
(X-VLA Google Robot requires `simpler-xvla` for absolute EE controller)

## Configs

### Visual Matching (VM)

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `widowx_vm.yaml` | WidowX Bridge tasks (4 tasks) | 4 | 24 |
| `widowx_vm_groot.yaml` | WidowX VM for GR00T (state-aware) | 4 | 24 |
| `widowx_vm_xvla.yaml` | WidowX VM for X-VLA (absolute EE) | 4 | 24 |
| `google_robot_vm.yaml` | Google Robot pick_coke_can | 1 | 24 |

### Variant Aggregation (VA)

VA configs use domain randomization (backgrounds, lighting, distractors,
camera angles, table textures) with explicit position grids.

| File | Description | Variants | Episodes |
|------|-------------|:--------:|:--------:|
| `google_robot_pick_coke_can_va.yaml` | Pick Coke Can VA | 33 | varies |
| `google_robot_move_near_va.yaml` | Move Near VA | 10 | 60/variant |
| `google_robot_drawer_va.yaml` | Open/Close Drawer VA | 42 | varies |
| `google_robot_put_in_drawer_va.yaml` | Put in Drawer VA | 7 | varies |
