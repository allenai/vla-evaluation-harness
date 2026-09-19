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

## Rendering without a GPU

`--render cpu` uses Mesa lavapipe and starts the container without a GPU:

```bash
vla-eval run --render cpu -c configs/benchmarks/simpler/widowx_vm.yaml
```

The image carries a compatible lavapipe ICD at `/opt/lavapipe/lvp_icd.json`. Set
`SIMPLER_LAVAPIPE_ICD` to use another manifest; missing or non-lavapipe ICDs fail fast.

See [docs/render-backends.md](../../../docs/render-backends.md) for the mode's interaction with
`docker.gpus`, and for every benchmark's support.
