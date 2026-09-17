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

SAPIEN rasterizes through Vulkan. `--render cpu` moves that to Mesa lavapipe and starts the
container with no device attached, which is the way to run this benchmark on a host whose GPU
Vulkan path is unusable -- on some H100 nodes rendering hangs inside the first capture or raises
`vk::Device::waitForFences: ErrorDeviceLost`, while the same image renders normally on A100s.

```bash
vla-eval run --render cpu -c configs/benchmarks/simpler/widowx_vm.yaml
```

SAPIEN 2.2.2 needs a lavapipe from Mesa 24.3 or newer, which is older than what Ubuntu 22.04
ships, so the image installs one from conda-forge into `/opt/lavapipe-env` and stashes its ICD at
`/opt/lavapipe/lvp_icd.json`. Nothing else in the image sees that Mesa. If the ICD is missing the
run fails at startup naming the reason rather than falling back to the GPU; `SIMPLER_LAVAPIPE_ICD` points at
a manifest elsewhere.

### What it costs

Not measured for this benchmark. The `maniskill2` image, which shares this one's SAPIEN
2.2.2 and the same four Dockerfile lines, runs 18 to 20 policy steps/s per shard at 256x256
on a GPU-free DGX-H100 node with no black frames. Treat that as an order of magnitude, not as
a SimplerEnv number: this benchmark's scenes, cameras and episode lengths differ.

See [docs/render-backends.md](../../../docs/render-backends.md) for the mode's interaction with
`docker.gpus`, and for every benchmark's support.
