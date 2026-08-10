---
smoke_config: eval.yaml
---

# RoboMME

Multi-modal evaluation for robotic manipulation (ManiSkill3 fork / SAPIEN).
[Paper](https://arxiv.org/abs/2603.04639) | [GitHub](https://github.com/RoboMME/robomme_policy_learning)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robomme:latest`

## Rendering

These configs default to `render: cpu` (lavapipe, shipped in the image): the native
NVIDIA path is ~5-10x faster but hangs at the first capture on some hosts, and no
startup probe can certify a host (issue #112). Pass `--render gpu` on a known-good
host; if Vulkan then cannot find the NVIDIA driver, uncomment the host ICD mount in
the config. Details: [docs/render-backends.md](../../../docs/render-backends.md).

## Configs

| File | Description | Suites |
|------|-------------|:------:|
| `eval.yaml` | All 4 suites combined | 4 |
| `counting.yaml` | Counting suite only | 1 |
| `imitation.yaml` | Imitation suite only | 1 |
| `permanence.yaml` | Permanence suite only | 1 |
| `reference.yaml` | Reference suite only | 1 |
