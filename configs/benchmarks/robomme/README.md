---
smoke_config: eval.yaml
---

# RoboMME

Multi-modal evaluation for robotic manipulation (ManiSkill3 fork / SAPIEN).
[Paper](https://arxiv.org/abs/2603.04639) | [GitHub](https://github.com/RoboMME/robomme_policy_learning)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robomme:latest`

## Rendering

These configs default to `render: cpu` (lavapipe software Vulkan). SAPIEN's native
NVIDIA path is roughly 5-10x faster but hangs at the first image capture on a small
subset of hosts, and no startup probe can reliably certify a host (issue #112). On a
known-good host, pass `--render gpu` to opt back into the native path. If lavapipe
cannot be resolved on your host, point `ROBOMME_LAVAPIPE_ICD` at a Mesa ICD file
(see [docs/render-backends.md](../../../docs/render-backends.md)).

## Configs

| File | Description | Suites |
|------|-------------|:------:|
| `eval.yaml` | All 4 suites combined | 4 |
| `counting.yaml` | Counting suite only | 1 |
| `imitation.yaml` | Imitation suite only | 1 |
| `permanence.yaml` | Permanence suite only | 1 |
| `reference.yaml` | Reference suite only | 1 |
