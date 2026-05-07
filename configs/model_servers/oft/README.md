---
smoke_config: libero_spatial.yaml
---

# OpenVLA-OFT

OpenVLA with parallel decoding (OFT head). [Paper](https://arxiv.org/abs/2502.19645) | [GitHub](https://github.com/moojink/openvla-oft)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `libero_spatial.yaml` | LIBERO Spatial | per-suite fine-tuned |
| `libero_object.yaml` | LIBERO Object | per-suite fine-tuned |
| `libero_goal.yaml` | LIBERO Goal | per-suite fine-tuned |
| `libero_10.yaml` | LIBERO-10 | per-suite fine-tuned |
| `libero_joint.yaml` | LIBERO 4-suite joint | shared weights |
| `libero_joint_*.yaml` | LIBERO joint per-suite | shared weights |

Uses `_base.yaml` for shared settings.
