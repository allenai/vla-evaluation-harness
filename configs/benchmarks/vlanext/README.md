---
smoke_config: libero_spatial.yaml
---

# VLANeXt

LIBERO configs matching upstream's eval (`scripts/libero_bench_eval.py`): env seed 0, 10 wait steps.
[Paper](https://arxiv.org/abs/2602.18532) | [GitHub](https://github.com/DravenALG/VLANeXt)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `libero_spatial.yaml` | LIBERO-Spatial for VLANeXt | 10 | 50 |
| `libero_object.yaml` | LIBERO-Object for VLANeXt | 10 | 50 |
| `libero_goal.yaml` | LIBERO-Goal for VLANeXt | 10 | 50 |
| `libero_10.yaml` | LIBERO-10 for VLANeXt | 10 | 50 |
