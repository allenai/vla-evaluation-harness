---
smoke_config: eval.yaml
---

# ManiSkill2

Generalizable manipulation benchmark (SAPIEN).
[Paper](https://arxiv.org/abs/2302.04659) | [GitHub](https://github.com/haosulab/ManiSkill2)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/maniskill2:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Full ManiSkill2 evaluation | 20 | 100 |

## Rendering without a GPU

`--render cpu` uses Mesa lavapipe and starts the container without a GPU:

```bash
vla-eval run --render cpu -c configs/benchmarks/maniskill2/eval.yaml
```

The image carries a compatible lavapipe ICD at `/opt/lavapipe/lvp_icd.json`. Set
`MANISKILL2_LAVAPIPE_ICD` to use another manifest; missing or non-lavapipe ICDs fail fast.

See [docs/render-backends.md](../../../docs/render-backends.md) for the mode's interaction with
`docker.gpus`, and for every benchmark's support.
