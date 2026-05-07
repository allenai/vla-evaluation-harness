---
smoke_config: eval.yaml
---

# Kinetix

Physics-based 2D manipulation benchmark (JAX).
[Paper](https://arxiv.org/abs/2410.23208) | [GitHub](https://github.com/FLAIROx/Kinetix)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/kinetix:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Standard Kinetix evaluation | 40 | 100 |
| `realtime.yaml` | Real-time chunking evaluation | 40 | 100 |
