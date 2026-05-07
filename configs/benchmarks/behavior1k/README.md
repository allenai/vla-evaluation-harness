---
smoke_config: null  # requires Isaac Sim + R1Pro
---

# BEHAVIOR-1K

Large-scale household activity benchmark (OmniGibson/Isaac Sim).
[Paper](https://arxiv.org/abs/2403.09227) | [GitHub](https://github.com/StanfordVL/OmniGibson)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/behavior1k:latest`

Requires an R1Pro-compatible model server. See [behavior1k.md](../../../docs/reproductions/behavior1k.md).

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Full BEHAVIOR-1K evaluation | 50 | 5 |
