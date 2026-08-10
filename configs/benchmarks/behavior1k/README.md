---
smoke_config: null  # requires Isaac Sim + R1Pro
---

# BEHAVIOR-1K (2025 challenge protocol)

Large-scale household activity benchmark (OmniGibson/Isaac Sim).
[Paper](https://arxiv.org/abs/2403.09227) | [GitHub](https://github.com/StanfordVL/OmniGibson)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/behavior1k:latest`

**Protocol version:** targets the BEHAVIOR Challenge 2025 stack (v3.7.2, Isaac Sim
4.5.0, B50). It does not exercise the official 2026 evaluator (v3.9.1, 100 tasks,
`python -m omnigibson.eval.eval`); see [issue #113](https://github.com/allenai/vla-evaluation-harness/issues/113).

Requires an R1Pro-compatible model server. See [behavior1k.md](../../../docs/reproductions/behavior1k.md).

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Full BEHAVIOR-1K evaluation | 50 | 5 |
