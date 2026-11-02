---
smoke_config: eval.yaml
---

# Kinetix

Physics-based 2D manipulation benchmark (JAX).
[Paper](https://arxiv.org/abs/2410.23208) | [GitHub](https://github.com/FLAIROx/Kinetix)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/kinetix:latest`

The 12 levels of the RTC paper (arXiv:2506.07339), symbolic observations, action noise σ = 0.1, 256-step cap;
success is the env's `GoalR`. Pair with `configs/model_servers/rtc/` (see [rtc.md](../../../docs/reproductions/rtc.md)).

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Sync (step-counted delay) | 12 | 2048 |
| `realtime.yaml` | Live, 10 Hz wall clock | 12 | 2048 |
