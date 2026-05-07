---
smoke_config: null  # requires real-time capable server
---

# Real-Time Evaluation

Wall-clock paced evaluation on LIBERO using vla-eval's `mode: realtime` async episode runner.
Not a standalone benchmark; uses the LIBERO docker image and benchmark class with real-time pacing (`hz: 10.0`).

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/libero:latest`

## Configs

| File | Description | Suites | Episodes/task |
|------|-------------|:------:|:-------------:|
| `eval.yaml` | 4 LIBERO suites at 10 Hz wall-clock | 4 | 50 |
