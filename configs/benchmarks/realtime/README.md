---
smoke_config: null  # requires real-time capable server
---

# Real-Time Evaluation

Wall-clock-paced evaluation via vla-eval's `mode: realtime` async episode runner.

A benchmark can run here only if it implements `get_hold_action` (the safe stale-tick hold that the runner reuses when the model hasn't produced a fresh action). Kinetix implements it; LIBERO is sync-only and does not.

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/kinetix:latest`

## Configs

| File | Description | Episodes/task |
|------|-------------|:-------------:|
| `eval.yaml` | Kinetix at 10 Hz wall-clock (minimal example) | 10 |

For the full Kinetix real-time run, see `configs/benchmarks/kinetix/realtime.yaml`.
