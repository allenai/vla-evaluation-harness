---
smoke_config: smoke.yaml
---

# RoboCasa365

Official 50-task multi-task benchmark using RoboCasa's Panda-Omron Gymnasium wrapper.
[Paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf) ·
[Code](https://github.com/robocasa/robocasa) ·
[Protocol](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html) ·
[Leaderboard](https://robocasa.ai/leaderboard.html)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robocasa:latest`

## Configs

| File | Description | Tasks | Episodes/task |
| --- | --- | ---: | ---: |
| `eval.yaml` | Official multi-task protocol | 50 | 50 |
| `smoke.yaml` | Contract smoke test | 2 | 1 |

The adapter reads task membership and task-specific horizons from RoboCasa's registry.
It evaluates the target50 tasks on the `target` environment split.
The wire contract preserves all 12 Panda-Omron dimensions in the official dataset order instead of padding a 7-D arm action.

## Implementation boundary

- Reused upstream: the registered RoboCasa Gym environment, task registry, target split, per-task horizon, observation schema, and strict success predicate.
- Implemented here: canonical observation mapping and lossless 12-D named-action decoding.
- Dependency boundary: Docker pins robosuite `v1.5.2` and verifies RoboCasa package `1.0.1` at its tested commit because upstream has no `v1.0.1` tag.
- Regression check: run `experiments/robocasa_parity.py` in the RoboCasa image after an upstream change.
