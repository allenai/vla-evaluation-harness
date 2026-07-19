---
smoke_config: smoke.yaml
---

# RoboCasa and RoboCasa365

Official 50-task multi-task benchmark using RoboCasa's Panda-Omron Gymnasium wrapper.
[Paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf) ·
[Code](https://github.com/robocasa/robocasa) ·
[Protocol](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html) ·
[Leaderboard](https://robocasa.ai/leaderboard.html)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robocasa:latest`

## Configs

| File | Description | Tasks | Episodes/task |
| --- | --- | ---: | ---: |
| `eval.yaml` | Legacy atomic-task evaluation | 6 | 5 |
| `rc365.yaml` | Official multi-task protocol | 50 | 50 |
| `smoke.yaml` | Contract smoke test | 2 | 1 |

The adapter reads task membership and task-specific horizons from RoboCasa's registry.
The `rc365.yaml` and `smoke.yaml` configs select this behavior with `protocol: rc365`; omitting `protocol` preserves the legacy arbitrary-task, configurable-camera, 7-D adapter.
It evaluates the target50 tasks in pretraining kitchens (`split: pretrain`), matching the official multi-task leaderboard protocol.
Success is sampled and accumulated at the end of each 16-action chunk, while every episode continues to its task-specific horizon.
The wire contract preserves all 12 Panda-Omron dimensions in the official dataset order instead of padding a 7-D arm action.

## Implementation boundary

- Reused upstream: the registered RoboCasa Gym environment, task registry, pretrain split, per-task horizon, observation schema, and strict success predicate.
- Implemented here: canonical observation mapping and lossless 12-D named-action decoding.
- Dependency boundary: Docker verifies RoboCasa `1.0.1` and robosuite `1.5.2` at tested patch revisions because those upstream fixes have no new semantic tags.
