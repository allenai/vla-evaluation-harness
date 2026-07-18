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

The adapter reads task membership and task-specific horizons from RoboCasa's
registry. It evaluates the target50 tasks on RoboCasa's `target` environment
split (target object instances and target kitchen layouts). It does not pad
7-D arm actions: the wire contract preserves all 12
Panda-Omron dimensions in the official dataset order: base motion, control
mode, end-effector position, end-effector rotation, and gripper.

## Implementation boundary

- Reused upstream: the registered RoboCasa Gym environment, task registry,
  target split, per-task horizon, observation schema, and strict success
  predicate.
- Implemented here: canonical vla-eval observation mapping, lossless 12-D
  named-action decoding, orchestration, recording, and provenance capture.
- Pinned revisions: see `docker/Dockerfile.robocasa`; validate behavioral
  parity with `experiments/robocasa_parity.py` after an upstream change.
