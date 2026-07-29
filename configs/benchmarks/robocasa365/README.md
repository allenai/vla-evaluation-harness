---
smoke_config: smoke.yaml
---

# RoboCasa365

Multi-task kitchen manipulation benchmark (MuJoCo/robosuite).
[Paper](https://arxiv.org/abs/2603.04356) | [GitHub](https://github.com/robocasa/robocasa) |
[Protocol](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html) |
[Leaderboard](https://robocasa.ai/leaderboard.html)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robocasa365:latest`

The image pins RoboCasa `1.0.1` on robosuite `1.5.2`, both at revisions past
their latest tag because the fixes this benchmark needs shipped without a new
semantic version. The original benchmark is a separate integration — see
[robocasa](../robocasa/); scores are not comparable across the two.

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Official multi-task protocol | 50 | 50 |
| `smoke.yaml` | Contract smoke test | 1 | 1 |

## Protocol

`eval.yaml` follows the official multi-task evaluation: the 50 target tasks
(`atomic_seen` + `composite_seen` + `composite_unseen`) in pretrain kitchens,
50 episodes each. Restrict a run to one suite with `tasks: [atomic_seen]`.

The adapter drives robocasa's own `robocasa/<task>` Gymnasium environment, so
task membership, the pretrain split, per-task horizons, the observation
schema and the success predicate all come from upstream. Two protocol details
are reproduced here rather than inherited, because upstream implements them in
the evaluator rather than the environment:

- every episode runs to its registry horizon (`get_task_horizon`), rather than
  stopping at the first success;
- success is read once per action chunk (`success_check_interval`, 16 by
  default) and latched, so a transient mid-chunk success is not counted.

Actions carry all 12 Panda-Omron dimensions in the layout of robocasa's own
`env_utils.convert_action` — end-effector position (3), rotation (3), gripper
(1), base motion (4), control mode (1) — so nothing is padded or dropped.

## Model servers

| Config | Checkpoint |
|--------|------------|
| [`groot/robocasa365.yaml`](../../model_servers/groot/robocasa365.yaml) | `robocasa/robocasa365_checkpoints` GR00T N1.5 |
