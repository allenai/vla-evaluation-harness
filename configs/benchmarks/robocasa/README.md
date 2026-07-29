---
smoke_config: smoke.yaml
---

# RoboCasa

Kitchen manipulation benchmark (MuJoCo/robosuite).
[Paper](https://arxiv.org/abs/2406.02523) | [GitHub](https://github.com/robocasa/robocasa)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robocasa:latest`

The image pins the original RoboCasa release (`v0.2`) on robosuite `v1.5.0`.
The successor protocol is a separate benchmark — see
[robocasa365](../robocasa365/); its task names, action space and Python API
are not compatible with this one.

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | 24 atomic tasks | 24 | 50 |
| `smoke.yaml` | Contract smoke test | 1 | 1 |

`eval.yaml` runs the benchmark's 24 atomic tasks with a 7-D delta-pose action
that is zero-padded to the robot's full action space, so the mobile base and
torso stay still. Each task runs to its own horizon from robocasa's dataset
registry (300–1000 steps), and an episode ends early on success.

Episodes are drawn from the benchmark's held-out evaluation distribution, taken
from robocasa's own `eval_utils.create_eval_env`: object instances from split
`B`, and the five fixed layout/style pairs `(1,1) (2,2) (4,4) (6,9) (7,10)`.
That helper cannot be called directly — it imports `load_controller_config`,
which robosuite v1.5 replaced — so the adapter passes the same distribution to
`create_env`. Set `obj_instance_split: A` and `eval_scenes: false` to evaluate
on the training distribution instead; those scores are not comparable.
