---
smoke_config: eval.yaml
---

# RoboTwin 2.0

Dual-arm manipulation benchmark (SAPIEN). 50 tasks, Easy + Hard protocols.
[Paper](https://arxiv.org/abs/2409.02920) | [GitHub](https://github.com/TianxingChen/RoboTwin)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/robotwin:latest`

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Smoke test (`grab_roller`, no expert check) | 1 | 1 |
| `xvla.yaml` | Easy (`demo_clean`) under X-VLA's protocol: `ee` actions, at most 300 steps | 50 | 100 |

Each expert-checked seed is one task, so `test_num` sets the episodes per task;
`episodes_per_task` repeats each seed.

## Action types

`action_type: qpos` (default) takes 14D joint targets. `action_type: ee` takes 16D
end-effector poses in RoboTwin's `take_action(..., "ee")` layout, `[pos3, quat_wxyz4,
gripper1]` per arm, left first; RoboTwin clips the gripper to [0, 1] (1 = open) and
plans each pose with CuRobo, so `fast_init` is off. Model servers convert their own
action format to this layout. `send_state: true` adds `states`, the current end pose
in the same layout.

In `ee` mode a long-lived benchmark process eventually fails to create SAPIEN's renderer
(`Failed to find a supported physical device "cuda:0"`), observed after 9–36 episodes; a
sharded run's failed shard then errors every item left in the queue. Check the errored
episodes and rerun them in fresh processes.
