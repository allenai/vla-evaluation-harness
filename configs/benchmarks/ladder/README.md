---
smoke_config:
  ladder_simpler: simpler_smoke.yaml
  ladder_robot_arena: robot_arena_smoke.yaml
  ladder_robolab: robolab_smoke.yaml
---

# LADDER

Long-horizon robot manipulation benchmark across 145 tasks of 1 to 10 subgoals in three simulation suites.
[Paper](https://arxiv.org/abs/2610.18053) | [GitHub](https://github.com/insait-institute/ladder) | [Website](https://ladderrobotics.ai)

**Docker images:** `ghcr.io/allenai/vla-evaluation-harness/ladder-simpler:latest`, `ghcr.io/allenai/vla-evaluation-harness/ladder-robot-arena:latest`, and `ghcr.io/allenai/vla-evaluation-harness/ladder-robolab:latest` (built locally via `docker/build.sh ladder_robolab --accept-license ladder_robolab`).

## Suites & Configs

| Suite | Eval Config | Lite Config | Smoke Config | Simulator | Robot | Tasks | Episodes/task |
|-------|-------------|-------------|--------------|-----------|-------|:-----:|:-------------:|
| **SimplerEnv-Long** | `simpler_eval.yaml` | `simpler_lite.yaml` | `simpler_smoke.yaml` | SAPIEN 3 / ManiSkill 3 | WidowX | 54 | 24 |
| **RobotArena-∞-Long** | `robot_arena_eval.yaml` | `robot_arena_lite.yaml` | `robot_arena_smoke.yaml` | Genesis | WidowX | 47 | 8 |
| **RoboLab-Long** | `robolab_eval.yaml` | `robolab_lite.yaml` | `robolab_smoke.yaml` | Isaac Sim 6 / Isaac Lab 3 | Franka | 44 | 8 |

`*_lite.yaml` runs **LADDER-Lite**, the 10-task subset (2 SimplerEnv-Long, 4 RobotArena-∞-Long, 4 RoboLab-Long tasks; 112 episodes total) that preserves model rankings within 5pp of the full benchmark.

## Run

```bash
# Terminal 1: start a Bridge/WidowX model server (e.g. StarVLA or X-VLA)
vla-eval serve -c configs/model_servers/starvla/groot_qwen3_simpler.yaml
# or: vla-eval serve -c configs/model_servers/xvla/simpler_widowx.yaml

# Terminal 2: run LADDER SimplerEnv-Long or RobotArena-Long
vla-eval run -c configs/benchmarks/ladder/simpler_smoke.yaml
vla-eval run -c configs/benchmarks/ladder/robot_arena_smoke.yaml

# For absolute-EE WidowX policies (e.g. X-VLA simpler_widowx), pass action_mode=abs_ee:
vla-eval run -c configs/benchmarks/ladder/simpler_eval.yaml --param action_mode=abs_ee --param control_mode=arm_pd_ee_target_base_pose_gripper_pd_joint_pos

# For RoboLab-Long (8-D Franka DROID joint actions):
vla-eval serve -c configs/model_servers/pi0/droid.yaml
vla-eval run -c configs/benchmarks/ladder/robolab_smoke.yaml
```

## Metrics

Every suite reports:
- `success`: binary task completion (all subgoals satisfied, respecting order/milestone constraints and distractor stillness).
- `subgoal_progress`: fraction of subgoals completed in `[0.0, 1.0]`.
- `num_subgoals_complete`: count of completed subgoals.

## Rendering without a GPU (`SimplerEnv-Long`)

`LadderSimplerBenchmark` supports `--render cpu` via Mesa lavapipe software Vulkan:

```bash
vla-eval run --render cpu -c configs/benchmarks/ladder/simpler_smoke.yaml
```
