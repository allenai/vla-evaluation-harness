"""LADDER long-horizon robot manipulation benchmark adapters (arXiv:2610.18053).

LADDER (https://github.com/insait-institute/ladder, https://ladderrobotics.ai) evaluates
policies across 145 long-horizon manipulation tasks (1 to 10 subgoals) in three suites:

* :class:`LadderSimplerBenchmark` (``SimplerEnv-Long``): 54 WidowX tasks on SAPIEN 3 / ManiSkill 3,
  24 episodes per task.
* :class:`LadderRobotArenaBenchmark` (``RobotArena-∞-Long``): 47 WidowX tasks on Genesis,
  8 episodes per task.
* :class:`LadderRoboLabBenchmark` (``RoboLab-Long``): 44 Franka DROID tasks on Isaac Sim 6 / Isaac Lab 3,
  8 episodes per task.
"""

from __future__ import annotations

from vla_eval.benchmarks.ladder.robolab import LadderRoboLabBenchmark
from vla_eval.benchmarks.ladder.robot_arena import LadderRobotArenaBenchmark
from vla_eval.benchmarks.ladder.simpler import SIMPLER_LAVAPIPE_ICD_ENV_VAR, LadderSimplerBenchmark
from vla_eval.benchmarks.ladder.tasks import (
    ROBOLAB_LITE_TASKS,
    ROBOLAB_TASKS,
    ROBOT_ARENA_K8_OFFSETS,
    ROBOT_ARENA_LITE_TASKS,
    ROBOT_ARENA_TASKS,
    SIMPLER_LITE_TASKS,
    SIMPLER_TASKS,
)

__all__ = [
    "LadderRoboLabBenchmark",
    "LadderRobotArenaBenchmark",
    "LadderSimplerBenchmark",
    "ROBOLAB_LITE_TASKS",
    "ROBOLAB_TASKS",
    "ROBOT_ARENA_K8_OFFSETS",
    "ROBOT_ARENA_LITE_TASKS",
    "ROBOT_ARENA_TASKS",
    "SIMPLER_LAVAPIPE_ICD_ENV_VAR",
    "SIMPLER_LITE_TASKS",
    "SIMPLER_TASKS",
]
