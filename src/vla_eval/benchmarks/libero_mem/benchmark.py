"""LIBERO-Mem benchmark — memory-dependent, non-Markovian manipulation tasks."""

from __future__ import annotations

from typing import Any

from vla_eval.benchmarks.base import StepResult
from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.types import Action, Task

_MAX_STEPS = 1000


class LIBEROMemBenchmark(LIBEROBenchmark):
    """Extends LIBEROBenchmark with sequential subgoal tracking for libero-mem."""

    def __init__(
        self,
        suite: str = "libero_mem",
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
    ) -> None:
        super().__init__(
            suite=suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
        )

    def reset(self, task: Task) -> Any:
        obs = super().reset(task)
        # Raise robosuite's internal horizon so the harness's max_steps
        # controls episode length (default horizon ~500 < our 1000).
        assert self._env is not None
        self._env.env.horizon = _MAX_STEPS + self.num_steps_wait + 10
        # Reset subgoal state machine — without this, completed subgoals
        # from a previous episode leak into the next one.
        if hasattr(self._env.env, "reset_subgoal_progress"):
            self._env.env.reset_subgoal_progress()
        return obs

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        # libero-mem's Sequence/Or goals require inc=True to advance the
        # subgoal state machine. Preserve the env's own done flag too.
        assert self._env is not None
        done = result.done or self._env.env._check_success(inc=True)
        return StepResult(obs=result.obs, reward=result.reward, done=done, info=result.info)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "max_steps": _MAX_STEPS,
            "max_episodes_per_task": 50,
            "suite": self.suite,
        }
