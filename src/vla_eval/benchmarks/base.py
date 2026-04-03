"""Benchmark ABCs: the environment interface for evaluation.

Two classes:

* :class:`Benchmark` — async, universal contract.  Runners depend only on
  this.  Suitable for both simulation and real-robot environments.
* :class:`StepBenchmark` — sync convenience subclass.  Users implement
  ``reset`` / ``step`` / ``make_obs`` and the class auto-bridges to the
  async parent methods.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from vla_eval.specs import DimSpec

import numpy as np

from vla_eval.types import Action, EpisodeResult, Observation, Task


@dataclass
class StepResult:
    """Result of a single environment step."""

    obs: Any
    reward: float
    done: bool
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Async Benchmark ABC (parent)
# ---------------------------------------------------------------------------


class Benchmark(ABC):
    """Universal async benchmark contract.

    Runners call these methods — they never touch sync helpers directly.

    Command methods (mutate state):
        - ``start_episode(task)`` → None (stores env internally).
        - ``apply_action(action)`` → None (actuate only).

    Query methods (read state):
        - ``get_observation()`` → observation dict for the model server.
        - ``is_done()`` → bool.
        - ``get_time()`` → environment time in seconds.

    Data methods:
        - ``get_tasks()`` → list of task dicts.
        - ``get_result()`` → episode result dict.
        - ``get_metadata()`` → benchmark defaults / metadata.
    """

    # -- abstract: data ---------------------------------------------------

    @abstractmethod
    def get_tasks(self) -> list[Task]:
        """Return the list of tasks this benchmark provides."""

    # -- abstract: commands -----------------------------------------------

    @abstractmethod
    async def start_episode(self, task: Task) -> None:
        """Initialise an episode (env stored internally)."""

    @abstractmethod
    async def apply_action(self, action: Action) -> None:
        """Execute *action* in the environment (fire-and-forget)."""

    # -- abstract: queries ------------------------------------------------

    @abstractmethod
    async def get_observation(self) -> Observation:
        """Read the current observation from the environment."""

    @abstractmethod
    async def is_done(self) -> bool:
        """Return ``True`` when the episode should end."""

    @abstractmethod
    async def get_time(self) -> float:
        """Return the current environment time (seconds since episode start)."""

    @abstractmethod
    async def get_result(self) -> EpisodeResult:
        """Return the episode result (at least ``{"success": bool}``)."""

    # -- optional overrides -----------------------------------------------

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Declare the action input format this benchmark's env consumes.

        Returns a ``{component_name: DimSpec}`` dict.  Use ``accepts`` on
        DimSpec to declare convertible formats (e.g. benchmark converts
        axis-angle to euler internally).

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_action_spec()")

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation output format this benchmark produces.

        Returns a ``{component_name: DimSpec}`` dict describing what
        ``get_observation()`` / ``make_obs()`` sends to the model server.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_observation_spec()")

    def get_metric_keys(self) -> dict[str, str]:
        """Declare which metrics from ``get_result()`` to aggregate.

        Returns ``{field: aggregation}`` where aggregation is one of
        ``"mean"``, ``"sum"``, ``"max"``, ``"min"``.  Fields are stored
        under ``episode["metrics"]`` in the result JSON and aggregated
        into ``TaskResult`` / ``BenchmarkResult`` as ``{agg}_{field}``.

        The default declares ``success`` with ``"mean"`` (= success rate).
        Override to add benchmark-specific metrics.
        """
        return {"success": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        """Return benchmark defaults and metadata. Optional override."""
        return {}

    def cleanup(self) -> None:
        """Release benchmark resources (environments, renderers, etc.). Optional override."""

    def render(self) -> np.ndarray | None:
        """Render current env state as image. Optional override."""
        return None


# ---------------------------------------------------------------------------
# Step-based convenience subclass
# ---------------------------------------------------------------------------


class StepBenchmark(Benchmark, ABC):
    """Sync step-based benchmark (simulations).

    Subclasses implement:
        - ``reset(task)`` → initial_raw_obs (store env on self)
        - ``step(action)`` → ``StepResult``
        - ``make_obs(raw_obs, task)`` → observation dict for model server
        - ``check_done(step_result)`` → bool  (default: ``step_result.done``)
        - ``get_step_result(step_result)`` → EpisodeResult (abstract)

    The class auto-bridges these to the async parent API via internal state.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_result: StepResult = StepResult(obs=None, reward=0.0, done=False, info={})
        self._task: Task = {}
        self._t0: float = 0.0

    # -- abstract: user implements ----------------------------------------

    @abstractmethod
    def reset(self, task: Task) -> Any:
        """Reset environment for *task*. Returns initial raw observation (store env on self)."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply action to environment and return :class:`StepResult`."""

    @abstractmethod
    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        """Convert raw observation to the model server's :class:`Observation` format."""

    @abstractmethod
    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        """Extract episode result from the final :class:`StepResult`."""

    def check_done(self, step_result: StepResult) -> bool:
        """Check if episode should terminate. Default: ``step_result.done``."""
        return step_result.done

    # -- async bridge (auto-provided) -------------------------------------
    # NOTE: reset(), step(), make_obs() are sync and may block the event loop
    # when MuJoCo physics or image rendering takes non-trivial time.
    # Offloading them via anyio.to_thread.run_sync() could improve
    # concurrency under high shard counts.  However, the default anyio
    # thread-pool limiter only has 40 tokens — under 50+ concurrent shards
    # a dedicated CapacityLimiter is needed to avoid starvation
    # (see _DECODE_LIMITER in serve.py for an example).

    async def start_episode(self, task: Task) -> None:
        self._t0 = time.monotonic()
        self._task = task
        raw_obs = self.reset(task)
        self._last_result = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        self._last_result = self.step(action)

    async def get_observation(self) -> Observation:
        return self.make_obs(self._last_result.obs, self._task)

    async def is_done(self) -> bool:
        return self.check_done(self._last_result)

    async def get_time(self) -> float:
        return time.monotonic() - self._t0

    async def get_result(self) -> EpisodeResult:
        return self.get_step_result(self._last_result)
