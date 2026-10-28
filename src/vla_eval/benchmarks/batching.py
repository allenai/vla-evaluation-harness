"""Batched benchmarks: one simulator with several envs, exposed as one benchmark view per env.

The orchestrator drives each episode through its own runner, while a batched simulator
resets and steps its envs together. :class:`BatchCore` joins the two: every view submits
its reset / action, and once all live views have submitted, one of them runs the batched
call on a worker thread and hands each view its own slice.

A view leaves the batch when its episode ends, its runner fails (e.g. a model error) or
its reset asks for a retry; the remaining views stop waiting for it. An exception from
the simulator itself fails every view of the batch, since the shared state is no longer
trustworthy.

A benchmark opts in by returning ``batch_size() > 1`` and building its views with
:func:`open_views` over a :class:`BatchCore` subclass that implements the two batched calls.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import anyio
from anyio.to_thread import run_sync as _run_in_thread

from vla_eval.benchmarks.base import Benchmark, StepResult
from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder
from vla_eval.specs import DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

#: Longest a view waits at a barrier while no batched call is running. A view stuck
#: longer than this (e.g. on model inference) fails the whole batch instead of hanging it.
BARRIER_TIMEOUT_S = 600.0


class BatchCore(ABC):
    """Owns the batch's barriers; the simulator itself stays on the parent benchmark.

    Subclasses implement :meth:`_reset_batch` / :meth:`_step_batch` (run on a worker thread)
    and the per-slot :meth:`make_obs` / :meth:`get_result`. A batched call returns
    ``(results, finished)``: one result per submitted slot, and the slots whose episode is
    over. A reset result that is an exception (e.g. :class:`~vla_eval.runners.base.RetryEpisode`)
    is raised in that slot only; such slots must also be in ``finished``.
    """

    def __init__(self, bench: Benchmark, size: int) -> None:
        """Bind ``size`` slots (at most ``bench.batch_size()``) to ``bench``; every slot starts live."""
        if not 1 <= size <= bench.batch_size():
            raise ValueError(f"{type(bench).__name__}: batch of {size} does not fit num_envs={bench.batch_size()}")
        self.bench = bench
        self.size = size
        self._live: set[int] = set(range(size))
        self._pending: dict[int, tuple[str, Any]] = {}
        self._results: dict[int, Any] = {}
        self._running = False
        self._error: BaseException | None = None
        self._event = anyio.Event()

    # -- barrier -----------------------------------------------------------

    def _wake(self) -> None:
        """Wake every view waiting at the barrier and arm a fresh event for the next wait."""
        self._event.set()
        self._event = anyio.Event()

    def leave(self, slot: int) -> None:
        """Stop waiting for ``slot`` (its episode ended or its runner gave up)."""
        self._live.discard(slot)
        self._pending.pop(slot, None)
        self._wake()

    def _failed(self) -> RuntimeError:
        """The error raised in every slot once the batch has failed."""
        return RuntimeError(f"{type(self.bench).__name__} batch failed: {self._error}")

    async def _rendezvous(self, slot: int, kind: str, payload: Any) -> Any:
        """Submit ``slot``'s call and wait for its result; the last live slot to submit runs the batch."""
        if self._error is not None:
            raise self._failed()
        if slot not in self._live:
            raise RuntimeError(f"{type(self.bench).__name__}: slot {slot} already left the batch")
        self._pending[slot] = (kind, payload)
        self._wake()
        while True:
            if self._error is not None:
                raise self._failed()
            if slot in self._results:
                return self._results.pop(slot)
            if not self._running and self._live <= self._pending.keys():
                await self._run()
                continue
            event = self._event
            with anyio.move_on_after(BARRIER_TIMEOUT_S) as scope:
                await event.wait()
            if scope.cancelled_caught and not self._running:
                waiting = sorted(self._live - self._pending.keys())
                self._error = RuntimeError(f"barrier timed out after {BARRIER_TIMEOUT_S:.0f}s waiting for {waiting}")
                self._wake()

    async def _run(self) -> None:
        """Run the pending calls as one batched call on a worker thread and hand out the results."""
        batch = dict(self._pending)
        self._pending.clear()
        kinds = {kind for kind, _ in batch.values()}
        self._running = True
        try:
            if len(kinds) != 1:
                raise RuntimeError(f"{type(self.bench).__name__}: slots out of step, mixed calls {sorted(kinds)}")
            payloads = {slot: payload for slot, (_, payload) in batch.items()}
            run = self._reset_batch if kinds == {"reset"} else self._step_batch
            results, finished = await _run_in_thread(run, payloads)
        except BaseException as exc:
            self._error = exc
            raise
        finally:
            self._running = False
            self._wake()
        self._results.update(results)
        self._live -= finished

    # -- public API used by the views --------------------------------------

    async def reset(self, slot: int, task: Task, recorder: EpisodeRecorder) -> Any:
        """Reset ``slot`` to ``task``; return its raw observation or raise its per-slot exception."""
        result = await self._rendezvous(slot, "reset", (task, recorder))
        if isinstance(result, BaseException):
            raise result
        return result

    async def step(self, slot: int, action: Action) -> StepResult:
        """Step ``slot`` with ``action`` together with the other live slots."""
        return await self._rendezvous(slot, "step", action)

    # -- implemented per benchmark -----------------------------------------

    @abstractmethod
    def _reset_batch(self, payloads: dict[int, tuple[Task, EpisodeRecorder]]) -> tuple[dict[int, Any], set[int]]:
        """Reset the submitted slots together; each result is the slot's raw observation or an exception."""

    @abstractmethod
    def _step_batch(self, payloads: dict[int, Action]) -> tuple[dict[int, StepResult], set[int]]:
        """Step the submitted slots together; ``finished`` holds the slots whose episode ended."""

    @abstractmethod
    def make_obs(self, slot: int, raw_obs: Any, task: Task) -> Observation:
        """The model server's observation for ``slot``."""

    @abstractmethod
    def get_result(self, slot: int, last: StepResult) -> EpisodeResult:
        """The episode result of ``slot`` from its last step."""


class SlotView(Benchmark):
    """One env of a batch, driven through the single-episode interface by its own runner."""

    def __init__(self, core: BatchCore, slot: int) -> None:
        """View ``slot`` of ``core``."""
        self._core = core
        self._slot = slot
        self._bench = core.bench
        self._task: Task = {}
        self._t0 = 0.0
        self._last = StepResult(obs=None, reward=0.0, done=False, info={})

    async def start_episode(self, task: Task, recorder: EpisodeRecorder | None = None) -> None:
        """Reset this slot to ``task`` at the batch's next reset."""
        self._t0 = time.monotonic()
        self._task = task
        raw_obs = await self._core.reset(self._slot, task, recorder or NullEpisodeRecorder())
        self._last = StepResult(obs=raw_obs, reward=0.0, done=False, info={})

    async def apply_action(self, action: Action) -> None:
        """Step this slot with ``action`` at the batch's next step."""
        self._last = await self._core.step(self._slot, action)

    async def get_observation(self) -> Observation:
        """The model server's observation for this slot's last step."""
        return self._core.make_obs(self._slot, self._last.obs, self._task)

    async def is_done(self) -> bool:
        """Whether this slot's episode has ended."""
        return self._last.done

    async def get_time(self) -> float:
        """Seconds since this slot's episode started."""
        return time.monotonic() - self._t0

    async def get_result(self) -> EpisodeResult:
        """The episode result of this slot."""
        return self._core.get_result(self._slot, self._last)

    def get_tasks(self) -> list[Task]:
        """The parent benchmark's tasks."""
        return self._bench.get_tasks()

    def get_action_spec(self) -> dict[str, DimSpec]:
        """The parent benchmark's action spec."""
        return self._bench.get_action_spec()

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """The parent benchmark's observation spec."""
        return self._bench.get_observation_spec()

    def get_metric_keys(self) -> dict[str, str]:
        """The parent benchmark's metric keys."""
        return self._bench.get_metric_keys()

    def get_metadata(self) -> dict[str, Any]:
        """The parent benchmark's metadata."""
        return self._bench.get_metadata()

    def cleanup(self) -> None:
        """Leave the batch; the parent benchmark keeps the simulator."""
        # The simulator belongs to the parent benchmark; only release this slot's place in the batch.
        self._core.leave(self._slot)


def open_views(core: BatchCore) -> list[Benchmark]:
    """One :class:`SlotView` per slot of ``core``."""
    return [SlotView(core, slot) for slot in range(core.size)]
