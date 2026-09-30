"""EpisodeRunner ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import websockets

from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder, RecordingError
from vla_eval.types import EpisodeResult, Task


class EpisodeError(RuntimeError):
    """The subclass names the phase that raised; it becomes the episode's ``failure_reason``."""

    phase: ClassVar[str]


class EnvStartError(EpisodeError):
    phase = "env_start"  # start_episode + first observation


class EnvStepError(EpisodeError):
    phase = "env_step"  # apply_action / is_done / get_observation / get_result


class ModelActError(EpisodeError):
    phase = "model_act"  # conn.act (transport errors keep their own handling)


@contextmanager
def phase(error: type[EpisodeError]) -> Iterator[None]:
    try:
        yield
    except (EpisodeError, ConnectionError, TimeoutError, websockets.exceptions.ConnectionClosed, RecordingError):
        raise
    except Exception as exc:
        raise error(f"{type(exc).__name__}: {exc}") from exc


class EpisodeRunner(ABC):
    """Abstract base class for episode execution strategies."""

    @abstractmethod
    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a single episode and return the result.

        ``recorder`` (when active) is forwarded to ``benchmark.start_episode``
        so video frames and step rows are captured. The runner also bundles
        ``{sid, eid, eval_id, db_path}`` into the ``EPISODE_START`` WS payload
        so model-server code (e.g. a training pipeline) can open its own
        :class:`vla_eval.recording.StepRecorder` against the same SQLite file
        and field-union its inference traces with the benchmark's step rows.
        """
