"""EpisodeRunner ABC."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

import websockets

from vla_eval.benchmarks.base import Benchmark
from vla_eval.connection import ServerEpisodeError
from vla_eval.recording import EpisodeRecorder, RecordingError
from vla_eval.types import EpisodeResult, Task

logger = logging.getLogger(__name__)


class EpisodeError(RuntimeError):
    """The subclass names the phase that raised; it becomes the episode's ``failure_reason``."""

    phase: ClassVar[str]


class EnvStartError(EpisodeError):
    phase = "env_start"  # start_episode + first observation


class ModelStartError(EpisodeError):
    phase = "model_start"  # the server failed EPISODE_START


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


def failure_reason(error: BaseException) -> str:
    """The ``failure_reason`` of an episode that ended in ``error``."""
    if isinstance(error, ConnectionError):
        return "server_unreachable"
    if isinstance(error, websockets.exceptions.ConnectionClosed):
        return "connection_closed"
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, ServerEpisodeError):
        return "model_act"
    return error.phase if isinstance(error, EpisodeError) else "exception"


def episode_status(result: Mapping[str, Any]) -> Literal["success", "fail"]:
    """The status of an episode that ended with ``result``."""
    metrics = result.get("metrics")
    return "success" if isinstance(metrics, dict) and metrics.get("success") else "fail"


def episode_end_payload(result: Mapping[str, Any] | None = None, error: BaseException | None = None) -> dict[str, Any]:
    """``EPISODE_END`` payload: ``result`` with its ``status``, or ``status: error`` and why (``error``)."""
    if error is not None:
        return {"status": "error", "failure_reason": failure_reason(error)}
    assert result is not None
    return {**result, "status": episode_status(result), "failure_reason": None}


async def end_failed_episode(conn: Any, error: BaseException) -> None:
    """Tell the server the episode failed with ``error``; a failure to tell never replaces ``error``."""
    try:
        await conn.end_episode(episode_end_payload(error=error))
    except Exception as exc:
        logger.debug("EPISODE_END not sent: %s: %s", type(exc).__name__, exc)


async def start_model_episode(conn: Any, payload: dict[str, Any]) -> None:
    """``EPISODE_START``, awaiting its ``EPISODE_READY``.  A start that timed out may still begin, so it is ended;
    one the server failed (``ModelStartError``) never began."""
    try:
        with phase(ModelStartError):
            await conn.start_episode(payload)
    except TimeoutError as exc:
        await end_failed_episode(conn, exc)
        raise


def episode_start_payload(task: Task, recorder: EpisodeRecorder | None, **extra: Any) -> dict[str, Any]:
    """``EPISODE_START`` payload: the task's plain fields, plus the recording handle when one is active."""
    payload: dict[str, Any] = {
        "task": {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))},
        **extra,
    }
    if recorder is not None and recorder.is_active:
        payload["recording"] = {
            "sid": recorder.sid,
            "eid": recorder.eid,
            "eval_id": recorder.eval_id,
            "db_path": recorder.db_path,
        }
    return payload


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
