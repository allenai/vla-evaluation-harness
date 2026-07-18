"""Base ModelServer ABC and SessionContext."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Literal

from vla_eval.specs import DimSpec
from vla_eval.types import Action, Observation

# Type alias for the async send_action callback injected by the framework.
# NOTE: Dict (not dict) for Python 3.8 compatibility in type aliases.
SendActionFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class SessionContext:
    """Per-session state passed to ModelServer callbacks.

    Attributes:
        session_id: Persistent across episodes within one WebSocket connection.
            When the harness supplies a ``recording.sid`` in ``EPISODE_START``,
            it overrides the WS-level session id so external recording emits
            (e.g. opening a :class:`vla_eval.recording.StepRecorder` with
            ``ctx.session_id`` / ``ctx.episode_id``) land in the same SQLite
            bucket the harness wrote to.
        episode_id: Regenerated on each ``EPISODE_START``. When the harness
            supplies a ``recording.eid`` it is used verbatim.
        eval_id: Run-level identifier (from ``EPISODE_START.recording.eval_id``),
            or ``""`` when recording is disabled.
        recording_db_path: Path to the SQLite file the harness writes to, or
            ``""`` when recording is disabled. External callers (e.g.
            a training pipeline) open a :class:`vla_eval.recording.StepRecorder` at
            this path to record per-step inference traces alongside the
            benchmark's step rows.
        task: Task metadata dict sent by the client in ``EPISODE_START``.
        step: Number of observations processed so far in this episode.
            Inside ``predict()``, this is the count *before* the current
            observation (i.e. 0 on the first call).
        mode: Evaluation mode (currently always ``"sync"``).
        is_first: True when ``step == 0`` (first observation of the episode).
    """

    def __init__(
        self,
        session_id: str,
        episode_id: str,
        mode: Literal["sync", "live"] = "sync",
        eval_id: str = "",
        recording_db_path: str = "",
    ) -> None:
        self._session_id = session_id
        self._episode_id = episode_id
        self._mode: Literal["sync", "live"] = mode
        self._eval_id = eval_id
        self._recording_db_path = recording_db_path
        self._step = 0
        self._send_action_fn: SendActionFn | None = None  # set by framework

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def eval_id(self) -> str:
        return self._eval_id

    @property
    def recording_db_path(self) -> str:
        return self._recording_db_path

    @property
    def mode(self) -> Literal["sync", "live"]:
        return self._mode

    @property
    def step(self) -> int:
        return self._step

    @property
    def is_first(self) -> bool:
        return self._step == 0

    async def send_action(self, action: Action) -> None:
        """Send an action back to the benchmark client."""
        if self._send_action_fn is None:
            raise RuntimeError("send_action_fn not set by framework")
        await self._send_action_fn(action)

    def _increment_step(self) -> None:
        self._step += 1


class ModelServer(ABC):
    """Base async model server. For advanced use cases only.

    Subclasses MUST load all weights and complete any setup needed
    to serve inference inside ``__init__`` — including JIT warmup
    (e.g. one dummy forward) when first-call latency would otherwise
    blow the HELLO response budget. The framework starts accepting
    WebSocket connections as soon as ``__init__`` returns.
    """

    @abstractmethod
    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        """Called when an observation arrives. Run inference and call ctx.send_action()."""

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Called at episode start. Override to reset model state."""

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        """Called at episode end. Optional."""

    def get_action_spec(self) -> dict[str, DimSpec]:
        """Declare the action output format of this model server.

        Returns a ``{component_name: DimSpec}`` dict describing what this
        server produces.  The orchestrator compares this against the
        benchmark's action spec and warns on mismatches.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_action_spec()")

    def get_observation_spec(self) -> dict[str, DimSpec]:
        """Declare the observation input format this model server expects.

        Returns a ``{component_name: DimSpec}`` dict describing what this
        server needs from the benchmark.  The orchestrator warns when the
        benchmark doesn't provide a declared component.

        Override in every subclass — the default raises ``NotImplementedError``.
        """
        raise NotImplementedError(f"{type(self).__name__} must override get_observation_spec()")

    def get_observation_params(self) -> dict[str, Any]:
        """Declare observation requirements for this model.

        Returned params are sent in the HELLO response and auto-merged into
        benchmark params by the orchestrator. This lets the model server
        tell the benchmark what observation data it needs (e.g. wrist images,
        proprioceptive state) without requiring manual ``--param`` flags.

        Override in subclasses to auto-detect from model config, or pass
        an explicit ``observation_params`` dict to ``PredictModelServer``.
        """
        return {}

    def get_metadata(self) -> dict[str, Any]:
        """Return model/checkpoint provenance to persist with evaluation results."""
        return {}
