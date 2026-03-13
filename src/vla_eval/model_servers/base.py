"""Base ModelServer ABC and SessionContext."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, Literal

from vla_eval.types import Action, Observation

# Type alias for the async send_action callback injected by the framework.
# NOTE: Dict (not dict) for Python 3.8 compatibility in type aliases.
SendActionFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


class SessionContext:
    """Per-session state passed to ModelServer callbacks.

    Attributes:
        session_id: Persistent across episodes within one WebSocket connection.
        episode_id: Regenerated on each ``EPISODE_START``.
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
        mode: Literal["sync", "realtime"] = "sync",
    ) -> None:
        self._session_id = session_id
        self._episode_id = episode_id
        self._mode: Literal["sync", "realtime"] = mode
        self._step = 0
        self._send_action_fn: SendActionFn | None = None  # set by framework

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def mode(self) -> Literal["sync", "realtime"]:
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
    """Base async model server. For advanced use cases only."""

    @abstractmethod
    async def on_observation(self, obs: Observation, ctx: SessionContext) -> None:
        """Called when an observation arrives. Run inference and call ctx.send_action()."""

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        """Called at episode start. Override to reset model state."""

    async def on_episode_end(self, result: dict[str, Any], ctx: SessionContext) -> None:
        """Called at episode end. Optional."""
