"""ActionBuffer: stores the latest action for real-time episode runners.

In real-time mode, the environment steps on a fixed wall-clock schedule.
If no new action has arrived from the model server, the hold policy
determines what action to use.

Hold policies:
    - ``repeat_last``: replay the most recent action (default).
    - ``zero``: output a zero-filled action dict.
    - A callable ``() -> dict`` for custom fallback logic.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np

from vla_eval.types import Action


class ActionBuffer:
    """Thread-safe buffer for the latest action from the model server.

    Attributes:
        hold_policy: Strategy when no new action is available.
        action_dim: Dimension of the action vector (for ``zero`` policy).
    """

    def __init__(
        self,
        hold_policy: str | Callable[[], Action] = "repeat_last",
        action_dim: int = 7,
    ) -> None:
        self._lock = threading.Lock()
        self._latest_action: Action | None = None
        self._new_since_last_get: bool = False
        self._update_count: int = 0
        self._stale_count: int = 0
        self._last_update_time: float | None = None
        self.hold_policy = hold_policy
        self.action_dim = action_dim

    def update(self, action: Action) -> None:
        """Called by the on_action callback when a new action arrives."""
        with self._lock:
            self._latest_action = action
            self._new_since_last_get = True
            self._update_count += 1
            self._last_update_time = time.monotonic()

    def get(self) -> Action:
        """Return the current action (new or held).

        Returns the latest action if one has arrived since the last ``get()``.
        Otherwise, applies the hold policy.
        """
        with self._lock:
            if self._new_since_last_get:
                self._new_since_last_get = False
                assert self._latest_action is not None
                return self._latest_action

            # Hold policy
            self._stale_count += 1
            if self._latest_action is not None:
                return self._apply_hold_policy()

            # No action ever received — return zero
            return self._zero_action()

    def has_action(self) -> bool:
        """True if at least one action has been received."""
        with self._lock:
            return self._latest_action is not None

    def is_new(self) -> bool:
        """True if a new action arrived since the last ``get()``."""
        with self._lock:
            return self._new_since_last_get

    @property
    def update_count(self) -> int:
        """Total number of actions received."""
        return self._update_count

    @property
    def stale_count(self) -> int:
        """Number of times hold policy was applied."""
        return self._stale_count

    @property
    def last_update_time(self) -> float | None:
        """Monotonic timestamp of the last action update."""
        return self._last_update_time

    def reset(self) -> None:
        """Clear buffer state for a new episode."""
        with self._lock:
            self._latest_action = None
            self._new_since_last_get = False
            self._update_count = 0
            self._stale_count = 0
            self._last_update_time = None

    def _apply_hold_policy(self) -> Action:
        """Apply hold policy when no new action is available."""
        if not isinstance(self.hold_policy, str) and callable(self.hold_policy):
            return self.hold_policy()
        if self.hold_policy == "repeat_last":
            assert self._latest_action is not None
            return self._latest_action
        if self.hold_policy == "zero":
            return self._zero_action()
        raise ValueError(f"Unknown hold_policy: {self.hold_policy!r}")

    def _zero_action(self) -> Action:
        return {"actions": np.zeros(self.action_dim, dtype=np.float32)}

    def get_metrics(self) -> dict[str, Any]:
        """Return real-time metrics for this buffer."""
        total = self._update_count + self._stale_count
        return {
            "update_count": self._update_count,
            "stale_count": self._stale_count,
            "stale_action_ratio": self._stale_count / total if total > 0 else 0.0,
        }
