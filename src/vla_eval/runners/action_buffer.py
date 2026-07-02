"""ActionBuffer: holds the latest action for real-time episode runners.

In real-time mode the environment steps on a fixed wall-clock schedule. When no
fresh action has arrived from the model server since the last tick, the buffer
falls back to a *hold* action produced by ``hold_fn`` — the embodiment's safe
reuse for a stale tick.

``hold_fn`` receives the last *fresh* action (``None`` before the first one) and
returns the action to command:
    * absolute-position control → repeat the last action (hold the target);
    * delta / velocity control → a fixed null action (stay put) — repeating a
      delta would keep moving.

The policy lives on the benchmark (:meth:`vla_eval.benchmarks.base.Benchmark.get_hold_action`),
not in a config string, because only the embodiment knows what "do nothing" means.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from vla_eval.types import Action


class ActionBuffer:
    """Thread-safe latest-action buffer with an embodiment-defined stale hold.

    Args:
        hold_fn: ``(last_fresh_action | None) -> Action`` used on stale ticks
            (and before the first action arrives, with ``None``).
    """

    def __init__(self, hold_fn: Callable[[Action | None], Action]) -> None:
        self._lock = threading.Lock()
        self._latest_action: Action | None = None
        self._last_fresh: Action | None = None
        self._new_since_last_get: bool = False
        self._update_count: int = 0
        self._stale_count: int = 0
        self._last_update_time: float | None = None
        self.hold_fn = hold_fn

    def update(self, action: Action) -> None:
        """Called by the on_action callback when a new action arrives."""
        with self._lock:
            self._latest_action = action
            self._new_since_last_get = True
            self._update_count += 1
            self._last_update_time = time.monotonic()

    def get(self) -> Action:
        """Return the fresh action if one arrived since the last ``get()``,
        otherwise the hold action from ``hold_fn(last_fresh_action)``."""
        with self._lock:
            if self._new_since_last_get:
                self._new_since_last_get = False
                assert self._latest_action is not None
                self._last_fresh = self._latest_action
                return self._latest_action
            self._stale_count += 1
            return self.hold_fn(self._last_fresh)

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
        """Number of times the hold action was used."""
        return self._stale_count

    @property
    def last_update_time(self) -> float | None:
        """Monotonic timestamp of the last action update."""
        return self._last_update_time

    def reset(self) -> None:
        """Clear buffer state for a new episode."""
        with self._lock:
            self._latest_action = None
            self._last_fresh = None
            self._new_since_last_get = False
            self._update_count = 0
            self._stale_count = 0
            self._last_update_time = None

    def get_metrics(self) -> dict[str, Any]:
        """Return real-time metrics for this buffer."""
        total = self._update_count + self._stale_count
        return {
            "update_count": self._update_count,
            "stale_count": self._stale_count,
            "stale_action_ratio": self._stale_count / total if total > 0 else 0.0,
        }
