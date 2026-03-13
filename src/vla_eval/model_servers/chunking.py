"""Action chunking buffer for PredictModelServer."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np


def _ensemble_newest(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return new


def _ensemble_average(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return (old + new) / 2.0


def _make_ensemble_ema(alpha: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def _ema(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        return alpha * new + (1 - alpha) * old

    return _ema


def get_ensemble_fn(
    strategy: str | Callable[[np.ndarray, np.ndarray], np.ndarray],
    ema_alpha: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Resolve ensemble strategy to a callable."""
    if not isinstance(strategy, str):
        assert callable(strategy), f"Strategy must be a string or a callable, got {type(strategy)}"
        return strategy
    if strategy == "newest":
        return _ensemble_newest
    if strategy == "average":
        return _ensemble_average
    if strategy == "ema":
        return _make_ensemble_ema(ema_alpha)
    raise ValueError(f"Unknown action_ensemble strategy: {strategy!r}")


class ActionChunkBuffer:
    """FIFO buffer for action chunking with optional ensemble blending.

    When a new chunk is pushed while actions remain in the buffer, the
    **overlapping** portion is blended using the ensemble function.
    Non-overlapping new actions are appended as-is.

    Example with buffer ``[a, b, c]`` and new chunk ``[x, y]``:
        Result = ``[ensemble(a, x), ensemble(b, y), c]``

    Pop returns and removes the first action (O(1) via deque).
    """

    def __init__(self, chunk_size: int, ensemble_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.chunk_size = chunk_size
        self.ensemble_fn = ensemble_fn
        self._queue: deque[np.ndarray] = deque()

    def push_chunk(self, actions: np.ndarray) -> None:
        """Push a new action chunk. actions shape: (chunk_size, action_dim)."""
        if self._queue:
            # Ensemble overlapping actions
            remaining = list(self._queue)
            self._queue.clear()
            overlap = min(len(remaining), len(actions))
            for i in range(overlap):
                ensembled = self.ensemble_fn(remaining[i], actions[i])
                self._queue.append(ensembled)
            # Add non-overlapping new actions
            for i in range(overlap, len(actions)):
                self._queue.append(actions[i])
        else:
            for action in actions:
                self._queue.append(action)

    def pop(self) -> np.ndarray | None:
        """Pop the next single action. Returns None if buffer is empty."""
        if self._queue:
            return self._queue.popleft()
        return None

    @property
    def empty(self) -> bool:
        return len(self._queue) == 0

    def clear(self) -> None:
        self._queue.clear()
