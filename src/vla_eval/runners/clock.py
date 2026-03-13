"""Clock with adjustable pace for episode runners."""

from __future__ import annotations

import math
import time

import anyio


class Clock:
    """Wall-clock with pace control.

    Args:
        pace: Speed multiplier. 1.0 = real-time, 10.0 = 10x faster,
              math.inf = no waiting.
    """

    def __init__(self, pace: float = 1.0) -> None:
        self.pace = pace
        self._t0 = time.monotonic()

    def time(self) -> float:
        """Elapsed seconds since last reset."""
        return time.monotonic() - self._t0

    async def wait_until(self, t: float) -> None:
        """Wait until clock reaches *t* seconds, scaled by pace.

        Always yields at least once so background tasks (e.g. the
        action listener) get a chance to run.
        """
        dt = t - self.time()
        if dt > 0 and not math.isinf(self.pace):
            await anyio.sleep(dt / self.pace)
        else:
            await anyio.sleep(0)

    def reset(self) -> None:
        """Reset epoch to now."""
        self._t0 = time.monotonic()
