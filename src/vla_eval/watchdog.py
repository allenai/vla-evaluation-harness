"""Progress watchdog: panic a wedged benchmark instead of hanging forever.

A blocking native sim call (SAPIEN/Vulkan) freezes the event loop, so no
await-timeout can fire. Only a separate OS thread keeps running, and only
os._exit reliably kills a process stuck in native code.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

PANIC_EXIT_CODE = 124  # `timeout(1)` convention — marks a watchdog kill


class ProgressWatchdog:
    """Daemon thread that os._exit()s the process if no progress for timeout_s."""

    def __init__(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError(f"watchdog timeout_s must be positive, got {timeout_s}")
        self._timeout_s = float(timeout_s)
        self._last = time.monotonic()
        self._phase = "startup"

    def pet(self, phase: str) -> None:
        """Record progress; phase is surfaced if the watchdog later panics."""
        # No lock: each is a lone attribute store, atomic under the GIL — and
        # the watchdog only ever reads these fields, never read-modify-writes.
        self._last = time.monotonic()
        self._phase = phase

    def idle_s(self) -> float:
        return time.monotonic() - self._last

    def phase(self) -> str:
        return self._phase

    def start(self) -> ProgressWatchdog:
        threading.Thread(target=self._loop, name="progress-watchdog", daemon=True).start()
        return self

    def _loop(self) -> None:
        poll = min(30.0, self._timeout_s / 4)
        while True:
            time.sleep(poll)
            idle = self.idle_s()
            if idle > self._timeout_s:
                logger.critical(
                    "Progress watchdog: no progress for %.0fs (phase: %r) — process wedged "
                    "(likely a native SAPIEN/Vulkan hang); os._exit(%d).",
                    idle,
                    self.phase(),
                    PANIC_EXIT_CODE,
                )
                logging.shutdown()  # os._exit skips atexit — flush the panic log first
                os._exit(PANIC_EXIT_CODE)


_watchdog: ProgressWatchdog | None = None


def start(timeout_s: float) -> None:
    """Arm the process-global watchdog (idempotent)."""
    global _watchdog
    if _watchdog is None:
        _watchdog = ProgressWatchdog(timeout_s).start()
        logger.info("Progress watchdog armed: %.0fs stall timeout.", timeout_s)


def pet(phase: str) -> None:
    """Pet the process-global watchdog; no-op if it was never armed."""
    if _watchdog is not None:
        _watchdog.pet(phase)
