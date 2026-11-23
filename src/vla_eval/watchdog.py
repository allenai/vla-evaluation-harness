"""Progress watchdog: panic a wedged benchmark instead of hanging forever.

A blocking native sim call (SAPIEN/Vulkan) freezes the event loop, so no
await-timeout can fire. Only a separate OS thread keeps running, and only
os._exit reliably kills a process stuck in native code.

A native call that blocks while *holding the GIL* (e.g. PhysX fetch_results
waiting on a GPU that hit an Xid fault) starves that Python thread too, so a
faulthandler timer runs alongside it as a backstop: it lives in a C thread that
needs no GIL, fires a little after the Python thread would have, dumps every
thread's stack and _exit(1)s.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

PANIC_EXIT_CODE = 124  # `timeout(1)` convention — marks a watchdog kill
# The GIL-free backstop cannot choose its code: faulthandler _exit(1)s after printing "Timeout (h:mm:ss)!".


class ProgressWatchdog:
    """Daemon thread that os._exit()s the process if no progress for timeout_s."""

    def __init__(self, timeout_s: float) -> None:
        if timeout_s <= 0:
            raise ValueError(f"watchdog timeout_s must be positive, got {timeout_s}")
        self._timeout_s = float(timeout_s)
        self._last = time.monotonic()
        self._phase = "startup"
        self._stopped = threading.Event()
        # The backstop is re-armed at most once per _rearm_s, and its deadline padded past the Python thread's
        # (which polls every <= _rearm_s), so it only fires when that thread could not run.
        self._rearm_s = min(60.0, self._timeout_s / 4)
        self._armed = 0.0
        self._fd = 2
        self._running = False

    def pet(self, phase: str | None = None) -> None:
        """Record progress; phase (kept when None) is surfaced if the watchdog later panics."""
        # No lock: each is a lone attribute store, atomic under the GIL — and
        # the watchdog only ever reads these fields, never read-modify-writes.
        self._last = time.monotonic()
        if phase is not None:
            self._phase = phase
        if self._running and self._last - self._armed >= self._rearm_s:
            self._arm_backstop()

    def _arm_backstop(self) -> None:
        """(Re)start faulthandler's timer: it fires timeout + one to two re-arm periods after the last pet."""
        self._armed = time.monotonic()
        faulthandler.dump_traceback_later(self._timeout_s + 2 * self._rearm_s + 5, file=self._fd, exit=True)

    def idle_s(self) -> float:
        return time.monotonic() - self._last

    def phase(self) -> str:
        return self._phase

    def start(self) -> ProgressWatchdog:
        # The backstop writes to a private copy of the stderr we start with: a benchmark may later dup2 fd 2 onto
        # a pipe (RoboDojo tees it) whose reader is starved by the same wedge, and a blocking write there would
        # keep faulthandler from reaching its _exit.
        try:
            self._fd = os.dup(2)
        except OSError:
            self._fd = 2
        threading.Thread(target=self._loop, name="progress-watchdog", daemon=True).start()
        self._running = True
        self._arm_backstop()
        return self

    def stop(self) -> None:
        """Disarm: the thread exits at its next poll. For library callers whose run has ended."""
        self._stopped.set()
        self._running = False
        faulthandler.cancel_dump_traceback_later()
        if self._fd != 2:
            os.close(self._fd)
            self._fd = 2

    def _loop(self) -> None:
        poll = min(30.0, self._timeout_s / 4)
        while not self._stopped.wait(poll):
            idle = self.idle_s()
            if idle > self._timeout_s:
                self._panic(idle)

    def _panic(self, idle: float) -> None:
        # free() can block forever on a malloc lock held by a crashed thread, so skip logging; stderr may be a
        # pipe nobody drains (RoboDojo tees fd 2), so write non-blocking. Only os._exit must be reached.
        msg = (
            f"Progress watchdog: no progress for {idle:.0f}s (phase: {self.phase()!r}) — process "
            f"wedged (likely a native sim hang); os._exit({PANIC_EXIT_CODE}).\n"
        )
        try:
            os.set_blocking(2, False)
            os.write(2, msg.encode("utf-8", "replace"))
            faulthandler.dump_traceback(2, all_threads=True)
        except Exception:  # a closed/invalid stderr must not keep the process alive
            pass
        os._exit(PANIC_EXIT_CODE)


_watchdog: ProgressWatchdog | None = None


def start(timeout_s: float) -> None:
    """Arm the process-global watchdog (idempotent)."""
    global _watchdog
    if _watchdog is None:
        _watchdog = ProgressWatchdog(timeout_s).start()
        logger.info("Progress watchdog armed: %.0fs stall timeout.", timeout_s)


def pet(phase: str | None = None) -> None:
    """Pet the process-global watchdog (keeping the phase when None); no-op if it was never armed."""
    if _watchdog is not None:
        _watchdog.pet(phase)


def stop() -> None:
    """Disarm the process-global watchdog (idempotent) so it can be re-armed by a later run."""
    global _watchdog
    if _watchdog is not None:
        _watchdog.stop()
        _watchdog = None
