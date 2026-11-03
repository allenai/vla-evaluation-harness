"""Tests for the progress watchdog."""

from __future__ import annotations

import time

import pytest

from vla_eval import watchdog
from vla_eval.watchdog import ProgressWatchdog


def test_rejects_nonpositive_timeout() -> None:
    # A 0/negative timeout (e.g. a typo'd env var) would panic instantly.
    with pytest.raises(ValueError):
        ProgressWatchdog(timeout_s=0)


def test_pet_resets_idle() -> None:
    wd = ProgressWatchdog(timeout_s=100.0)
    time.sleep(0.05)
    assert wd.idle_s() >= 0.05
    wd.pet("episode 1")
    assert wd.idle_s() < 0.05
    assert wd.phase() == "episode 1"


def test_idle_grows_past_timeout() -> None:
    # _loop() os._exit()s once idle crosses timeout_s — check that boundary directly.
    wd = ProgressWatchdog(timeout_s=0.1)
    wd.pet("work")
    assert wd.idle_s() < 0.1
    time.sleep(0.15)
    assert wd.idle_s() > 0.1


def test_module_pet_is_noop_before_start() -> None:
    # Callers pet unconditionally — pet() before start() must not raise.
    watchdog.pet("safe no-op")


def test_stop_ends_the_thread() -> None:
    import threading
    import time

    from vla_eval.watchdog import ProgressWatchdog

    wd = ProgressWatchdog(0.2).start()
    thread = next(t for t in threading.enumerate() if t.name == "progress-watchdog")
    wd.stop()
    thread.join(1.0)
    assert not thread.is_alive()
    time.sleep(0.3)  # past the timeout: still here, so no os._exit fired


def test_pet_without_phase_keeps_phase_and_refreshes_idle():
    """Runners pet once per step without a phase: the orchestrator's phase text survives, the idle clock resets."""
    dog = ProgressWatchdog(timeout_s=100)
    dog.pet("bench task ep3")
    dog._last -= 50
    assert dog.idle_s() >= 50
    dog.pet()
    assert dog.idle_s() < 1 and dog.phase() == "bench task ep3"


_WEDGED_LOGGING_SCRIPT = """
import logging, threading, time
from vla_eval import watchdog

class Wedged(logging.Handler):
    # Stands in for a handler stuck on the malloc lock held by a crashed thread.
    def emit(self, record):
        threading.Event().wait()
    def flush(self):
        threading.Event().wait()

logging.getLogger().addHandler(Wedged())
watchdog.ProgressWatchdog(0.2).start()
time.sleep(30)
"""


def test_panic_exits_even_when_logging_is_wedged() -> None:
    # Regression: logging.shutdown() blocked on the malloc lock and os._exit never ran.
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-c", _WEDGED_LOGGING_SCRIPT], capture_output=True, text=True, timeout=20)
    assert proc.returncode == watchdog.PANIC_EXIT_CODE
    assert "Progress watchdog: no progress for" in proc.stderr
    assert "most recent call first" in proc.stderr  # faulthandler stack dump
