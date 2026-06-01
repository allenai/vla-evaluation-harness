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
