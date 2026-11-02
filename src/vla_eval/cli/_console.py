"""Shared CLI console helpers."""

from __future__ import annotations

import functools


@functools.cache
def stderr_console():
    """Return a shared rich Console writing to stderr (lazy import)."""
    from rich.console import Console

    return Console(stderr=True, highlight=False)
