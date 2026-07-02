"""DuoBench helpers kept out of ``benchmark.py`` so the benchmark class stays the
episode lifecycle.

* ``extract_rgb`` — pull an HWC uint8 RGB array out of RCS's nested per-camera obs payload.
* ``resolve_enum`` — case-insensitive lookup of an RCS enum member by name (RCS enum
  member names aren't uniformly uppercase, so a plain ``enum_cls[name]`` won't do).
* ``ensure_mujoco_arena_memory`` — preempt MuJoCo's fatal ``mj_stackAlloc`` overflow.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def extract_rgb(payload: Any) -> np.ndarray | None:
    """Pull an HWC uint8 RGB array out of one camera's nested ``frames[cam]`` payload.

    RCS delivers each camera as ``{"rgb": {"data": <HWC array>}}``.  Returns None
    unless that yields a 3-D (H, W, C) image; slices 4-channel RGBA down to RGB.
    """
    rgb = payload.get("rgb") if isinstance(payload, dict) else None
    arr = rgb.get("data") if isinstance(rgb, dict) else None
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return None
    return np.ascontiguousarray(arr[:, :, :3])


def resolve_enum(enum_cls: Any, name: str, field: str) -> Any:
    """Look up ``name`` among ``enum_cls`` members, case-insensitively, with a helpful
    error.  RCS enum member names aren't uniformly uppercase, so a plain
    ``enum_cls[name]`` can't match the lowercase strings we accept in config.
    """
    for member in enum_cls:
        if member.name.lower() == name.lower():
            return member
    valid = sorted(m.name.lower() for m in enum_cls)
    raise ValueError(f"{field}={name!r} is not a valid {enum_cls.__name__} member; expected one of {valid}")


# Process-wide flag: the MuJoCo arena patch is applied at most once.
_MJ_ARENA_PATCHED = False

# MuJoCo sizes each scene's mjData stack/arena from the *static* (resting) model,
# which is blind to contact-heavy transients: a weak policy that jams the grippers
# and objects together spikes the live constraint count past that estimate, and
# MuJoCo fatally aborts the process with "mj_stackAlloc: insufficient memory".  We
# give every compiled scene a fixed, generous arena via ``MjSpec.memory`` (the API
# form of MJCF ``<size memory>``) — 256 MiB is ~18x the static size, one arena per
# env.  Tune with ``DUOBENCH_MJ_ARENA_MB`` (MiB; 0 disables).
_DEFAULT_MJ_ARENA_BYTES = 256 * 1024 * 1024


def ensure_mujoco_arena_memory() -> None:
    """Raise the MuJoCo arena floor for every DuoBench scene (preempts mj_stackAlloc).

    rcs compiles each scene in ``ModelComposer.get_model``; we wrap it to bump
    ``spec.memory`` to a floor before ``compile()``.  rcs exposes no config knob for
    arena sizing, and editing its installed source would fork the dependency — so
    wrapping the (pure-Python) composer at runtime is the cleanest hook.
    Idempotent and best-effort — a failure leaves MuJoCo's auto sizing in place.
    """
    global _MJ_ARENA_PATCHED
    if _MJ_ARENA_PATCHED:
        return
    _MJ_ARENA_PATCHED = True

    floor = _DEFAULT_MJ_ARENA_BYTES
    if (override := os.environ.get("DUOBENCH_MJ_ARENA_MB")) is not None:
        try:
            floor = int(float(override) * 1024 * 1024)
        except ValueError:
            logger.warning("[duobench] ignoring bad DUOBENCH_MJ_ARENA_MB=%r", override)
    if floor <= 0:
        return

    try:
        from rcs.sim.composer import ModelComposer
    except Exception as e:  # noqa: BLE001
        logger.warning("[duobench] MuJoCo arena bump skipped (no rcs composer: %s)", e)
        return

    orig_get_model = ModelComposer.get_model
    auto = (1 << 64) - 1  # MjSpec.memory sentinel meaning "auto-size"

    def _get_model_with_arena(self):
        try:
            current = int(self.spec.memory)
            if current == auto or current < floor:
                self.spec.memory = floor
        except Exception as e:  # noqa: BLE001
            logger.warning("[duobench] could not raise MjSpec.memory (%s)", e)
        return orig_get_model(self)

    ModelComposer.get_model = _get_model_with_arena
    logger.info("[duobench] MuJoCo arena floor set to %d MiB/scene.", floor // (1024 * 1024))
