"""Shared type definitions for the vla-eval wire protocol."""

from __future__ import annotations

from typing import Any, TypeAlias

# ---------------------------------------------------------------------------
# Wire protocol types
# ---------------------------------------------------------------------------

Observation: TypeAlias = dict[str, Any]
"""Observation dict passed from benchmark to model server over the wire.

The structure is intentionally open — benchmarks decide what keys to include.
The following convention (inspired by RLinf's ``EnvOutput.obs``) is adopted
by all built-in benchmarks and recommended for new integrations:

Recommended keys:
    images (dict[str, np.ndarray]):
        Mapping of camera name to image array ``[H, W, C]``.
    task_description (str):
        Natural-language task instruction.
    state (np.ndarray):
        Proprioceptive state vector (joint positions, gripper, …).

Example (LIBERO, ManiSkill, RoboCasa, …)::

    {
        "images": {"agentview": np.zeros((256, 256, 3), dtype=np.uint8)},
        "task_description": "pick up the red block",
        "state": np.zeros(7),
    }
"""

Action: TypeAlias = dict[str, Any]
"""Action dict returned by the model server in response to an observation.

Common practice:
    actions (np.ndarray):
        Raw action vector ``(action_dim,)`` or action chunk ``(chunk_size, action_dim)``.
"""

Task: TypeAlias = dict[str, Any]
"""Task descriptor returned by ``Benchmark.get_tasks()`` and threaded
through the episode lifecycle (``start_episode``, ``make_obs``, runners).

Common practice:
    name (str): Human-readable task name.
    suite (str): Task suite / category grouping.
"""

EpisodeResult: TypeAlias = dict[str, Any]
"""Episode result dict returned by ``Benchmark.get_result()``.

At minimum contains ``{"success": bool}``.  Benchmarks may include
additional metrics (e.g. ``completed_subtasks``, ``reward``).
"""
