"""LIBERO-Plus benchmark implementation.

LIBERO-Plus (https://github.com/sylvestf/LIBERO-plus) is a robustness-analysis
extension of LIBERO that replaces each of the original 40 evaluation tasks
with ~10,030 systematically perturbed variants across seven axes: object
layout, camera viewpoints, robot initial states, language instructions,
lighting, background textures, and sensor noise.

Because the fork installs under the same ``libero`` package namespace as
vanilla LIBERO and registers suites under identical names
(``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``,
``libero_90``), this class is a thin subclass of :class:`LIBEROBenchmark`
that delegates all env/observation/action logic to the parent and only adds
task filtering using ``benchmark/task_classification.json``.

The ``libero`` and ``libero-plus`` packages cannot coexist in the same
Python environment; run this benchmark from the dedicated
``ghcr.io/allenai/vla-evaluation-harness/libero-plus`` Docker image.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vla_eval.benchmarks.libero.benchmark import LIBEROBenchmark
from vla_eval.types import Task


def _registry_name(task: Task) -> str | None:
    """Return LIBERO's internal registry id (``task_obj.name``) for *task*.

    LIBEROBenchmark stores the human-readable ``task.language`` under
    ``task["name"]``; ``task_classification.json`` is keyed by the
    registry id, so filters and metadata joins must use this.
    """
    return getattr(task.get("task_obj"), "name", None)


class LIBEROPlusBenchmark(LIBEROBenchmark):
    """LIBERO-Plus robustness benchmark.

    Accepts every keyword argument :class:`LIBEROBenchmark` accepts (forwarded
    via ``**kwargs``) plus LIBERO-Plus-specific filters:

    Args:
        category: Optional filter on ``task_classification.json`` category
            (e.g. ``"Background Textures"``, ``"Camera Viewpoints"``). When
            set, only task variants tagged with this category are returned.
            ``libero_90`` has no classification metadata and accepts only
            ``category=None``.
        difficulty_level: Optional filter on ``difficulty_level`` (integer,
            typically 1-3). Combined with *category* via logical AND.

    Use the orchestrator's top-level ``max_tasks:`` config key to limit the
    task count after filtering.
    """

    def __init__(
        self,
        *,
        category: str | None = None,
        difficulty_level: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.category = category
        self.difficulty_level = difficulty_level
        self._classification: dict[str, dict[str, Any]] | None = None

    def _load_classification(self) -> dict[str, dict[str, Any]]:
        """Load and index task_classification.json as ``{task_name: entry}``."""
        if self._classification is not None:
            return self._classification

        # Lazy import — `libero` package is only available inside the
        # benchmark Docker image.
        from libero.libero import benchmark as libero_benchmark

        benchmark_dir = Path(libero_benchmark.__file__).parent
        classification_path = benchmark_dir / "task_classification.json"

        try:
            with open(classification_path) as f:
                raw = json.load(f)
        except FileNotFoundError:
            self._classification = {}
            return self._classification

        self._classification = {entry["name"]: entry for entry in raw.get(self.suite, []) if entry.get("name")}
        return self._classification

    def get_tasks(self) -> list[Task]:
        tasks = super().get_tasks()
        classification = self._load_classification()

        needs_classification = self.category is not None or self.difficulty_level is not None
        if needs_classification and not classification:
            raise RuntimeError(
                f"category/difficulty_level filter set but no classification metadata found "
                f"for suite {self.suite!r} (task_classification.json covers "
                f"libero_spatial/object/goal/10 only)."
            )

        filtered: list[Task] = []
        for task in tasks:
            entry = classification.get(_registry_name(task) or "")
            if needs_classification:
                if entry is None:
                    continue
                if self.category is not None and entry.get("category") != self.category:
                    continue
                if self.difficulty_level is not None and entry.get("difficulty_level") != self.difficulty_level:
                    continue
            if entry is not None:
                task["category"] = entry.get("category")
                task["difficulty_level"] = entry.get("difficulty_level")
            filtered.append(task)

        return filtered

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["category_filter"] = self.category
        meta["difficulty_filter"] = self.difficulty_level
        return meta
