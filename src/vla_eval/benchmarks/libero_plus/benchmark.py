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


class LIBEROPlusBenchmark(LIBEROBenchmark):
    """LIBERO-Plus robustness benchmark.

    Args:
        suite: LIBERO-Plus suite name. One of ``libero_spatial``,
            ``libero_object``, ``libero_goal``, ``libero_10``, ``libero_90``.
            Suite names are identical to vanilla LIBERO — only the task
            count differs (LIBERO-Plus registers thousands of perturbed
            variants per suite).
        category: Optional filter on ``task_classification.json`` category
            (e.g. ``"Background Textures"``, ``"Camera Views"``). When set,
            only task variants tagged with this category are returned.
            ``libero_90`` has no classification metadata and accepts only
            ``category=None``.
        difficulty_level: Optional filter on ``difficulty_level`` (integer,
            typically 1-3). Combined with *category* via logical AND.
        max_tasks: Optional hard cap on the number of task variants returned
            after filtering. Useful for smoke tests and quick reproduction
            runs — the full suite has ~2,400-2,600 variants.
        seed: Random seed for environment initialisation.
        num_steps_wait: Dummy action steps at episode start (default 10).
        send_wrist_image: Include wrist camera image in observations.
        send_state: Include proprioceptive state in observations.
        absolute_action: Use absolute (world-frame) actions instead of delta.
    """

    def __init__(
        self,
        suite: str = "libero_spatial",
        category: str | None = None,
        difficulty_level: int | None = None,
        max_tasks: int | None = None,
        seed: int = 7,
        num_steps_wait: int = 10,
        send_wrist_image: bool = False,
        send_state: bool = False,
        absolute_action: bool = False,
        max_steps: int | None = None,
        env_seed: int | None = None,
        quat_no_antipodal: bool = False,
    ) -> None:
        super().__init__(
            suite=suite,
            seed=seed,
            num_steps_wait=num_steps_wait,
            send_wrist_image=send_wrist_image,
            send_state=send_state,
            absolute_action=absolute_action,
            max_steps=max_steps,
            env_seed=env_seed,
            quat_no_antipodal=quat_no_antipodal,
        )
        self.category = category
        self.difficulty_level = difficulty_level
        self.max_tasks = max_tasks
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
        if not classification_path.exists():
            self._classification = {}
            return self._classification

        with open(classification_path) as f:
            raw = json.load(f)

        index: dict[str, dict[str, Any]] = {}
        suite_entries = raw.get(self.suite, [])
        for entry in suite_entries:
            name = entry.get("name")
            if name:
                index[name] = entry
        self._classification = index
        return self._classification

    def get_tasks(self) -> list[Task]:
        tasks = super().get_tasks()

        needs_classification = self.category is not None or self.difficulty_level is not None
        if needs_classification:
            classification = self._load_classification()
            if not classification:
                raise RuntimeError(
                    f"category/difficulty_level filter set but no classification "
                    f"metadata found for suite {self.suite!r} "
                    f"(task_classification.json covers libero_spatial/object/goal/10 only)."
                )

            def _matches(task: Task) -> bool:
                # task_classification.json is keyed by the LIBERO registry
                # name (task_obj.name), not the human-readable language that
                # LIBEROBenchmark.get_tasks() stores in task["name"].
                registry_name = getattr(task.get("task_obj"), "name", None)
                entry = classification.get(registry_name) if registry_name else None
                if entry is None:
                    return False
                if self.category is not None and entry.get("category") != self.category:
                    return False
                if self.difficulty_level is not None and entry.get("difficulty_level") != self.difficulty_level:
                    return False
                return True

            tasks = [t for t in tasks if _matches(t)]

        # Attach classification metadata to every task (useful for result
        # breakdown) when available.
        if self._classification:
            for task in tasks:
                registry_name = getattr(task.get("task_obj"), "name", None)
                entry = self._classification.get(registry_name) if registry_name else None
                if entry is not None:
                    task["category"] = entry.get("category")
                    task["difficulty_level"] = entry.get("difficulty_level")

        if self.max_tasks is not None:
            tasks = tasks[: self.max_tasks]

        return tasks

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["category_filter"] = self.category
        meta["difficulty_filter"] = self.difficulty_level
        meta["max_tasks"] = self.max_tasks
        return meta
