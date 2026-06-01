"""Reporting/tracker integration: minimal ``report_to`` plumbing for wandb / trackio.

Mirrors the patterns in ``transformers.integrations.integration_utils``:

- Lazy import + ``RuntimeError`` on absence (no ``[project.optional-dependencies]``
  extras — the backend lib is the user's install).
- Dict dispatch by name; ``report_to`` accepts ``"all"`` / ``"none"`` / str / list.
- Backend settings live in the backends' own env vars (``WANDB_PROJECT``,
  ``TRACKIO_PROJECT``, etc.). The harness only injects what the lib cannot
  derive from env: the ``eval_id`` (for converging live + merge writers onto
  the same run) and the YAML config dict.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def is_wandb_available() -> bool:
    if importlib.util.find_spec("wandb") is None:
        return False
    # try/except so `report_to="all"` doesn't crash on a partial/broken install.
    try:
        import wandb

        return hasattr(wandb, "run")
    except Exception:
        return False


def is_trackio_available() -> bool:
    return importlib.util.find_spec("trackio") is not None


class Tracker:
    """Base tracker — all hooks are no-ops by default. Subclass and override.

    Lifecycle:

    1. ``on_eval_begin``      — once per eval, before any benchmark.
    2. ``on_benchmark_begin`` — once per benchmark in the eval's ``benchmarks`` list.
    3. ``on_episode_end``     — every episode completion (success / fail / error).
    4. ``on_benchmark_end``   — once per benchmark, after its episodes finish.
    5. ``on_eval_end``        — once per eval, after the last benchmark.
    6. ``close``              — final cleanup (flush, release file handles).
    """

    def __init__(self) -> None:
        self._step = 0

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def on_eval_begin(self, eval_id: str, config: dict[str, Any]) -> None: ...
    def on_benchmark_begin(self, bench_name: str, bench_config: dict[str, Any]) -> None: ...
    def on_episode_end(self, bench_name: str, task_name: str, ep_dict: dict[str, Any], status: str) -> None: ...
    def on_benchmark_end(self, bench_name: str, result: dict[str, Any]) -> None: ...
    def on_eval_end(self, all_results: list[dict[str, Any]]) -> None: ...
    def close(self) -> None: ...


def _episode_log_dict(bench_name: str, task_name: str, ep_dict: dict[str, Any], status: str) -> dict[str, Any]:
    """Flatten an episode result into ``{bench/task/key: value}`` for wandb-style logging."""
    prefix = f"{bench_name}/{task_name}"
    log: dict[str, Any] = {f"{prefix}/status": status}
    for k, v in (ep_dict.get("metrics") or {}).items():
        if isinstance(v, (int, float, bool)):  # bool is int subclass; float() handles both
            log[f"{prefix}/{k}"] = float(v)
    for k in ("steps", "elapsed_sec"):
        v = ep_dict.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            log[f"{prefix}/{k}"] = float(v)
    return log


def _scalar_summary(d: dict[str, Any]) -> dict[str, float]:
    """Top-level aggregate fields: numeric only, drop bools (none expected at this level)."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


class WandbTracker(Tracker):
    """Pushes eval episodes + aggregates to Weights & Biases.

    All wandb settings (``project`` default ``"vla-eval"``, ``entity``, ``name``,
    ``group``, ``tags``, ``mode``, ``dir``, ``api_key``) come from the standard
    ``WANDB_*`` env vars. The tracker only injects ``id=eval_id`` + ``resume="allow"``
    so the orchestrator (live path) and ``vla-eval merge`` (sharded summary path)
    converge on the same wandb run.
    """

    def __init__(self) -> None:
        if not is_wandb_available():
            raise RuntimeError("WandbTracker requires wandb. Run `pip install wandb`.")
        import wandb

        super().__init__()
        self._wandb = wandb

    def on_eval_begin(self, eval_id: str, config: dict[str, Any]) -> None:
        if self._wandb.run is None:
            self._wandb.init(
                id=eval_id,
                resume="allow",
                project=os.getenv("WANDB_PROJECT", "vla-eval"),
            )
        try:
            self._wandb.config.update(config, allow_val_change=True)
        except Exception:
            logger.exception("wandb.config.update failed; eval config will not appear in the run.")

    def on_episode_end(self, bench_name: str, task_name: str, ep_dict: dict[str, Any], status: str) -> None:
        if self._wandb.run is None:
            return
        self._wandb.log(_episode_log_dict(bench_name, task_name, ep_dict, status), step=self._next_step())

    def on_benchmark_end(self, bench_name: str, result: dict[str, Any]) -> None:
        if self._wandb.run is None:
            return
        for k, v in _scalar_summary(result).items():
            self._wandb.run.summary[f"{bench_name}/{k}"] = v

    def on_eval_end(self, all_results: list[dict[str, Any]]) -> None:
        if self._wandb.run is not None:
            self._wandb.finish()

    def close(self) -> None:
        if self._wandb.run is not None:
            self._wandb.finish()


class TrackioTracker(Tracker):
    """Pushes eval episodes + aggregates to Trackio.

    All trackio settings (``project`` default ``"vla-eval"``, ``space_id``,
    ``bucket_id``) come from trackio's native env vars (``TRACKIO_*``). The
    tracker injects ``name=eval_id`` + ``resume="allow"`` (trackio uses ``name``
    as the run identity, matching ``transformers.TrackioCallback``).
    """

    def __init__(self) -> None:
        if not is_trackio_available():
            raise RuntimeError("TrackioTracker requires trackio. Run `pip install trackio`.")
        import trackio

        super().__init__()
        self._trackio = trackio

    def on_eval_begin(self, eval_id: str, config: dict[str, Any]) -> None:
        self._trackio.init(
            project=os.getenv("TRACKIO_PROJECT", "vla-eval"),
            name=eval_id,
            resume="allow",
        )
        try:
            self._trackio.config.update(config, allow_val_change=True)
        except Exception:
            logger.debug("trackio.config.update not available; skipping config push.")

    def on_episode_end(self, bench_name: str, task_name: str, ep_dict: dict[str, Any], status: str) -> None:
        self._trackio.log(_episode_log_dict(bench_name, task_name, ep_dict, status), step=self._next_step())

    def on_benchmark_end(self, bench_name: str, result: dict[str, Any]) -> None:
        # trackio has no run.summary; emit aggregate as a final log under a
        # "summary/" sub-namespace so it's still discoverable.
        log = {f"{bench_name}/summary/{k}": v for k, v in _scalar_summary(result).items()}
        if log:
            self._trackio.log(log, step=self._next_step())

    def on_eval_end(self, all_results: list[dict[str, Any]]) -> None:
        try:
            self._trackio.finish()
        except Exception:
            logger.debug("trackio.finish raised; ignoring.")

    def close(self) -> None:
        try:
            self._trackio.finish()
        except Exception:
            pass


INTEGRATION_TO_TRACKER: dict[str, type[Tracker]] = {
    "wandb": WandbTracker,
    "trackio": TrackioTracker,
}

_IS_AVAILABLE = {
    "wandb": is_wandb_available,
    "trackio": is_trackio_available,
}


def get_reporting_trackers(report_to: str | list[str] | None) -> list[Tracker]:
    """Build the list of trackers from a ``report_to`` value.

    Mirror of ``transformers.get_reporting_integration_callbacks``:

    - ``None`` / ``"none"`` / empty list  →  no trackers.
    - ``"all"``                            →  every backend whose lib is installed.
    - str (other)                          →  single backend by name.
    - ``list[str]``                        →  explicit selection by name.

    Unknown backend name raises ``ValueError`` with the supported list.
    Missing optional dependency raises ``RuntimeError`` from the backend's
    ``__init__`` (clear ``pip install`` hint).
    """
    if report_to is None:
        return []
    if isinstance(report_to, str):
        if report_to == "none":
            return []
        if report_to == "all":
            report_to = [name for name, check in _IS_AVAILABLE.items() if check()]
        else:
            report_to = [report_to]
    if not report_to:
        return []
    for name in report_to:
        if name not in INTEGRATION_TO_TRACKER:
            raise ValueError(f"{name!r} is not a supported tracker; available: {sorted(INTEGRATION_TO_TRACKER)}")
    return [INTEGRATION_TO_TRACKER[name]() for name in report_to]


def call_each(trackers: list[Tracker], hook: str, *args: Any) -> None:
    """Invoke ``hook`` on every tracker; one raising backend never aborts the eval."""
    for t in trackers:
        try:
            getattr(t, hook)(*args)
        except Exception:
            logger.exception("tracker %s.%s failed", type(t).__name__, hook)
