"""Composite per-episode recorder — video + structured JSONL in one directory.

Single ``output_dir`` receives both ``.mp4`` and ``.jsonl`` with matching
filenames (e.g. ``BinFill_ep0000_fail.mp4`` + ``BinFill_ep0000_fail.jsonl``).

Typical usage::

    recorder = EpisodeRecorder("/workspace/results/episodes")
    recorder.start({"env_id": "BinFill", "episode_idx": 0})
    recorder.record_frame(front_rgb)
    recorder.record_step({"step": n, "gt_subgoal": "pick up cube", ...})
    recorder.save(status="fail")
"""

from __future__ import annotations

import json
import logging
import os
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vla_eval.benchmarks.recording import EpisodeVideoRecorder

logger = logging.getLogger(__name__)


@dataclass
class RecordingConfig:
    """Configuration for per-episode recording, passed via benchmark YAML ``params.recording``."""

    output_dir: str = "/workspace/results/episodes"
    record_video: bool = True
    record_step: bool = True
    step_fields: list[str] = field(default_factory=list)


__all__ = ["EpisodeRecorder", "RecordingConfig"]


class EpisodeRecorder:
    """Composite recorder: optional video (mp4) + optional data (JSONL)."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        record_video: bool = True,
        record_step: bool = True,
        filename_stem: str = "{env_id}_ep{episode_idx:04d}_{status}",
        fps: int = 20,
    ) -> None:
        out = Path(output_dir)
        if record_video or record_step:
            out.mkdir(parents=True, exist_ok=True)
        self._video: EpisodeVideoRecorder | None = (
            EpisodeVideoRecorder(output_dir=out, filename=filename_stem + ".mp4", fps=fps) if record_video else None
        )
        self._data_dir = out if record_step else None
        self._data_filename = filename_stem + ".jsonl"
        self._data_fh: Any | None = None
        self._data_working: Path | None = None
        self._context: dict[str, Any] = {}
        self._required_context = tuple(
            bare
            for _, field_name, _, _ in string.Formatter().parse(filename_stem)
            if field_name
            for bare in [field_name.split(".")[0].split("[")[0]]
            if bare and bare != "status"
        )

    @property
    def active(self) -> bool:
        return (self._video is not None and self._video.active) or self._data_fh is not None

    def start(self, context: Mapping[str, Any]) -> None:
        missing = [k for k in self._required_context if k not in context]
        if missing:
            raise ValueError(f"EpisodeRecorder.start: missing required context keys: {missing}")
        self._context = dict(context)
        if self._video is not None:
            self._video.start(context)
        if self._data_dir is not None:
            if self._data_fh is not None:
                self._discard_data()
            uid = uuid.uuid4().hex[:12]
            self._data_working = self._data_dir / f".data-{uid}.jsonl"
            self._data_fh = open(self._data_working, "w", encoding="utf-8")  # noqa: SIM115

    def record_frame(self, frame: np.ndarray) -> None:
        if self._video is not None:
            self._video.record(frame)

    def record_step(self, data: dict[str, Any]) -> None:
        if self._data_fh is None:
            return
        self._data_fh.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    def save(self, **extra: Any) -> None:
        status_kwargs = {**self._context, **extra}
        if self._video is not None:
            self._video.save(**extra)
        if self._data_fh is not None:
            self._data_fh.close()
            self._data_fh = None
            try:
                final_name = self._data_filename.format(**status_kwargs)
                if self._data_dir is None or self._data_working is None:
                    return
                final_path = self._data_dir / final_name
                if final_path.exists():
                    final_path.unlink()
                self._data_working.rename(final_path)
                logger.info("Saved episode data: %s", final_path)
            except Exception:
                logger.warning("Failed to save episode data", exc_info=True)
            self._data_working = None

    def discard(self) -> None:
        if self._video is not None:
            self._video.discard()
        self._discard_data()

    def _discard_data(self) -> None:
        if self._data_fh is not None:
            self._data_fh.close()
            self._data_fh = None
        if self._data_working is not None and self._data_working.exists():
            self._data_working.unlink()
        self._data_working = None
