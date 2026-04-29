"""Per-episode video recording helper, shared across benchmarks.

Most benchmarks have a "save the agent's view of each episode as an mp4"
need (debugging failures, demo browsing, public showcase, qualitative
analysis).  This module is a single home for that pattern so each
benchmark doesn't reinvent it.

Typical use from a benchmark:

    from vla_eval.benchmarks.recording import EpisodeVideoRecorder
    from vla_eval.types import Task

    class MyBenchmark(StepBenchmark):
        def __init__(self, ..., save_episode_video: bool = False,
                     video_dir: str | None = None) -> None:
            ...
            self._recorder = EpisodeVideoRecorder(
                output_dir=video_dir or "/workspace/results/videos",
                fps=20,
            ) if save_episode_video else None

        def reset(self, task: Task) -> Any:
            ...
            if self._recorder is not None:
                self._recorder.start(task)
                # Capture the initial frame so the video covers the whole episode.
                self._recorder.record(initial_frame)

        def step(self, action) -> StepResult:
            ...
            if self._recorder is not None:
                # If the underlying buffer is reused (ManiSkill does), copy first.
                self._recorder.record(np.array(frame, copy=True))

        def get_step_result(self, step_result) -> EpisodeResult:
            success = ...
            if self._recorder is not None:
                self._recorder.save(success=success)
            return {"success": success}

        def cleanup(self) -> None:
            ...
            if self._recorder is not None:
                self._recorder.discard()  # drop any in-flight frames

The recorder is best-effort: encode failures log a warning and clear the
buffer rather than aborting the episode.  Frame buffering is in-memory
(``imageio.mimsave``); a typical 1300-step episode at 256×256×3 uint8 is
~250 MB.  If you need lower memory or non-blocking saves, switch to
``imageio.get_writer`` streaming and async ``cleanup()`` join — both are
straightforward upgrades that don't change the public API here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vla_eval.types import Task

logger = logging.getLogger(__name__)


class EpisodeVideoRecorder:
    """Buffers per-step frames and writes one mp4 per episode.

    Filename pattern: ``{task_name}_ep{episode_idx}_{success|fail}.mp4``.
    ``task["episode_idx"]`` is required at ``start()`` time — without it
    multi-episode runs of the same task collide on a single ``_ep0_`` filename
    and silently overwrite.
    """

    def __init__(self, output_dir: str, fps: int = 20) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._task: Task | None = None

    def start(self, task: Task) -> None:
        """Begin recording for ``task``. Drops any frames left from a prior
        episode (e.g. orchestrator skipped ``get_step_result`` after a
        crash) so we don't accumulate."""
        if "episode_idx" not in task:
            raise ValueError(
                "EpisodeVideoRecorder.start(task) requires task['episode_idx'] "
                "(otherwise per-episode mp4s collide on the same filename)"
            )
        self._frames = []
        self._task = task

    def record(self, frame: np.ndarray) -> None:
        """Append a frame.

        The caller is responsible for copying if the underlying buffer is
        reused across steps (ManiSkill / SAPIEN reuse one image buffer per
        camera; without ``np.array(frame, copy=True)`` every recorded
        frame ends up identical to the last one).
        """
        if self._task is None:
            # start() not called — silently ignore so a benchmark that
            # toggles recording mid-run doesn't crash.
            return
        self._frames.append(frame)

    def save(self, success: bool) -> str | None:
        """Encode the buffered frames to mp4 and reset.

        Returns the file path on success, ``None`` if there was nothing to
        save or the encode failed.  Encode failures log a warning rather
        than raise — a corrupted video should never fail an otherwise good
        eval episode.
        """
        if self._task is None or not self._frames:
            self._frames = []
            self._task = None
            return None

        try:
            import imageio

            os.makedirs(self.output_dir, exist_ok=True)
            task_name = self._task.get("name", self._task.get("env_id", "unknown"))
            status = "success" if success else "fail"
            fname = f"{task_name}_ep{self._task['episode_idx']}_{status}.mp4"
            path = os.path.join(self.output_dir, fname)
            imageio.mimsave(path, self._frames, fps=self.fps)
            logger.info("Saved episode video: %s (%d frames)", path, len(self._frames))
            return path
        except Exception as e:
            logger.warning("Failed to save episode video: %s", e)
            return None
        finally:
            self._frames = []
            self._task = None

    def discard(self) -> None:
        """Drop the buffer without saving. Call from ``cleanup()`` paths to
        avoid leaking frames if the orchestrator never calls ``save()``."""
        self._frames = []
        self._task = None
