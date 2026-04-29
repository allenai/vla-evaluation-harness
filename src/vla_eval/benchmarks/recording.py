"""Per-episode video recording helper, shared across benchmarks.

Most benchmarks have a "save the agent's view of each episode as an mp4"
need (failure-case debugging, demo browsing, public showcase, qualitative
analysis).  This module is a single home for that pattern so each
benchmark doesn't reinvent it.

## Design

* **Streaming**: frames are encoded to disk as they arrive
  (``imageio.get_writer`` + ``append_data``) rather than buffered in RAM.
  Memory is O(1) regardless of episode length; a 1300-step 256×256×3
  episode that previously held ~250 MB now holds one frame at a time.
  A side benefit is that a partially-written mp4 is left on disk if the
  process is killed mid-episode — playable up to the last completed
  frame, useful for debugging crashes.
* **Atomic finalize**: frames stream into a tempfile in the same output
  directory; ``save()`` resolves the final filename (which usually
  depends on success/failure status) and ``os.replace``-s the tempfile
  into place.  Concurrent jobs sharing a directory don't collide.
* **Logging-style filename templating**: the filename is a ``str.format``
  template (or callable) over a context dict that the caller passes at
  ``start()`` time, plus a ``status`` key injected at ``save()`` time.
  Default template ``"{task_name}_ep{episode_idx}_{status}.mp4"``;
  benchmarks with richer identifiers can use any context they like, e.g.
  ``"{suite}/{task}/{episode_idx:04d}_seed{seed}_{status}.mp4"``.
* **Best-effort**: every encode-side failure logs a warning with the
  context for debuggability and clears state — a corrupted video should
  never bring down an otherwise good eval episode.

## Caller pattern

    from vla_eval.benchmarks.recording import EpisodeVideoRecorder

    recorder = EpisodeVideoRecorder(
        output_dir="/workspace/results/videos",
        # filename can stay default, or e.g.
        # filename="{suite}/{task}_ep{episode_idx}_{status}.mp4",
        fps=20,
    )

    # In benchmark.reset(task):
    recorder.start({"task_name": task["env_id"], "episode_idx": task["episode_idx"]})
    recorder.record(initial_frame)

    # In benchmark.step(action):
    recorder.record(frame)  # caller may mutate `frame` after this returns
                            # (imageio's writer copies into its own buffer)

    # In benchmark.get_step_result(step_result):
    recorder.save(status="success" if success else "fail")

    # In benchmark.cleanup():
    recorder.discard()  # drops any in-flight writer + tempfile

## Multi-camera

The recorder is single-stream by design.  Benchmarks with multiple views
(e.g. front + wrist) instantiate one recorder per view with different
``filename`` templates (e.g. ``"{view}_ep{episode_idx}_{status}.mp4"``,
substituting the view name into the context at ``start()``).  This keeps
each recorder simple and lets callers mix-and-match resolutions, fps,
and codecs per view.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

# Either a `str.format`-style template or a callable that takes the resolved
# context (caller's start() context + injected ``status``) and returns a
# filename relative to ``output_dir``.
FilenameSpec = Union[str, Callable[[Mapping[str, Any]], str]]


class EpisodeVideoRecorder:
    """Streaming per-episode video recorder.

    Lifecycle: ``start()`` → ``record()`` × N → ``save()`` (or
    ``discard()``).  ``start()`` may be called again to begin a new
    episode; if a previous episode never reached ``save()``/``discard()``
    (e.g. orchestrator crash) the in-flight writer is closed and its
    tempfile cleaned up first.

    Inactive (no episode in progress) is a valid state: ``record()`` /
    ``save()`` / ``discard()`` are no-ops in that case so callers don't
    need defensive ``if recorder.active`` checks.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        filename: FilenameSpec = "{task_name}_ep{episode_idx}_{status}.mp4",
        fps: int = 20,
        required_context: Sequence[str] = ("episode_idx",),
        writer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            output_dir: Directory the final mp4 lands in.  Created if missing.
                Filename templates may include subdirectories (e.g.
                ``"{suite}/{task}_..."``); intermediate dirs are also created.
            filename: ``str.format`` template or callable producing the
                filename relative to ``output_dir``.  Resolved at ``save()``
                time over ``{**start_context, "status": status}``.
            fps: Output framerate.
            required_context: Keys that must be present in the dict passed to
                ``start()``.  ``ValueError`` is raised at ``start()`` if any
                are missing.  Default ``("episode_idx",)`` because without an
                episode index, multi-episode runs of the same task collide
                on a single ``_ep0_`` filename.
            writer_kwargs: Extra kwargs forwarded to ``imageio.get_writer``
                (e.g. ``{"codec": "libx264", "quality": 8}``).
        """
        self.output_dir = Path(output_dir)
        self._filename_spec = filename
        self.fps = fps
        self._required_context = tuple(required_context)
        self._writer_kwargs = dict(writer_kwargs or {})

        # Lifecycle state — None whenever no episode is in progress.
        self._writer: Any = None
        self._tempfile: Path | None = None
        self._context: dict[str, Any] | None = None
        self._frames_written = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, context: Mapping[str, Any]) -> None:
        """Begin a new episode.

        Validates required context keys, opens a streaming writer to a
        tempfile in ``output_dir``.  If a previous episode is still in
        flight (no ``save``/``discard`` called) it is discarded first.
        On writer-open failure the recorder stays inactive and subsequent
        ``record()``/``save()`` are no-ops; the failure is logged.
        """
        missing = [k for k in self._required_context if k not in context]
        if missing:
            raise ValueError(f"EpisodeVideoRecorder.start: missing required context keys: {missing}")

        if self.active:
            self.discard()

        self._context = dict(context)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Tempfile in the same directory so os.replace() at save time is
            # atomic on the same filesystem.
            # Tempfile keeps a real `.mp4` suffix so imageio's format
            # auto-detection works; the `.recorder-` prefix marks it as ours.
            fd, temp_path = tempfile.mkstemp(
                prefix=".recorder-",
                suffix=".mp4",
                dir=str(self.output_dir),
            )
            os.close(fd)
            self._tempfile = Path(temp_path)

            import imageio

            self._writer = imageio.get_writer(str(self._tempfile), fps=self.fps, **self._writer_kwargs)
        except Exception as e:
            logger.warning(
                "Failed to open video writer for context=%r: %s",
                self._context,
                e,
            )
            self._cleanup_tempfile()
            self._writer = None
            self._context = None

    def record(self, frame: np.ndarray) -> None:
        """Append a frame to the in-flight episode.

        ``imageio``'s writers copy the frame data synchronously, so the
        caller is free to mutate the underlying buffer once this returns
        — no defensive ``np.array(frame, copy=True)`` required.

        No-op if no episode is in progress.  Per-frame encode failures
        log a warning but don't disable the writer (could be transient).
        """
        if not self.active:
            return
        try:
            self._writer.append_data(frame)
            self._frames_written += 1
        except Exception as e:
            logger.warning(
                "Failed to write frame to video for context=%r: %s",
                self._context,
                e,
            )

    def save(self, status: str = "success") -> Path | None:
        """Finalize the in-flight episode.

        Closes the streaming writer, resolves the final filename from
        ``{**context, "status": status}``, and atomically moves the
        tempfile into place.  Returns the final ``Path`` on success,
        ``None`` if the recorder was inactive or any finalize step
        failed.  After this call the recorder is inactive again.
        """
        if not self.active:
            return None

        # Local copy + clear state up front so failure paths can't leak it.
        writer, tempfile_path, context = self._writer, self._tempfile, self._context
        frames_written = self._frames_written
        self._writer = None
        self._tempfile = None
        self._context = None
        self._frames_written = 0

        try:
            writer.close()
        except Exception as e:
            logger.warning(
                "Failed to close video writer for context=%r: %s",
                context,
                e,
            )
            _safe_unlink(tempfile_path)
            return None

        try:
            full_context = {**(context or {}), "status": status}
            relative_name = self._resolve_filename(full_context)
        except Exception as e:
            logger.warning(
                "Failed to resolve filename for context=%r status=%r: %s",
                context,
                status,
                e,
            )
            _safe_unlink(tempfile_path)
            return None

        final_path = self.output_dir / relative_name
        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(tempfile_path), str(final_path))
        except Exception as e:
            logger.warning(
                "Failed to finalize video at %s for context=%r: %s",
                final_path,
                context,
                e,
            )
            _safe_unlink(tempfile_path)
            return None

        logger.info("Saved episode video: %s (%d frames)", final_path, frames_written)
        return final_path

    def discard(self) -> None:
        """Abandon the in-flight episode without producing an mp4.

        Closes the writer (best-effort) and removes the tempfile.  Safe
        to call when no episode is in progress (no-op).
        """
        writer = self._writer
        tempfile_path = self._tempfile
        self._writer = None
        self._tempfile = None
        self._context = None
        self._frames_written = 0
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _safe_unlink(tempfile_path)

    def _resolve_filename(self, context: Mapping[str, Any]) -> str:
        if callable(self._filename_spec):
            return self._filename_spec(context)
        return self._filename_spec.format(**context)

    def _cleanup_tempfile(self) -> None:
        _safe_unlink(self._tempfile)
        self._tempfile = None


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
