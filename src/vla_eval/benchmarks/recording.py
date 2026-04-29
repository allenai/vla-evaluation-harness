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
  ``filename`` and ``required_context`` are required at construction time
  — there is no universal default because every benchmark names tasks
  differently (``env_id``, ``task_id``, ``suite/task`` …).  Forcing the
  caller to spell it out catches mismatched context keys at ``start()``
  rather than as a silent dropped mp4 at ``save()``.
* **Best-effort**: every encode-side failure logs a warning with the
  context for debuggability and clears state — a corrupted video should
  never bring down an otherwise good eval episode.

## Caller pattern

    from vla_eval.benchmarks.recording import EpisodeVideoRecorder

    recorder = EpisodeVideoRecorder(
        output_dir="/workspace/results/videos",
        filename="{env_id}_ep{episode_idx}_{status}.mp4",
        required_context=("env_id", "episode_idx"),
        fps=20,
    )

    # In benchmark.reset(task):
    recorder.start({"env_id": task["env_id"], "episode_idx": task["episode_idx"]})
    recorder.record(initial_frame)

    # In benchmark.step(action):
    recorder.record(frame)  # default ffmpeg path copies via .tobytes() —
                            # see record() docstring for backend caveats

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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Either a `str.format`-style template or a callable that takes the resolved
# context (caller's start() context + injected ``status``) and returns a
# filename relative to ``output_dir``.
FilenameSpec = str | Callable[[Mapping[str, Any]], str]


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
        filename: FilenameSpec,
        required_context: Sequence[str],
        fps: int = 20,
        writer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            output_dir: Directory the final mp4 lands in.  Created if missing.
                Filename templates may include subdirectories (e.g.
                ``"{suite}/{task}_..."``); intermediate dirs are also created.
            filename: ``str.format`` template or callable producing the
                filename relative to ``output_dir``.  Resolved at ``save()``
                time over ``{**start_context, "status": status}``.  Required
                because every benchmark identifies tasks differently
                (``env_id``, ``task_id``, ``suite/task``) — there is no
                universally safe default.
            required_context: Keys that must be present in the dict passed to
                ``start()``.  ``ValueError`` is raised at ``start()`` if any
                are missing.  Required so callers explicitly declare the
                template's expectations; failing fast at ``start()`` avoids
                a silent ``KeyError`` -> dropped mp4 at ``save()`` time.
                Should include every key the ``filename`` template references.
            fps: Output framerate.
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
        # Latched on the first record() failure so we don't log-spam every
        # subsequent step with the same warning when the underlying writer
        # is wedged (corrupt subprocess pipe, etc.).
        self._record_failed = False

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
        self._record_failed = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
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
            logger.warning("Failed to open video writer for context=%r: %s", self._context, e)
            _safe_unlink(self._tempfile)
            self._tempfile = None
            self._writer = None
            self._context = None

    def record(self, frame: np.ndarray) -> None:
        """Append a frame to the in-flight episode.

        On the default ``.mp4`` / ffmpeg path, imageio serializes the
        frame via ``np.ndarray.tobytes()`` before piping it to the
        encoder subprocess — that's a synchronous copy, so the caller
        can mutate the underlying buffer once this returns.  If you
        configure a non-ffmpeg writer that retains references (e.g.
        the pillow plugin appends ``Image.fromarray(arr)`` to a list
        flushed at close), you must pass copies yourself.

        No-op if no episode is in progress.  The first encode failure
        latches the recorder so subsequent ``record()`` calls become
        no-ops rather than flooding the log.
        """
        if not self.active or self._record_failed:
            return
        try:
            self._writer.append_data(frame)
            self._frames_written += 1
        except Exception as e:
            logger.warning(
                "record() failed for context=%r at frame %d: %s; remaining frames in this episode will be dropped",
                self._context,
                self._frames_written,
                e,
            )
            self._record_failed = True

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

        writer, tempfile_path, context = self._writer, self._tempfile, self._context
        frames_written = self._frames_written
        try:
            try:
                writer.close()
            except Exception as e:
                logger.warning("Failed to close video writer for context=%r: %s", context, e)
                _safe_unlink(tempfile_path)
                return None

            try:
                relative_name = self._resolve_filename({**(context or {}), "status": status})
            except Exception as e:
                logger.warning("Failed to resolve filename for context=%r status=%r: %s", context, status, e)
                _safe_unlink(tempfile_path)
                return None

            final_path = self.output_dir / relative_name
            try:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(str(tempfile_path), str(final_path))
            except Exception as e:
                logger.warning("Failed to finalize video at %s for context=%r: %s", final_path, context, e)
                _safe_unlink(tempfile_path)
                return None

            logger.info("Saved episode video: %s (%d frames)", final_path, frames_written)
            return final_path
        finally:
            self._writer = None
            self._tempfile = None
            self._context = None
            self._frames_written = 0
            self._record_failed = False

    def discard(self) -> None:
        """Abandon the in-flight episode without producing an mp4.

        Closes the writer (best-effort) and removes the tempfile.  Safe
        to call when no episode is in progress (no-op).
        """
        writer, tempfile_path = self._writer, self._tempfile
        self._writer = None
        self._tempfile = None
        self._context = None
        self._frames_written = 0
        self._record_failed = False
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _safe_unlink(tempfile_path)

    def _resolve_filename(self, context: Mapping[str, Any]) -> str:
        spec = self._filename_spec
        if isinstance(spec, str):
            return spec.format(**context)
        return spec(context)


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass
