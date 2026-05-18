"""Per-episode video recording helper, shared across benchmarks.

Most benchmarks have a "save the agent's view of each episode as an mp4" need (failure-case debugging,
demo browsing, qualitative analysis).  This module is one home for the pattern so each benchmark doesn't
reinvent it.

## Design notes

* **Streaming write**: frames are encoded to disk as they arrive (``imageio.get_writer`` + ``append_data``).
  Memory is O(1) regardless of episode length — a 1300-step 256×256×3 episode that would have buffered
  ~250 MB now holds one frame at a time.
* **Filename can encode success/fail status**: the final filename often depends on whether the episode
  succeeded, which isn't known until the episode ends.  ``record()`` therefore writes to a hidden working
  file (``.recorder-<uid>.mp4`` in ``output_dir``); ``save()`` resolves the final filename and renames the
  working file into place.
* **Required-context fail-fast**: ``filename`` declares the template, and ``required_context`` declares
  the keys the caller will pass at ``start()``.  Missing keys raise at ``start()``, before frames are
  recorded — not as a silent ``KeyError`` -> dropped mp4 at ``save()``.  For ``str.format`` templates,
  ``required_context`` is auto-derived from the field names so callers don't have to repeat themselves.
* **Collision detection at save**: if the resolved final path already exists and the recorder wasn't
  constructed with ``overwrite=True``, ``save()`` raises ``FileExistsError`` and leaves the working file
  on disk so the caller can recover the frames.

## Filename layout

Two non-obvious things that catch users out:

1. **Zero-pad ``episode_idx``**: ``"ep{episode_idx:04d}"`` not ``"ep{episode_idx}"`` — alphabetic sort
   otherwise puts ``ep10`` before ``ep2``.
2. **Put the field you'd want to scan adjacent files by first.**  For multi-camera recording (front +
   wrist + ...), the views of the same episode are usually what you want to compare side-by-side, so
   ``"ep{episode_idx:04d}_{view}_{status}.mp4"`` keeps them adjacent.  For multi-task single-camera, the
   task is more useful first: ``"{task}_ep{episode_idx:04d}_{status}.mp4"``.

## Caller pattern

    recorder = EpisodeVideoRecorder(
        output_dir="/workspace/results/videos",
        filename="{env_id}_ep{episode_idx:04d}_{status}.mp4",
        # required_context is auto-derived for str templates → ("env_id", "episode_idx")
        fps=20,
    )

    # In benchmark.reset(task):
    recorder.start({"env_id": task["env_id"], "episode_idx": task["episode_idx"]})
    recorder.record(initial_frame)

    # In benchmark.step(action):
    recorder.record(frame)

    # In benchmark.get_step_result(step_result):
    recorder.save(status="success" if success else "fail")

    # In benchmark.cleanup():
    recorder.discard()  # drops any in-flight working file

Each recorder records a single stream.  To capture multiple views (e.g. front + wrist), construct one
recorder per view; they don't share state.
"""

from __future__ import annotations

import logging
import os
import string
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# A `str.format`-style template, or a callable taking the resolved context (caller's start() context +
# injected ``status``) and returning a filename relative to ``output_dir``.
FilenameSpec = str | Callable[[Mapping[str, Any]], str]


class EpisodeVideoRecorder:
    """Streaming per-episode video recorder.

    Lifecycle: ``start()`` → ``record()`` × N → ``save()`` (or ``discard()``).  ``start()`` may be called
    again to begin a new episode; if a previous episode never reached ``save()``/``discard()``, its
    working file is dropped first.

    Inactive (no episode in progress) is a valid state: ``record()`` / ``save()`` / ``discard()`` are
    no-ops then, so callers don't need defensive ``if recorder.active`` checks.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        filename: FilenameSpec,
        required_context: Sequence[str] | None = None,
        fps: int = 20,
        overwrite: bool = False,
    ) -> None:
        """
        Args:
            output_dir: Directory the final mp4 lands in.  Created on first ``start()`` if missing.
                Filename templates may include subdirectories (``"{suite}/{task}_..."``); intermediate
                dirs are created at ``save()`` time.
            filename: ``str.format`` template or callable producing the filename relative to
                ``output_dir``.  Resolved at ``save()`` time over ``{**start_context, "status": status}``.
                Required because every benchmark identifies tasks differently (``env_id``, ``task_id``,
                ``suite/task``); there is no universally safe default.
            required_context: Keys that must be present in the dict passed to ``start()``.  ``ValueError``
                is raised at ``start()`` if any are missing.  When ``None`` (the default) and ``filename``
                is a ``str.format`` template, this is auto-derived from the template's field names
                (``status`` excluded, since ``save()`` injects it) — callers don't have to repeat
                themselves.  When the ``filename`` is a callable, this must be specified explicitly:
                there's no way to introspect a callable's key dependencies.  An explicit value is allowed
                to be a subset of the template's keys (i.e. some keys can be optional — they'll fail at
                ``save()`` time if ultimately missing).
            fps: Output framerate.
            overwrite: When False (default), ``save()`` raises ``FileExistsError`` if the resolved final
                path is already taken.  When True, an existing file is replaced.
        """
        self.output_dir = Path(output_dir)
        self._filename_spec = filename
        self.fps = fps
        if required_context is None:
            if not isinstance(filename, str):
                raise ValueError(
                    "required_context must be specified when filename is a callable; "
                    "the recorder can't introspect callables to discover key dependencies."
                )
            required_context = _fields_from_template(filename)
        self._required_context = tuple(required_context)
        self._overwrite = overwrite

        # One working file per recorder instance, reused across episodes.  Hidden (`.recorder-`) so it
        # doesn't show up in casual listings, uuid-suffixed so concurrent recorders sharing output_dir
        # don't collide.
        self._working_path = self.output_dir / f".recorder-{uuid.uuid4().hex[:12]}.mp4"

        # Lifecycle state — None whenever no episode is in progress.
        self._writer: Any = None
        self._context: dict[str, Any] | None = None
        self._frames_written = 0
        # Latched on the first record() failure so the writer being wedged (corrupt subprocess pipe, etc.)
        # doesn't produce one warning per step.
        self._record_failed = False

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, context: Mapping[str, Any]) -> None:
        """Begin a new episode.

        Validates required context keys, opens a streaming writer to the working file.  If a previous
        episode is still in flight (no ``save()`` / ``discard()``) it is dropped first.  On writer-open
        failure the recorder stays inactive and subsequent ``record()`` / ``save()`` are no-ops; the
        failure is logged.
        """
        missing = [k for k in self._required_context if k not in context]
        if missing:
            raise ValueError(f"EpisodeVideoRecorder.start: missing required context keys: {missing}")

        if self.active:
            self.discard()

        self._context = dict(context)
        self._frames_written = 0
        self._record_failed = False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            import imageio

            self._writer = imageio.get_writer(str(self._working_path), fps=self.fps)
        except Exception as e:
            logger.warning("Failed to open video writer for context=%r: %s", self._context, e)
            self._writer = None
            self._context = None
            _safe_unlink(self._working_path)

    def record(self, frame: np.ndarray) -> None:
        """Append a frame to the in-flight episode.

        On the default ``.mp4`` / ffmpeg path, imageio serializes the frame via ``ndarray.tobytes()``
        before piping it to the encoder subprocess — that's a synchronous copy, so the caller can mutate
        the underlying buffer once this returns.  If you configure a non-ffmpeg writer that retains
        references (e.g. the pillow plugin appends ``Image.fromarray(arr)`` to a list flushed at close),
        pass copies yourself.

        No-op if no episode is in progress.  The first encode failure latches the recorder so subsequent
        ``record()`` calls become no-ops rather than flooding the log.
        """
        if not self.active or self._record_failed:
            return
        try:
            self._writer.append_data(frame)
            self._frames_written += 1
        except Exception as e:
            logger.warning(
                "record() failed for context=%r at frame %d: %s; remaining frames will be dropped",
                self._context,
                self._frames_written,
                e,
            )
            self._record_failed = True

    def save(self, status: str = "success") -> Path | None:
        """Finalize the in-flight episode.

        Closes the writer, resolves the final filename from ``{**context, "status": status}``, and moves
        the working file into place.  Returns the final ``Path``, or ``None`` if the recorder was inactive
        or filename resolution / writer close failed.

        Raises:
            FileExistsError: if the resolved final path already exists and the recorder was constructed
                with ``overwrite=False`` (the default).  The working file is left on disk so the caller
                can recover the frames manually.
        """
        if not self.active:
            return None

        writer, context = self._writer, self._context
        frames_written = self._frames_written
        # Reset state up front: any return path below leaves the recorder inactive.
        self._writer = None
        self._context = None
        self._frames_written = 0
        self._record_failed = False

        try:
            writer.close()
        except Exception as e:
            logger.warning("Failed to close video writer for context=%r: %s", context, e)
            _safe_unlink(self._working_path)
            return None

        try:
            relative_name = self._resolve_filename({**(context or {}), "status": status})
        except Exception as e:
            logger.warning("Failed to resolve filename for context=%r status=%r: %s", context, status, e)
            _safe_unlink(self._working_path)
            return None

        final_path = self.output_dir / relative_name
        if final_path.exists() and not self._overwrite:
            raise FileExistsError(
                f"{final_path} already exists. Recorded frames are at {self._working_path}. "
                f"Pass overwrite=True to replace, or rename the working file manually."
            )

        try:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(self._working_path), str(final_path))
        except Exception as e:
            logger.warning("Failed to move working file to %s for context=%r: %s", final_path, context, e)
            _safe_unlink(self._working_path)
            return None

        logger.info("Saved episode video: %s (%d frames)", final_path, frames_written)
        return final_path

    def discard(self) -> None:
        """Abandon the in-flight episode without producing an mp4.

        Closes the writer (best-effort) and removes the working file.  Safe to call when no episode is in
        progress (no-op).
        """
        writer = self._writer
        self._writer = None
        self._context = None
        self._frames_written = 0
        self._record_failed = False
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        _safe_unlink(self._working_path)

    def _resolve_filename(self, context: Mapping[str, Any]) -> str:
        spec = self._filename_spec
        if isinstance(spec, str):
            return spec.format(**context)
        return spec(context)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _fields_from_template(template: str) -> tuple[str, ...]:
    """Extract top-level field names from a ``str.format`` template.

    ``status`` is excluded — ``save()`` always injects it and it would just cause a spurious "missing
    required context key" failure if treated as required.  Format specs (``:04d``) and attribute /
    indexing access are stripped: ``"{episode_idx:04d}"`` → ``"episode_idx"``, ``"{task.name}"`` →
    ``"task"``.  Order is preserved (first occurrence wins) so error messages are deterministic.
    """
    seen: list[str] = []
    for _, field_name, _, _ in string.Formatter().parse(template):
        if not field_name:
            continue
        # `field_name` can be "name", "name.attr", "name[0]", or just "0" for positional.
        bare = field_name.split(".", 1)[0].split("[", 1)[0]
        if bare and bare != "status" and bare not in seen:
            seen.append(bare)
    return tuple(seen)
