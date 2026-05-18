"""SQLite-backed per-episode step rows, episode results, and eval metadata."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


# ``json_patch`` was added in SQLite 3.38.0 (Feb 2022). Check once at import
# rather than at first write so old libsqlite3 fails fast with a useful message
# (Ubuntu 22.04 ships 3.37.2; uv-bundled Python ships a much newer sqlite).
_MIN_SQLITE = (3, 38, 0)
if sqlite3.sqlite_version_info < _MIN_SQLITE:
    raise RuntimeError(
        f"vla_eval.recording requires SQLite >= {'.'.join(map(str, _MIN_SQLITE))} "
        f"(json_patch); detected {sqlite3.sqlite_version}. "
        "Upgrade libsqlite3 or run via uv-bundled Python."
    )


EpisodeStatus = Literal["success", "fail", "error"]


def _json_default(obj: Any) -> Any:
    """JSON fallback that turns numpy arrays/scalars into native Python via ``.tolist()``."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=30000;

CREATE TABLE IF NOT EXISTS eval_metadata (
    eval_id    TEXT PRIMARY KEY,
    safe_name  TEXT NOT NULL,
    metadata   TEXT NOT NULL  -- JSON: benchmark, mode, config, harness_version, server_info, metric_keys
);

CREATE TABLE IF NOT EXISTS episode_results (
    sid             TEXT NOT NULL,
    eid             TEXT NOT NULL,
    eval_id         TEXT NOT NULL,
    task_name       TEXT,
    episode_id      INTEGER,
    status          TEXT,            -- 'success' | 'fail' | 'error'
    metrics         TEXT,            -- JSON
    steps           INTEGER,
    elapsed_sec     REAL,
    context         TEXT,            -- JSON
    jsonl_path      TEXT,            -- resolved final filename for ``vla-eval merge``
    failure_reason  TEXT,
    failure_detail  TEXT,
    PRIMARY KEY (sid, eid)
);
CREATE INDEX IF NOT EXISTS idx_episode_results_eval ON episode_results(eval_id);

CREATE TABLE IF NOT EXISTS step_rows (
    sid      TEXT NOT NULL,
    eid      TEXT NOT NULL,
    step_id  INTEGER NOT NULL,
    fields   TEXT NOT NULL,  -- JSON document; multi-writer field-union via json_patch
    PRIMARY KEY (sid, eid, step_id)
);
"""


# ---------------------------------------------------------------------------
# Recording config helpers
# ---------------------------------------------------------------------------


# Default when ``recording.filename_stem`` is omitted — uses only keys the
# orchestrator always injects, so it renders for any benchmark.
DEFAULT_FILENAME_STEM = "ep{episode_idx:04d}_{status}"


def serializable_task_kwargs(task: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly subset of *task* — safe for str.format and SQLite JSON columns."""
    return {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool))}


# ---------------------------------------------------------------------------
# RecordingStore — SQLite connection + idempotent writes
# ---------------------------------------------------------------------------


def db_path_for_eval(output_dir: str | Path, eval_id: str) -> Path:
    """Canonical SQLite path for an eval. All shards on one host point here."""
    return Path(output_dir) / f"recording-{eval_id}.sqlite"


class RecordingStore:
    """SQLite connection holder. One per process; same-file concurrency via WAL."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=30.0)
        self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def upsert_eval_metadata(self, eval_id: str, safe_name: str, metadata: dict[str, Any]) -> None:
        """First-writer-wins. Repeat calls with the same eval_id are no-ops."""
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO eval_metadata (eval_id, safe_name, metadata) VALUES (?, ?, ?)",
                (eval_id, safe_name, json.dumps(metadata, default=_json_default)),
            )

    def upsert_episode_result(
        self,
        *,
        sid: str,
        eid: str,
        eval_id: str,
        task_name: str,
        episode_id: int,
        status: str,
        metrics: dict[str, Any],
        steps: int,
        elapsed_sec: float,
        context: dict[str, Any],
        jsonl_path: str,
        failure_reason: str | None,
        failure_detail: str | None,
    ) -> None:
        """Insert-or-replace; safe under orchestrator retry with the same (sid, eid)."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO episode_results
                  (sid, eid, eval_id, task_name, episode_id, status, metrics,
                   steps, elapsed_sec, context, jsonl_path,
                   failure_reason, failure_detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    eid,
                    eval_id,
                    task_name,
                    episode_id,
                    status,
                    json.dumps(metrics, default=_json_default),
                    steps,
                    elapsed_sec,
                    json.dumps(context, default=_json_default),
                    jsonl_path,
                    failure_reason,
                    failure_detail,
                ),
            )

    def upsert_step_rows(self, sid: str, eid: str, rows: dict[int, dict[str, Any]]) -> None:
        """Multi-writer field-union UPSERT via ``json_patch`` (per-key last-writer-wins)."""
        if not rows:
            return
        payload = [(sid, eid, step_id, json.dumps(fields, default=_json_default)) for step_id, fields in rows.items()]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO step_rows (sid, eid, step_id, fields)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sid, eid, step_id)
                  DO UPDATE SET fields = json_patch(fields, excluded.fields)
                """,
                payload,
            )


# ---------------------------------------------------------------------------
# EpisodeRecorder — orchestrator side (owns video + episode lifecycle)
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Per-episode recorder. Benchmark records frames/steps; orchestrator calls close()."""

    def __init__(
        self,
        *,
        store: RecordingStore,
        sid: str,
        eid: str,
        eval_id: str,
        output_dir: str | Path,
        filename_stem: str,
        context: dict[str, Any],
        record_video: bool = True,
        record_step: bool = True,
        video_fps: int = 20,
    ) -> None:
        self._store = store
        self._sid = sid
        self._eid = eid
        self._eval_id = eval_id
        self._output_dir = Path(output_dir)
        self._filename_stem = filename_stem
        self._context = dict(context)
        self._record_step = record_step
        self._steps: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._closed = False
        self._video: Any = None
        if record_video:
            from vla_eval.benchmarks.video import EpisodeVideoRecorder

            self._video = EpisodeVideoRecorder(
                output_dir=self._output_dir,
                filename=filename_stem + ".mp4",
                fps=video_fps,
            )
            try:
                self._video.start(self._context)
            except Exception:
                logger.exception("EpisodeVideoRecorder.start failed; video disabled for this episode")
                self._video = None

    # -- Identifiers -------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return True

    @property
    def sid(self) -> str:
        return self._sid

    @property
    def eid(self) -> str:
        return self._eid

    @property
    def eval_id(self) -> str:
        return self._eval_id

    @property
    def db_path(self) -> str:
        return str(self._store.db_path)

    # -- Capture API -------------------------------------------------------

    def record_video(self, frame: "np.ndarray") -> None:
        if self._video is not None:
            self._video.record(frame)

    def record_step(self, row: dict[str, Any]) -> None:
        if not self._record_step:
            return
        step_id = int(row.get("step", self._next_step))
        self._next_step = step_id + 1
        existing = self._steps.setdefault(step_id, {})
        existing.update((k, v) for k, v in row.items() if k != "step")

    # -- Close (orchestrator) ---------------------------------------------

    def close(
        self,
        *,
        status: EpisodeStatus,
        metrics: dict[str, Any],
        task_name: str = "",
        episode_id: int = 0,
        steps: int = 0,
        elapsed_sec: float = 0.0,
        failure_reason: str | None = None,
        failure_detail: str | None = None,
    ) -> None:
        if self._closed:
            return
        self._closed = True

        if self._video is not None:
            try:
                self._video.save(status=status)
            except FileExistsError as exc:
                logger.warning("Episode video already exists: %s", exc)
            except Exception:
                logger.exception("video.save failed for sid=%s eid=%s", self._sid, self._eid)
            self._video = None

        try:
            jsonl_name = (self._filename_stem + ".jsonl").format(status=status, **self._context)
        except Exception:
            logger.exception("filename_stem render failed; using fallback name")
            jsonl_name = f"{self._sid}-{self._eid}_{status}.jsonl"
        # Store the path RELATIVE to the SQLite file's directory whenever
        # possible so that `vla-eval merge` resolves it correctly whether the
        # run happens inside Docker (where output_dir = /workspace/results)
        # and merge happens on the host (different absolute prefix).
        abs_jsonl = (self._output_dir / jsonl_name).resolve()
        db_dir = Path(self._store.db_path).resolve().parent
        try:
            jsonl_path = str(abs_jsonl.relative_to(db_dir))
        except ValueError:
            jsonl_path = str(abs_jsonl)

        try:
            self._store.upsert_step_rows(self._sid, self._eid, self._steps)
        except Exception:
            logger.exception("Failed to upsert step rows for sid=%s eid=%s", self._sid, self._eid)

        try:
            self._store.upsert_episode_result(
                sid=self._sid,
                eid=self._eid,
                eval_id=self._eval_id,
                task_name=task_name,
                episode_id=episode_id,
                status=status,
                metrics=metrics,
                steps=steps,
                elapsed_sec=elapsed_sec,
                context=self._context,
                jsonl_path=jsonl_path,
                failure_reason=failure_reason,
                failure_detail=failure_detail,
            )
        except Exception:
            logger.exception("Failed to upsert episode result for sid=%s eid=%s", self._sid, self._eid)


class NullEpisodeRecorder(EpisodeRecorder):
    """No-op recorder used when recording is off."""

    def __init__(self) -> None:  # type: ignore[override]
        self._closed = True
        self._video = None
        self._steps = {}

    @property
    def is_active(self) -> bool:  # type: ignore[override]
        return False

    @property
    def sid(self) -> str:  # type: ignore[override]
        return ""

    @property
    def eid(self) -> str:  # type: ignore[override]
        return ""

    @property
    def eval_id(self) -> str:  # type: ignore[override]
        return ""

    @property
    def db_path(self) -> str:  # type: ignore[override]
        return ""

    def record_video(self, frame: "np.ndarray") -> None:  # type: ignore[override]
        pass

    def record_step(self, row: dict[str, Any]) -> None:  # type: ignore[override]
        pass

    def close(  # type: ignore[override]
        self,
        *,
        status: EpisodeStatus,
        metrics: dict[str, Any],
        task_name: str = "",
        episode_id: int = 0,
        steps: int = 0,
        elapsed_sec: float = 0.0,
        failure_reason: str | None = None,
        failure_detail: str | None = None,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# StepRecorder — lightweight external API (model server side)
# ---------------------------------------------------------------------------


class StepRecorder:
    """Per-episode step-row writer for external callers (e.g. model server).

    Open against the ``db_path`` forwarded in ``EPISODE_START.recording``;
    rows are field-unioned with the harness's rows via ``json_patch``.
    """

    def __init__(self, db_path: str | Path, sid: str, eid: str) -> None:
        self._store = RecordingStore(db_path)
        self._sid = sid
        self._eid = eid
        self._steps: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._closed = False

    def record(self, row: dict[str, Any]) -> None:
        step_id = int(row.get("step", self._next_step))
        self._next_step = step_id + 1
        existing = self._steps.setdefault(step_id, {})
        existing.update((k, v) for k, v in row.items() if k != "step")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._store.upsert_step_rows(self._sid, self._eid, self._steps)
        except Exception:
            logger.exception("StepRecorder: failed to upsert step rows for sid=%s eid=%s", self._sid, self._eid)
        self._store.close()

    def __enter__(self) -> "StepRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
