"""SQLite-backed per-episode step rows, episode results, and eval metadata."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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


class RecordingError(RuntimeError):
    """Required recording data could not be persisted."""


def _json_default(obj: Any) -> Any:
    """JSON fallback that turns numpy arrays/scalars into native Python via ``.tolist()``."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


SCHEMA_SQL = """
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS run_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    eval_id TEXT NOT NULL,
    config TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_metadata (
    eval_id    TEXT PRIMARY KEY,
    safe_name  TEXT NOT NULL,
    metadata   TEXT NOT NULL  -- JSON: benchmark, mode, config, harness_version, server_info, metric_keys
);

CREATE TABLE IF NOT EXISTS eval_shards (
    eval_id TEXT NOT NULL,
    shard_id INTEGER NOT NULL,
    num_shards INTEGER NOT NULL,
    complete INTEGER NOT NULL,
    PRIMARY KEY (eval_id, shard_id)
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
    jsonl_path      TEXT,            -- resolved final filename for ``vla-eval export``
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
COMMIT;
"""


# ---------------------------------------------------------------------------
# Recording config helpers
# ---------------------------------------------------------------------------


# Default when ``recording.filename_stem`` is omitted. Numeric orchestrator ids keep
# paths short and collision-free across shards; ``episode_id`` is the raw run episode,
# not the throughput-mode-wrapped ``episode_idx``.
DEFAULT_FILENAME_STEM = "{benchmark_safe_name}/task{task_idx:04d}_ep{episode_id:04d}_{status}"


def serializable_task_kwargs(task: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly subset of *task* — safe for str.format and SQLite JSON columns."""
    return {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool))}


def recording_filename_context(*, benchmark_safe_name: str, task_idx: int, episode_id: int) -> dict[str, Any]:
    """Filename-template keys injected by the orchestrator."""
    return {"benchmark_safe_name": benchmark_safe_name[:96], "task_idx": task_idx, "episode_id": episode_id}


# ---------------------------------------------------------------------------
# RecordingStore — SQLite connection + idempotent writes
# ---------------------------------------------------------------------------


def db_path_for_eval(output_dir: str | Path, eval_id: str) -> Path:
    """Canonical SQLite path for an eval. All writers for an evaluation point here."""
    return Path(output_dir) / f"recording-{eval_id}.sqlite"


def _host_translate(path: Path) -> Path:
    """Rewrite ``/workspace/results/...`` to the host root under
    ``VLA_EVAL_HOST_OUTPUT_DIR`` (set by the outer CLI on ``docker run``).
    Passes through unchanged otherwise."""
    host_root = os.environ.get("VLA_EVAL_HOST_OUTPUT_DIR")
    if not host_root:
        return path
    try:
        rel = path.resolve().relative_to(Path("/workspace/results"))
    except ValueError:
        return path
    return Path(host_root) / rel


class RecordingStore:
    """One connection per process; SQLite serializes recording transactions."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(self.db_path.parent, os.W_OK | os.X_OK):
            raise PermissionError(
                f"Recording directory must be writable to create rollback journals: {self.db_path.parent}. "
                "Use docker.user: host or grant all writers directory access through a shared group."
            )
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=60.0)
        try:
            self._init_schema()
            # Different-UID writers also need directory write access for rollback journals.
            try:
                os.chmod(self.db_path, 0o666)
            except OSError:
                pass
        except BaseException:
            self._conn.close()
            raise

    def _init_schema(self) -> None:
        """Retry contention during startup, including conversion of an idle WAL database."""
        for attempt in range(40):
            try:
                mode = self._conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
                if mode != "delete":
                    raise RuntimeError(f"Expected DELETE journal mode, got {mode}")
                self._conn.execute("PRAGMA synchronous=EXTRA")
                self._conn.executescript(SCHEMA_SQL)
                return
            except sqlite3.OperationalError as exc:
                self._conn.rollback()
                if "locked" not in str(exc).lower() or attempt == 39:
                    raise
                time.sleep(min(1.0, 0.1 * (attempt + 1)))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Allow a caller to commit step rows and episode results together."""
        if self._conn.in_transaction:
            yield
            return
        with self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            yield

    def set_run_metadata(self, eval_id: str, config: dict[str, Any]) -> None:
        # Persist only reporting settings; resolved docker.env can contain credentials.
        reporting = {"tracking": {"report_to": (config.get("tracking") or {}).get("report_to")}}
        with self.transaction():
            self._conn.execute(
                "INSERT OR IGNORE INTO run_metadata (singleton, eval_id, config) VALUES (1, ?, ?)",
                (eval_id, json.dumps(reporting, default=_json_default)),
            )
            stored_id = self._conn.execute("SELECT eval_id FROM run_metadata").fetchone()[0]
            if stored_id != eval_id:
                raise ValueError(f"Recording belongs to evaluation {stored_id}, not {eval_id}")

    def close(self) -> None:
        self._conn.close()

    def set_shard_complete(self, eval_id: str, shard_id: int, num_shards: int, *, complete: bool) -> None:
        with self.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO eval_shards VALUES (?, ?, ?, ?)",
                (eval_id, shard_id, num_shards, complete),
            )

    def upsert_eval_metadata(self, eval_id: str, safe_name: str, metadata: dict[str, Any]) -> None:
        """Keep the first metadata; flag renderer disagreement between shards."""
        with self.transaction():
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO eval_metadata (eval_id, safe_name, metadata) VALUES (?, ?, ?)",
                (eval_id, safe_name, json.dumps(metadata, default=_json_default)),
            )
            if cur.rowcount:
                return
            row = self._conn.execute("SELECT metadata FROM eval_metadata WHERE eval_id = ?", (eval_id,)).fetchone()
            stored = json.loads(row[0]).get("render") if row else None
            if isinstance(stored, dict):
                # Ignore the flag itself, or every later agreeing shard would re-warn.
                stored = {k: v for k, v in stored.items() if k != "divergent"}
            mine = metadata.get("render")
            if stored is None or mine is None or stored == mine:
                return
            logger.warning(
                "Renderer provenance differs across shards for eval_id=%s (stored=%r, this shard=%r); "
                "flagging render.divergent",
                eval_id,
                stored,
                mine,
            )
            self._conn.execute(
                "UPDATE eval_metadata SET metadata = json_set(metadata, '$.render.divergent', json('true')) "
                "WHERE eval_id = ?",
                (eval_id,),
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
        with self.transaction():
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
        with self.transaction():
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
        filename_context: dict[str, Any] | None = None,
        record_video: bool = False,
        record_step: bool = True,
        video_fps: int = 20,
        step_fields: Iterable[str] | None = None,
        allowed_fields: Iterable[str] | None = None,
    ) -> None:
        self._store = store
        self._sid = sid
        self._eid = eid
        self._eval_id = eval_id
        self._output_dir = Path(output_dir)
        self._filename_stem = filename_stem
        self._context = dict(context)
        self._filename_context = {**self._context, **(filename_context or {})}
        self._record_step = record_step
        self._steps: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._closed = False
        self._video: Any = None
        # step_fields=None → record everything in ``allowed_fields`` (or
        # everything, if both are None). Explicit empty list = record nothing.
        allowed = frozenset(allowed_fields) if allowed_fields is not None else None
        if step_fields is None:
            self._step_fields: frozenset[str] | None = allowed
        else:
            if isinstance(step_fields, str):
                raise TypeError(
                    f"step_fields must be a list of field names, got bare string {step_fields!r} — "
                    "did you forget the YAML list brackets?"
                )
            requested = frozenset(step_fields)
            if allowed is not None:
                unknown = requested - allowed
                if unknown:
                    raise ValueError(f"Unknown step_fields: {sorted(unknown)}. Valid: {sorted(allowed)}")
            self._step_fields = requested
        if record_video:
            from vla_eval.benchmarks.video import EpisodeVideoRecorder

            self._video = EpisodeVideoRecorder(
                output_dir=self._output_dir,
                filename=filename_stem + ".mp4",
                fps=video_fps,
            )
            try:
                self._video.start(self._filename_context)
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
        """Host-resolvable SQLite path (translated when orchestrator is in docker)."""
        return str(_host_translate(self._store.db_path))

    # -- Capture API -------------------------------------------------------

    def record_video(self, frame: "np.ndarray | None") -> None:
        """Append one frame to the per-episode mp4. ``None`` is a no-op so
        benchmarks can pass ``self._extract_frame(obs)`` directly."""
        if frame is None or self._video is None:
            return
        self._video.record(frame)

    def record_step(self, **fields: Any) -> None:
        """``step_fields`` filters caller keys; ``step`` kwarg overrides
        auto-increment (used to amend a previous row)."""
        if not self._record_step:
            return
        if self._step_fields is not None:
            fields = {k: v for k, v in fields.items() if k == "step" or k in self._step_fields}
        step_id = int(fields.pop("step", self._next_step))
        self._next_step = step_id + 1
        self._steps.setdefault(step_id, {}).update(fields)

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
            jsonl_name = (self._filename_stem + ".jsonl").format(status=status, **self._filename_context)
        except Exception:
            logger.exception("filename_stem render failed; using fallback name")
            jsonl_name = f"{self._sid}-{self._eid}_{status}.jsonl"
        # Relative paths let host-side export resolve recordings made inside containers.
        abs_jsonl = (self._output_dir / jsonl_name).resolve()
        db_dir = Path(self._store.db_path).resolve().parent
        try:
            jsonl_path = str(abs_jsonl.relative_to(db_dir))
        except ValueError:
            jsonl_path = str(abs_jsonl)

        try:
            with self._store.transaction():
                self._store.upsert_step_rows(self._sid, self._eid, self._steps)
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
        except Exception as exc:
            raise RecordingError(f"Failed to save episode sid={self._sid} eid={self._eid}") from exc


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

    def record_video(self, frame: "np.ndarray | None") -> None:  # type: ignore[override]
        pass

    def record_step(self, **fields: Any) -> None:  # type: ignore[override]
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
        except Exception as exc:
            raise RecordingError(f"Failed to save steps sid={self._sid} eid={self._eid}") from exc
        finally:
            self._store.close()

    def __enter__(self) -> "StepRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
