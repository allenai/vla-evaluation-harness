"""SQLite-backed per-episode step rows, episode results, and eval metadata."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections.abc import Iterable
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
PRAGMA busy_timeout=60000;  -- 60s, consistent with connect(timeout=60); SCHEMA_SQL runs last so this is the effective post-init value

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


# Default when ``recording.filename_stem`` is omitted. Use orchestrator-provided
# numeric identifiers rather than raw task names: natural-language task
# descriptions can be very long, duplicate across benchmark entries, and contain
# path separators. ``episode_id`` is the raw run episode, not the benchmark's
# possibly wrapped ``episode_idx`` used for simulator initial states.
DEFAULT_FILENAME_STEM = "{benchmark_safe_name}/task{task_idx:04d}_ep{episode_id:04d}_{status}"

_SAFE_FILENAME_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def serializable_task_kwargs(task: dict[str, Any]) -> dict[str, Any]:
    """JSON-friendly subset of *task* — safe for str.format and SQLite JSON columns."""
    return {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool))}


def _safe_filename_component(value: Any, *, max_len: int = 96) -> str:
    raw = str(value).strip()
    text = raw
    text = _SAFE_FILENAME_COMPONENT_RE.sub("_", text).strip("._-")
    if not text:
        return "unknown"
    return text[:max_len].rstrip("._-") or "unknown"


def recording_filename_context(
    task: dict[str, Any], *, benchmark_safe_name: str, task_idx: int, episode_id: int
) -> dict[str, Any]:
    """Context used only for rendering recording filenames.

    The persisted episode context keeps the benchmark's original task values.
    Filenames additionally get stable orchestrator-level keys that are short,
    path-safe, and shard-independent.
    """
    context = serializable_task_kwargs(task)
    return {
        **context,
        "benchmark_safe_name": _safe_filename_component(benchmark_safe_name),
        "task_idx": int(task_idx),
        "episode_id": int(episode_id),
    }


# ---------------------------------------------------------------------------
# RecordingStore — SQLite connection + idempotent writes
# ---------------------------------------------------------------------------


def db_path_for_eval(output_dir: str | Path, eval_id: str) -> Path:
    """Canonical SQLite path for an eval. All shards on one host point here."""
    return Path(output_dir) / f"recording-{eval_id}.sqlite"


def eval_id_from_db_path(path: str | Path) -> str | None:
    """Inverse of :func:`db_path_for_eval`: ``recording-<id>.sqlite`` → ``<id>`` or ``None``."""
    stem = Path(path).stem
    return stem[len("recording-") :] if stem.startswith("recording-") else None


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


# ---------------------------------------------------------------------------
# Network/parallel filesystem detection (WAL safety)
# ---------------------------------------------------------------------------

# SQLite WAL coordinates writers through a memory-mapped ``-shm`` index plus POSIX
# advisory locks that network/parallel filesystems do not provide with the coherence
# SQLite assumes, so concurrent multi-process writes can corrupt the DB ("database
# disk image is malformed"). A shared multi-writer sink can't be transparently
# relocated like a per-process cache, so we warn and let the caller place it on local
# storage. Detection mirrors pixi's ``detect_network_filesystem`` (prefix-dev/pixi,
# crates/pixi_config): ``statfs(2)`` f_type compared against known magics.
_NETWORK_FS_MAGICS: dict[int, str] = {
    0x6969: "nfs",  # NFS_SUPER_MAGIC
    0x517B: "smb",  # SMB_SUPER_MAGIC (smbfs)
    0xFF534D42: "cifs",  # fs/smb/client/cifsfs.h
    0x65735546: "fuse",  # FUSE_SUPER_MAGIC
    0x0187: "autofs",  # AUTOFS_SUPER_MAGIC
    0x19830326: "beegfs",  # BeeGFS / fhgfs
    0x0BD00BD0: "lustre",  # Lustre LL_SUPER_MAGIC
    0x47504653: "gpfs",  # GPFS / IBM Spectrum Scale ("GPFS")
    0x00C36400: "ceph",  # CephFS CEPH_SUPER_MAGIC
}


def _statfs_f_type(path: str) -> int | None:
    """``statfs(2)`` f_type magic (low 32 bits) for *path*, or None if unavailable. Linux-only:
    ``struct statfs`` layout is platform-specific, so other OSes return None rather than misread."""
    if sys.platform != "linux":
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        statfs = libc.statfs
        statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        statfs.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(256)  # >= sizeof(struct statfs) on Linux
        if statfs(os.fsencode(path), buf) != 0:
            return None
        # f_type is the first field of ``struct statfs`` (__fsword_t == long on Linux).
        return ctypes.c_long.from_buffer_copy(buf).value & 0xFFFFFFFF
    except (OSError, ValueError, AttributeError):  # best-effort: never break recording
        return None


def _nearest_existing(path: Path) -> Path | None:
    """Closest ancestor of *path* that exists (the DB file may not exist yet)."""
    p = path
    while True:
        try:
            if p.exists():
                return p
        except OSError:  # unreadable ancestor → give up (best-effort)
            return None
        if p.parent == p:
            return None
        p = p.parent


def _detect_network_fs(path: Path) -> str | None:
    """Name of the network/parallel filesystem *path* lives on (where SQLite WAL is unsafe),
    else None. Mirrors pixi: ``statfs(2)`` f_type vs known magics. Best-effort, Linux-only
    (None when it can't tell). Set ``VLA_EVAL_DISABLE_NETFS_WARNING`` to skip."""
    if os.environ.get("VLA_EVAL_DISABLE_NETFS_WARNING"):
        return None
    existing = _nearest_existing(path)
    if existing is None:
        return None
    f_type = _statfs_f_type(str(existing))
    return _NETWORK_FS_MAGICS.get(f_type) if f_type is not None else None


# RecordingStore is constructed many times (per shard, and per episode by an external
# StepRecorder), so warn about a network filesystem at most once per DB path.
_checked_netfs_paths: set[str] = set()


class RecordingStore:
    """SQLite connection holder. One per process; same-file concurrency via WAL."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.db_path)
        if key not in _checked_netfs_paths:
            _checked_netfs_paths.add(key)
            netfs = _detect_network_fs(self.db_path.parent)
            if netfs is not None:
                logger.warning(
                    "Recording DB %s is on a %r filesystem. SQLite WAL relies on coherent shared "
                    "memory + advisory locks that network/parallel filesystems do not reliably "
                    "provide, so concurrent multi-writer recording can corrupt it ('database disk "
                    "image is malformed'). Place the recording output on node-local storage (e.g. "
                    "/dev/shm or a local disk) and copy the finished file out, or use a single writer.",
                    self.db_path,
                    netfs,
                )
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=60.0)
        # journal_mode=WAL (SCHEMA_SQL line 1) switches under an exclusive lock
        # SQLite won't reliably retry; arm busy_timeout first + retry so N shards
        # opening one fresh DB at once don't lose the race ("database is locked").
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._init_schema()
        # Mode 666 on main + WAL/SHM so external writers (different uid) can
        # co-write via field-union upsert. SQLite WAL needs SHM writable.
        for suffix in ("", "-wal", "-shm"):
            try:
                os.chmod(str(self.db_path) + suffix, 0o666)
            except OSError:
                pass

    def _init_schema(self) -> None:
        """Run the idempotent schema script, retrying the WAL-switch lock race
        that surfaces when many writers open a fresh DB concurrently."""
        for attempt in range(40):
            try:
                self._conn.executescript(SCHEMA_SQL)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 39:
                    raise
                time.sleep(min(1.0, 0.1 * (attempt + 1)))

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
        filename_context: dict[str, Any] | None = None,
        record_video: bool = True,
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
        except Exception:
            logger.exception("StepRecorder: failed to upsert step rows for sid=%s eid=%s", self._sid, self._eid)
        self._store.close()

    def __enter__(self) -> "StepRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
