"""Materialize the SQLite recording → per-episode jsonl + per-benchmark aggregate JSON."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vla_eval import __version__
from vla_eval.recording import db_path_for_eval
from vla_eval.results.collector import _aggregate_metrics, _build_task_result, _extract_seed, print_task_table

logger = logging.getLogger(__name__)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    os.replace(str(tmp), str(path))


def _write_json_atomic(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))


def merge_db(db_path: Path, output_dir: Path, *, report: bool = False) -> list[dict[str, Any]]:
    """Read one consistent snapshot, then write artifacts and optionally report metrics."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Recording DB not found: {db_path}")
    aggregates: list[dict[str, Any]] = []
    run = None
    with _recording_snapshot(db_path) as conn:
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='run_metadata'").fetchone():
            run = conn.execute("SELECT eval_id, config FROM run_metadata WHERE singleton=1").fetchone()
        for row in conn.execute("SELECT eval_id, safe_name, metadata FROM eval_metadata"):
            metadata = json.loads(row["metadata"])
            aggregate = _build_aggregate(conn, row["eval_id"], row["safe_name"], metadata, output_dir)
            path = output_dir / f"{row['safe_name']}_aggregate.json"
            _write_json_atomic(path, aggregate)
            logger.info("Wrote aggregate: %s (%d episodes)", path, aggregate.get("num_episodes_total", 0))
            aggregates.append(aggregate)
    if report and run:
        _report_aggregates(run["eval_id"], json.loads(run["config"]), aggregates)
    return aggregates


@contextmanager
def _recording_snapshot(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Release source locks before exporting, without buffering all steps in RAM."""
    with tempfile.TemporaryDirectory(prefix="vla-eval-merge-") as directory:
        snapshot = sqlite3.connect(str(Path(directory) / "snapshot.sqlite"))
        try:
            source = sqlite3.connect(db_path.resolve().as_uri() + "?mode=rw", uri=True, timeout=60.0)
            try:
                source.execute("BEGIN")
                # Establish the read snapshot with the connection's bounded busy timeout.
                source.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
                source.backup(snapshot)
            finally:
                source.close()
            snapshot.row_factory = sqlite3.Row
            yield snapshot
        finally:
            snapshot.close()


def _report_aggregates(eval_id: str, config: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    from vla_eval.tracking import call_each, get_reporting_trackers

    trackers = get_reporting_trackers((config.get("tracking") or {}).get("report_to"))
    try:
        call_each(trackers, "on_eval_begin", eval_id, config)
        for aggregate in aggregates:
            name = aggregate.get("benchmark", "")
            call_each(trackers, "on_benchmark_begin", name, {})
            call_each(trackers, "on_benchmark_end", name, aggregate)
        call_each(trackers, "on_eval_end", aggregates)
    finally:
        call_each(trackers, "close")


def _build_aggregate(
    conn: sqlite3.Connection,
    eval_id: str,
    safe_name: str,
    metadata: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """For one benchmark: walk its episodes, write per-episode jsonl, build aggregate."""
    metric_keys = dict(metadata.get("metric_keys") or {})
    config = metadata.get("config") or {}

    tasks_acc: dict[str, list[dict[str, Any]]] = {}
    all_episodes: list[dict[str, Any]] = []
    episode_count = 0

    for er in conn.execute(
        """
        SELECT sid, eid, task_name, episode_id, status, metrics, steps, elapsed_sec,
               context, jsonl_path, failure_reason, failure_detail
        FROM episode_results
        WHERE eval_id = ?
        ORDER BY task_name, episode_id, sid, eid
        """,
        (eval_id,),
    ):
        context = json.loads(er["context"]) if er["context"] else {}
        metrics = json.loads(er["metrics"]) if er["metrics"] else {}
        episode_row: dict[str, Any] = {
            "sid": er["sid"],
            "eid": er["eid"],
            "episode_id": er["episode_id"],
            "metrics": metrics,
            "steps": er["steps"],
            "elapsed_sec": er["elapsed_sec"],
            **context,
        }
        if er["failure_reason"]:
            episode_row["failure_reason"] = er["failure_reason"]
        if er["failure_detail"]:
            episode_row["failure_detail"] = er["failure_detail"]

        task_name = str(er["task_name"] or "_unknown")
        tasks_acc.setdefault(task_name, []).append(episode_row)
        all_episodes.append(episode_row)
        episode_count += 1

        if er["jsonl_path"]:
            jsonl_p = Path(er["jsonl_path"])
            if not jsonl_p.is_absolute():
                jsonl_p = output_dir / jsonl_p
            rows = _read_episode_steps(conn, er["sid"], er["eid"])
            if rows:
                _write_jsonl_atomic(jsonl_p, rows)

    tasks_out: list[Any] = []
    for task_name in sorted(tasks_acc):
        tasks_out.append(_build_task_result(task_name, tasks_acc[task_name], metric_keys))

    body: dict[str, Any] = {
        "benchmark": metadata.get("benchmark", safe_name),
        "mode": metadata.get("mode"),
        "harness_version": metadata.get("harness_version") or __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks_out,
        "config": config,
        "eval_id": eval_id,
    }
    if "server_info" in metadata:
        body["server_info"] = metadata["server_info"]
    if "render" in metadata:
        body["render"] = metadata["render"]
    seed = _extract_seed(config)
    if seed is not None:
        body["seed"] = seed
    if metric_keys:
        body["metric_keys"] = metric_keys
        _aggregate_metrics(body, all_episodes, metric_keys)
    body["num_episodes_total"] = episode_count
    body["num_errors"] = sum(t.get("num_errors", 0) for t in tasks_out)
    return body


def _read_episode_steps(conn: sqlite3.Connection, sid: str, eid: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sr in conn.execute(
        "SELECT step_id, fields FROM step_rows WHERE sid = ? AND eid = ? ORDER BY step_id",
        (sid, eid),
    ):
        row: dict[str, Any] = {"step": sr["step_id"]}
        try:
            row.update(json.loads(sr["fields"]))
        except Exception:
            logger.warning(
                "step_rows row for sid=%s eid=%s step=%d has bad JSON; skipping",
                sid,
                eid,
                sr["step_id"],
            )
            continue
        rows.append(row)
    return rows


def merge_eval(output_dir: Path, eval_id: str) -> list[dict[str, Any]]:
    """Convenience wrapper: ``merge_db(db_path_for_eval(output_dir, eval_id), output_dir)``."""
    return merge_db(db_path_for_eval(output_dir, eval_id), output_dir)


def print_merge_summary(aggregates: list[dict[str, Any]]) -> None:
    """Reuse the collector's task table for the final printed summary."""
    from rich.console import Console

    console = Console(highlight=False)
    for body in aggregates:
        rate = body.get("mean_success", 0.0)
        rate_color = "green" if rate >= 0.5 else "red"
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold]Benchmark: {body['benchmark']}[/bold] (mode: {body.get('mode')})")
        console.print(f"{'=' * 60}")
        print_task_table(console, body["tasks"], rate, rate_color)
        console.print(f"{'=' * 60}\n")
