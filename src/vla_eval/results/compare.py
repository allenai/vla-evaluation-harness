"""``vla-eval compare``: two runs' aggregate JSONs, paired by episode, or one run against a reported score.

Inputs are the ``<benchmark>_aggregate.json`` files ``vla-eval run`` / ``vla-eval export`` write: they carry every
episode (``tasks[].episodes[]`` with ``episode_id``, ``metrics`` and, since recording, ``task_idx``), so nothing
else is needed.  Episodes pair on ``(task, task_idx, episode_id)`` — ``task_idx`` only when both runs record it —
so a pair is the same task and initial state run twice.  Paired runs get the exact McNemar test (binary metrics)
and a task-stratified paired bootstrap of the mean difference; when nothing pairs the comparison falls back to
independent samples (Newcombe for rates, bootstrap otherwise).  Values are taken exactly as the aggregate's mean
takes them (:func:`vla_eval.results.stats.metric_values`): harness errors count as failures.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from vla_eval.results import stats

_PairKey = tuple[Any, ...]
_MAX_TASK_ROWS = 60  # longer per-task tables (CALVIN: 1000 one-episode sequences) only go to --json


def load_aggregate(path: str | Path) -> dict[str, Any]:
    """Read one aggregate JSON; ``ValueError`` unless it has per-episode rows under ``tasks[].episodes``."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict) or not isinstance(body.get("tasks"), list):
        raise ValueError(f"{path}: not an aggregate JSON (no 'tasks' list)")
    if not any(isinstance(t, dict) and t.get("episodes") for t in body["tasks"]):
        raise ValueError(f"{path}: aggregate has no per-episode rows (tasks[].episodes)")
    return body


def parse_reference(text: str, binary: bool) -> float:
    """``"96.9%"`` → 0.969; a bare number is on the metric's own scale (a fraction for rates)."""
    pct = text.strip().endswith("%")
    value = float(text.strip().rstrip("%")) / (100 if pct else 1)
    if binary and not 0.0 <= value <= 1.0:
        raise ValueError(f"reference {text} is outside [0, 1] for a rate; pass a fraction or append %")
    return value


def _episodes(body: dict[str, Any], metric: str) -> list[tuple[str, dict[str, Any], Any]]:
    """``(task, episode, value)`` for every episode with a numeric ``metrics[metric]``, in aggregate order."""
    out = []
    for task in body["tasks"]:
        for ep in task.get("episodes") or []:
            values = stats.metric_values([ep], metric)
            if values:
                out.append((str(task.get("task")), ep, values[0]))
    return out


def _mean(values: list[Any]) -> float:
    return sum(float(v) for v in values) / len(values)


def _r(x: float) -> float:
    return round(x, 4)


def _interval(lo: float, hi: float) -> list[float]:
    return [_r(lo), _r(hi)]


def summarize(body: dict[str, Any], metric: str, path: str | Path | None = None) -> dict[str, Any]:
    """One run's mean, 95% interval (as the aggregate's ``ci95_<metric>``), n and harness-error count."""
    rows = _episodes(body, metric)
    if not rows:
        raise ValueError(f"{path or body.get('benchmark')}: no episode has a numeric metric {metric!r}")
    values = [v for _, _, v in rows]
    ci, n = stats.metric_interval(values)
    return {
        "path": str(path) if path is not None else None,
        "benchmark": body.get("benchmark"),
        "mean": _r(_mean(values)),
        "ci95": ci,
        "n": n,
        "num_errors": sum(1 for _, ep, _ in rows if ep.get("failure_reason")),
    }


def _keyed(rows: list[tuple[str, dict[str, Any], Any]], use_idx: bool) -> tuple[dict[_PairKey, Any], int]:
    """Map pair key → ``(task, value)``.  Episodes sharing a key are ambiguous: dropped, and their count returned."""
    groups: dict[_PairKey, list[tuple[str, Any]]] = {}
    for task, ep, value in rows:
        key = (task, ep.get("task_idx"), ep.get("episode_id")) if use_idx else (task, ep.get("episode_id"))
        groups.setdefault(key, []).append((task, value))
    keyed = {k: g[0] for k, g in groups.items() if len(g) == 1}
    return keyed, sum(len(g) for g in groups.values() if len(g) > 1)


def _difference_stats(
    a: list[Any], b: list[Any], *, paired: bool, binary: bool, strata: list[str] | None = None
) -> tuple[float, list[float], str]:
    """``(estimate, [lo, hi], method)`` of ``mean(a) - mean(b)``; paired inputs are aligned pair by pair."""
    estimate = _mean(a) - _mean(b)
    if paired:
        diffs = [float(x) - float(y) for x, y in zip(a, b)]
        lo, hi = stats.paired_bootstrap_difference(diffs, strata)
        method = "paired bootstrap" + (", stratified by task" if strata is not None else "")
    elif binary:
        xa, xb = sum(bool(v) for v in a), sum(bool(v) for v in b)
        lo, hi = stats.newcombe_difference_interval(xa, len(a), xb, len(b))
        method = "Newcombe hybrid score (independent)"
    else:
        lo, hi = stats.independent_bootstrap_difference([float(v) for v in a], [float(v) for v in b])
        method = "bootstrap (independent)"
    return _r(estimate), _interval(lo, hi), method


def _mcnemar(a: list[Any], b: list[Any]) -> dict[str, Any]:
    a_only = sum(1 for x, y in zip(a, b) if bool(x) and not bool(y))
    b_only = sum(1 for x, y in zip(a, b) if bool(y) and not bool(x))
    return {"a_only": a_only, "b_only": b_only, "p_value": stats.mcnemar_exact(a_only, b_only)}


def compare_runs(a: dict[str, Any], b: dict[str, Any], metric: str = "success") -> dict[str, Any]:
    """Compare run A against run B on ``metric`` (difference is A − B), overall and per task."""
    rows_a, rows_b = _episodes(a, metric), _episodes(b, metric)
    if not rows_a or not rows_b:
        raise ValueError(f"no episode has a numeric metric {metric!r}")
    binary = stats.is_binary(v for _, _, v in rows_a + rows_b)
    use_idx = all("task_idx" in ep for _, ep, _ in rows_a + rows_b)
    keyed_a, dupes_a = _keyed(rows_a, use_idx)
    keyed_b, dupes_b = _keyed(rows_b, use_idx)
    common = [k for k in keyed_a if k in keyed_b]
    out: dict[str, Any] = {
        "pair_key": ["task", "task_idx", "episode_id"] if use_idx else ["task", "episode_id"],
        "paired": len(common),
        "unmatched_a": len(rows_a) - len(common),
        "unmatched_b": len(rows_b) - len(common),
    }
    if dupes_a or dupes_b:
        out["duplicate_keys"] = [dupes_a, dupes_b]

    by_task: dict[str, tuple[list[Any], list[Any]]] = {}
    if common:
        out["method"] = "paired"
        tasks = [keyed_a[k][0] for k in common]
        va, vb = [keyed_a[k][1] for k in common], [keyed_b[k][1] for k in common]
        out["mean_a"], out["mean_b"] = _r(_mean(va)), _r(_mean(vb))
        # A one-pair stratum never varies under resampling, so stratify only when every task has two or more
        # (CALVIN's 1000 one-episode sequences are resampled as one pool).
        stratify = min(Counter(tasks).values()) >= 2
        out["difference"], out["ci95"], out["interval"] = _difference_stats(
            va, vb, paired=True, binary=binary, strata=tasks if stratify else None
        )
        if binary:
            out["mcnemar"] = _mcnemar(va, vb)
        for k in common:
            ta, tb = by_task.setdefault(keyed_a[k][0], ([], []))
            ta.append(keyed_a[k][1])
            tb.append(keyed_b[k][1])
    else:
        out["method"] = "independent"
        va, vb = [v for _, _, v in rows_a], [v for _, _, v in rows_b]
        out["mean_a"], out["mean_b"] = _r(_mean(va)), _r(_mean(vb))
        out["difference"], out["ci95"], out["interval"] = _difference_stats(va, vb, paired=False, binary=binary)
        for task, _, v in rows_a:
            by_task.setdefault(task, ([], []))[0].append(v)
        for task, _, v in rows_b:
            if task in by_task:
                by_task[task][1].append(v)
        by_task = {t: ab for t, ab in by_task.items() if ab[1]}

    task_rows = []
    for task in sorted(by_task):
        ta, tb = by_task[task]
        diff, ci, _ = _difference_stats(ta, tb, paired=bool(common), binary=binary)
        row: dict[str, Any] = {
            "task": task,
            "n_a": len(ta),
            "n_b": len(tb),
            "mean_a": _r(_mean(ta)),
            "mean_b": _r(_mean(tb)),
            "difference": diff,
            "ci95": ci,
        }
        if common and binary:
            row["p_value"] = _mcnemar(ta, tb)["p_value"]
        task_rows.append(row)
    out["tasks"] = task_rows
    return out


def reference_verdict(run: dict[str, Any], reference: float, tolerance: float = stats.TOLERANCE) -> dict[str, Any]:
    """The reproduction verdict for a :func:`summarize` result against a reported score."""
    verdict = stats.reproduction_verdict(run["mean"], run["ci95"], reference, tolerance)
    symbol, label = stats.VERDICTS[verdict]
    return {
        "value": reference,
        "difference": _r(run["mean"] - reference),
        "tolerance": tolerance,
        "verdict": verdict,
        "symbol": symbol,
        "label": label,
    }


def build_report(
    a_path: str | Path,
    b_path: str | Path | None = None,
    *,
    metric: str = "success",
    reference: str | None = None,
    tolerance: float = stats.TOLERANCE,
) -> dict[str, Any]:
    """Everything ``vla-eval compare`` prints (and emits with ``--json``)."""
    a = load_aggregate(a_path)
    report: dict[str, Any] = {"metric": metric, "a": summarize(a, metric, a_path)}
    values = [v for _, _, v in _episodes(a, metric)]
    if b_path is not None:
        b = load_aggregate(b_path)
        report["b"] = summarize(b, metric, b_path)
        report["comparison"] = compare_runs(a, b, metric)
        values += [v for _, _, v in _episodes(b, metric)]
    report["binary"] = stats.is_binary(values)
    if reference is not None:
        report["reference"] = reference_verdict(report["a"], parse_reference(reference, report["binary"]), tolerance)
    return report


def print_report(report: dict[str, Any]) -> None:
    """Rich rendering of :func:`build_report`: runs, the difference, per-task deltas, the verdict."""
    from rich.console import Console
    from rich.table import Table

    console = Console(highlight=False)
    binary = report["binary"]

    def val(x: float) -> str:
        return f"{x:.1%}" if binary else f"{x:.3f}"

    def delta(x: float) -> str:
        return f"{x * 100:+.1f} pp" if binary else f"{x:+.3f}"

    def ci(lo_hi: list[float]) -> str:
        lo, hi = lo_hi
        return f"[{lo * 100:.1f}, {hi * 100:.1f}]" if binary else f"[{lo:.3f}, {hi:.3f}]"

    metric = report["metric"]
    runs = Table(title=f"{metric} (95% CI)", title_justify="left")
    for col in ("run", "benchmark", metric, "95% CI", "n", "errors"):
        runs.add_column(col, justify="left" if col in ("run", "benchmark") else "right", no_wrap=True)
    for name in ("a", "b"):
        if run := report.get(name):
            console.print(f"{name.upper()}: {run['path']}", soft_wrap=True)
            runs.add_row(
                name.upper(),
                str(run["benchmark"]),
                val(run["mean"]),
                ci(run["ci95"]),
                str(run["n"]),
                str(run["num_errors"]),
            )
    console.print(runs)

    if cmp := report.get("comparison"):
        if cmp["method"] == "paired":
            console.print(
                f"Paired on ({', '.join(cmp['pair_key'])}): {cmp['paired']} pairs; "
                f"unmatched A {cmp['unmatched_a']}, B {cmp['unmatched_b']}"
            )
        else:
            console.print("[yellow]No episodes pair; comparing as independent samples[/yellow]")
        line = f"A − B = [bold]{delta(cmp['difference'])}[/bold], 95% CI {ci(cmp['ci95'])} ({cmp['interval']})"
        console.print(line, soft_wrap=True)
        if mc := cmp.get("mcnemar"):
            console.print(
                f"Exact McNemar: A-only {mc['a_only']}, B-only {mc['b_only']}, p = {mc['p_value']:.3g}",
                soft_wrap=True,
            )
        if len(cmp["tasks"]) > _MAX_TASK_ROWS:
            console.print(f"{len(cmp['tasks'])} tasks; per-task rows are in --json")
        elif cmp["tasks"]:
            table = Table(title="Per task", title_justify="left")
            table.add_column("task", no_wrap=True, overflow="ellipsis", max_width=48)
            for col in ("n", "A", "B", "A − B", "95% CI") + (("p",) if cmp["method"] == "paired" and binary else ()):
                table.add_column(col, justify="right", no_wrap=True)
            for row in cmp["tasks"]:
                n = str(row["n_a"]) if row["n_a"] == row["n_b"] else f"{row['n_a']}/{row['n_b']}"
                cells = [row["task"], n, val(row["mean_a"]), val(row["mean_b"]), delta(row["difference"])]
                cells.append(ci(row["ci95"]))
                if "p_value" in row:
                    cells.append(f"{row['p_value']:.3g}")
                table.add_row(*cells)
            console.print(table)

    if ref := report.get("reference"):
        a = report["a"]
        tol = f"{ref['tolerance'] * 100:g} pp" if binary else f"{ref['tolerance']:g}"
        console.print(
            f"Reference {val(ref['value'])}: A = {val(a['mean'])} {ci(a['ci95'])}, "
            f"Δ {delta(ref['difference'])} → {ref['symbol']} {ref['label']} (tolerance {tol})",
            soft_wrap=True,
        )
