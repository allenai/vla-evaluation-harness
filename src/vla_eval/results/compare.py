"""``vla-eval compare``: two runs' aggregate JSONs, paired by episode, or one run against a reported score.

Inputs are the ``<benchmark>_aggregate.json`` files ``vla-eval run`` / ``vla-eval export`` write: they carry every
episode (``tasks[].episodes[]`` with ``episode_id``, ``metrics`` and, from a recording, the task's fields), so
nothing else is needed.

- **Tasks** match on identity, not position: the task name, plus ``suite`` when both runs record it.  ``task_idx``
  (a position after ``tasks:`` / ``max_tasks`` filtering) only tells apart same-named tasks within a run.  Only
  tasks both runs ran are compared; the rest are reported and left out.  Different ``benchmark`` names are refused
  unless explicitly allowed.
- **Pairing:** episodes pair on ``(task, episode_id)``, i.e. the same task and initial state run twice.  The runs
  are compared as pairs only when pairs cover at least :data:`PAIRED_COVERAGE` of both runs' episodes on the shared
  tasks; a partial overlap is not a representative sample, so otherwise the shared tasks are compared as
  independent samples.
- **Intervals:** paired binary metrics get Newcombe's paired score interval and the exact McNemar test; paired
  non-binary metrics a task-stratified paired bootstrap; independent samples Newcombe's hybrid score interval
  (binary) or a bootstrap.  Whether a metric is binary is decided over both runs.

Values are taken exactly as the aggregate's mean takes them (:func:`vla_eval.results.stats.metric_values`): a
harness error counts as a failure in ``success`` and is absent from every other metric.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from vla_eval.results import stats

PAIRED_COVERAGE = 0.9  # pairs must cover this share of both runs' episodes on the shared tasks to compare as pairs
_MAX_TASK_ROWS = 60  # longer per-task tables (CALVIN: 1000 one-episode sequences) only go to --json

_TaskId = tuple[Any, ...]  # (task name, suite or None, rank among same-named tasks)
_Key = tuple[_TaskId, Any]  # (task identity, episode_id)


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


def _all_episodes(body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(task, episode)`` for every episode, in aggregate order."""
    return [(str(t.get("task")), ep) for t in body["tasks"] for ep in t.get("episodes") or []]


def _episodes(body: dict[str, Any], metric: str) -> list[tuple[str, dict[str, Any], Any]]:
    """``(task, episode, value)`` for every episode with a numeric ``metrics[metric]``, in aggregate order."""
    return [(task, ep, vs[0]) for task, ep in _all_episodes(body) if (vs := stats.metric_values([ep], metric))]


def _mean(values: list[Any]) -> float:
    return sum(float(v) for v in values) / len(values)


def _r(x: float) -> float:
    return round(x, 4)


def _interval(lo: float, hi: float) -> list[float]:
    return [_r(lo), _r(hi)]


def summarize(body: dict[str, Any], metric: str, path: str | Path | None, binary: bool) -> dict[str, Any]:
    """One run's mean, 95% interval (as the aggregate's ``ci95_<metric>``; ``None`` below two non-binary values),
    n, and its harness errors: all of them, and those absent from ``metric``."""
    episodes = [ep for _, ep in _all_episodes(body)]
    values = stats.metric_values(episodes, metric)
    if not values:
        raise ValueError(f"{path or body.get('benchmark')}: no episode has a numeric metric {metric!r}")
    errors = [ep for ep in episodes if ep.get("failure_reason")]
    return {
        "path": str(path) if path is not None else None,
        "benchmark": body.get("benchmark"),
        "mean": _r(_mean(values)),
        "ci95": stats.metric_interval(values, binary),
        "n": len(values),
        "num_episodes": len(episodes),
        "num_errors": len(errors),
        "errors_without_metric": sum(1 for ep in errors if not stats.metric_values([ep], metric)),
        "partial": bool(body.get("partial")),
    }


def _identify(rows: list[tuple[str, dict[str, Any], Any]], use_suite: bool) -> list[tuple[_TaskId, Any, Any]]:
    """``(task identity, episode_id, value)`` per row.  Same-named tasks within the run (merged under one name in
    the aggregate) are told apart by the rank of their ``task_idx``."""

    def base(task: str, ep: dict[str, Any]) -> tuple[Any, ...]:
        return (task, ep.get("suite") if use_suite else None)

    indices: dict[tuple[Any, ...], set[int]] = {}
    for task, ep, _ in rows:
        if isinstance(idx := ep.get("task_idx"), int):
            indices.setdefault(base(task, ep), set()).add(idx)
    rank = {b: {idx: k for k, idx in enumerate(sorted(s))} for b, s in indices.items()}
    return [
        (base(task, ep) + (rank.get(base(task, ep), {}).get(ep.get("task_idx"), 0),), ep.get("episode_id"), v)
        for task, ep, v in rows
    ]


def _keyed(rows: list[tuple[_TaskId, Any, Any]]) -> tuple[dict[_Key, Any], int]:
    """Map ``(task identity, episode_id)`` → value.  Episodes sharing a key are ambiguous: dropped, and their count
    returned."""
    groups: dict[_Key, list[Any]] = {}
    for tid, eid, value in rows:
        groups.setdefault((tid, eid), []).append(value)
    keyed = {k: g[0] for k, g in groups.items() if len(g) == 1}
    return keyed, sum(len(g) for g in groups.values() if len(g) > 1)


def _labels(tids: set[_TaskId]) -> dict[_TaskId, str]:
    """Display name per task identity: the task name, with the suite when several are compared and ``#k`` for
    same-named tasks."""
    suites = {t[1] for t in tids}
    repeated = {t[:2] for t in tids if t[2] > 0}
    return {
        t: (f"{t[1]}/" if len(suites) > 1 else "") + str(t[0]) + (f" #{t[2] + 1}" if t[:2] in repeated else "")
        for t in tids
    }


def _paired_difference(
    a: list[Any], b: list[Any], *, binary: bool, strata: list[Any] | None = None
) -> tuple[float, list[float] | None, str]:
    """``(estimate, [lo, hi], method)`` of ``mean(a) - mean(b)`` over pairs ``(a[i], b[i])``; no interval from a
    bootstrap of fewer than two pairs."""
    estimate = _mean(a) - _mean(b)
    if binary:
        t = Counter((bool(x), bool(y)) for x, y in zip(a, b))
        lo, hi = stats.newcombe_paired_interval(t[True, True], t[True, False], t[False, True], t[False, False])
        return _r(estimate), _interval(lo, hi), "Newcombe paired score"
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    if len(diffs) < 2:
        return _r(estimate), None, "paired bootstrap"
    # A one-pair stratum cannot vary under resampling, so stratify only when every task has two or more pairs
    # (CALVIN's 1000 one-episode sequences are resampled as one pool).
    stratify = strata is not None and min(Counter(strata).values()) >= 2
    lo, hi = stats.paired_bootstrap_difference(diffs, strata if stratify else None)
    return _r(estimate), _interval(lo, hi), "paired bootstrap" + (", stratified by task" if stratify else "")


def _independent_difference(a: list[Any], b: list[Any], *, binary: bool) -> tuple[float, list[float] | None, str]:
    """``(estimate, [lo, hi], method)`` of ``mean(a) - mean(b)`` from independent samples; no interval from a
    bootstrap with fewer than two values on a side."""
    estimate = _mean(a) - _mean(b)
    if binary:
        xa, xb = sum(bool(v) for v in a), sum(bool(v) for v in b)
        lo, hi = stats.newcombe_difference_interval(xa, len(a), xb, len(b))
        return _r(estimate), _interval(lo, hi), "Newcombe hybrid score, independent"
    if min(len(a), len(b)) < 2:
        return _r(estimate), None, "bootstrap, independent"
    lo, hi = stats.independent_bootstrap_difference([float(v) for v in a], [float(v) for v in b])
    return _r(estimate), _interval(lo, hi), "bootstrap, independent"


def _mcnemar(a: list[Any], b: list[Any]) -> dict[str, Any]:
    a_only = sum(1 for x, y in zip(a, b) if bool(x) and not bool(y))
    b_only = sum(1 for x, y in zip(a, b) if bool(y) and not bool(x))
    return {"a_only": a_only, "b_only": b_only, "p_value": stats.mcnemar_exact(a_only, b_only)}


def _seed(body: dict[str, Any]) -> Any:
    params = (body.get("config") or {}).get("params")
    return body.get("seed", params.get("seed") if isinstance(params, dict) else None)


def condition_warnings(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Evaluation-condition differences visible in the two aggregates: seed, benchmark params, ``max_steps``,
    mode."""
    out = []
    seed_a, seed_b = _seed(a), _seed(b)
    if seed_a is not None and seed_b is not None and seed_a != seed_b:
        out.append(f"seeds differ (A {seed_a}, B {seed_b}): paired episodes may not start from the same state")
    ca, cb = a.get("config") or {}, b.get("config") or {}
    pa, pb = ca.get("params"), cb.get("params")
    if isinstance(pa, dict) and isinstance(pb, dict):
        keys = sorted(k for k in pa.keys() | pb.keys() if k != "seed" and pa.get(k) != pb.get(k))
        if keys:
            out.append(
                "benchmark params differ: " + ", ".join(f"{k} (A {pa.get(k)!r}, B {pb.get(k)!r})" for k in keys)
            )
    if "max_steps" in ca and "max_steps" in cb and ca["max_steps"] != cb["max_steps"]:
        out.append(f"max_steps differs (A {ca['max_steps']}, B {cb['max_steps']})")
    if a.get("mode") and b.get("mode") and a["mode"] != b["mode"]:
        out.append(f"modes differ (A {a['mode']}, B {b['mode']})")
    return out


def compare_runs(
    a: dict[str, Any], b: dict[str, Any], metric: str = "success", *, allow_benchmark_mismatch: bool = False
) -> dict[str, Any]:
    """Compare run A against run B on ``metric`` (difference is A − B) over the tasks both ran, overall and per
    task.  ``warnings`` lists what the reader must know: left-out tasks and episodes, condition differences."""
    if a.get("benchmark") != b.get("benchmark") and not allow_benchmark_mismatch:
        raise ValueError(
            f"A is benchmark {a.get('benchmark')!r}, B is {b.get('benchmark')!r}; "
            "pass --allow-benchmark-mismatch to compare them anyway"
        )
    rows_a, rows_b = _episodes(a, metric), _episodes(b, metric)
    if not rows_a or not rows_b:
        raise ValueError(f"no episode has a numeric metric {metric!r}")
    binary = stats.is_binary(v for _, _, v in rows_a + rows_b)
    use_suite = all("suite" in ep for _, ep, _ in rows_a + rows_b)
    ida, idb = _identify(rows_a, use_suite), _identify(rows_b, use_suite)
    tasks_a, tasks_b = {t for t, _, _ in ida}, {t for t, _, _ in idb}
    shared = tasks_a & tasks_b
    if not shared:
        raise ValueError(f"A and B share no task (matched on task name{' and suite' if use_suite else ''})")
    warnings = condition_warnings(a, b)
    if a.get("benchmark") != b.get("benchmark"):
        warnings.insert(0, f"benchmarks differ (A {a.get('benchmark')}, B {b.get('benchmark')})")
    sa, sb = [r for r in ida if r[0] in shared], [r for r in idb if r[0] in shared]
    if tasks_a - shared or tasks_b - shared:
        warnings.append(
            f"tasks run by only one side are left out: A {len(tasks_a - shared)} ({len(ida) - len(sa)} episodes), "
            f"B {len(tasks_b - shared)} ({len(idb) - len(sb)} episodes)"
        )
    keyed_a, dupes_a = _keyed(sa)
    keyed_b, dupes_b = _keyed(sb)
    common = [k for k in keyed_a if k in keyed_b]
    paired = bool(common) and len(common) >= PAIRED_COVERAGE * max(len(sa), len(sb))
    out: dict[str, Any] = {
        "pair_key": ["task", "suite", "episode_id"] if use_suite else ["task", "episode_id"],
        "tasks_compared": len(shared),
        "tasks_only_a": len(tasks_a - shared),
        "tasks_only_b": len(tasks_b - shared),
        "paired": len(common),
        "unmatched_a": len(sa) - len(common),
        "unmatched_b": len(sb) - len(common),
        "method": "paired" if paired else "independent",
    }
    if dupes_a or dupes_b:
        out["duplicate_keys"] = [dupes_a, dupes_b]
        warnings.append(f"episodes sharing a (task, episode_id) key are not paired: A {dupes_a}, B {dupes_b}")
    if paired and (out["unmatched_a"] or out["unmatched_b"]):
        warnings.append(f"unpaired episodes are left out: A {out['unmatched_a']}, B {out['unmatched_b']}")
    elif common and not paired:
        warnings.append(
            f"only {len(common)} pairs for {len(sa)} A / {len(sb)} B episodes on the shared tasks "
            f"(< {PAIRED_COVERAGE:.0%}): compared as independent samples"
        )

    labels = _labels(shared)
    by_task: dict[_TaskId, tuple[list[Any], list[Any]]] = {}
    if paired:
        for k in common:
            ta, tb = by_task.setdefault(k[0], ([], []))
            ta.append(keyed_a[k])
            tb.append(keyed_b[k])
        va, vb = [keyed_a[k] for k in common], [keyed_b[k] for k in common]
        out["difference"], out["ci95"], out["interval"] = _paired_difference(
            va, vb, binary=binary, strata=[labels[k[0]] for k in common]
        )
        if binary:
            out["mcnemar"] = _mcnemar(va, vb)
    else:
        for tid, _, v in sa:
            by_task.setdefault(tid, ([], []))[0].append(v)
        for tid, _, v in sb:
            by_task[tid][1].append(v)
        va, vb = [v for _, _, v in sa], [v for _, _, v in sb]
        out["difference"], out["ci95"], out["interval"] = _independent_difference(va, vb, binary=binary)
    out["n_a"], out["n_b"] = len(va), len(vb)
    out["mean_a"], out["mean_b"] = _r(_mean(va)), _r(_mean(vb))

    task_rows = []
    for tid in sorted(by_task, key=lambda t: labels[t]):
        ta, tb = by_task[tid]
        if paired:
            diff, ci, _ = _paired_difference(ta, tb, binary=binary)
        else:
            diff, ci, _ = _independent_difference(ta, tb, binary=binary)
        row: dict[str, Any] = {
            "task": labels[tid],
            "n_a": len(ta),
            "n_b": len(tb),
            "mean_a": _r(_mean(ta)),
            "mean_b": _r(_mean(tb)),
            "difference": diff,
            "ci95": ci,
        }
        if paired and binary:
            row["p_value"] = _mcnemar(ta, tb)["p_value"]
        task_rows.append(row)
    out["tasks"] = task_rows
    out["warnings"] = warnings
    return out


def reference_verdict(run: dict[str, Any], reference: float, tolerance: float) -> dict[str, Any]:
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
    tolerance: float | None = None,
    allow_benchmark_mismatch: bool = False,
) -> dict[str, Any]:
    """Everything ``vla-eval compare`` prints (and emits with ``--json``).  ``tolerance`` defaults to 5 pp for a
    rate; a non-rate metric's verdict needs one on its own scale."""
    runs = {"a": (a_path, load_aggregate(a_path))}
    if b_path is not None:
        runs["b"] = (b_path, load_aggregate(b_path))
    values = [v for _, body in runs.values() for _, _, v in _episodes(body, metric)]
    binary = stats.is_binary(values)  # one decision for both runs, so A, B and A − B share a scale
    report: dict[str, Any] = {"metric": metric, "binary": binary, "warnings": []}
    for name, (path, body) in runs.items():
        run = report[name] = summarize(body, metric, path, binary)
        if run["partial"]:
            report["warnings"].append(f"{name.upper()} is partial: some work items never finished")
        if run["errors_without_metric"]:
            report["warnings"].append(
                f"{name.upper()}: {run['errors_without_metric']} harness error(s) record no {metric!r} and are "
                "left out of it (an error counts only as a success failure)"
            )
    if b_path is not None:
        report["comparison"] = compare_runs(
            runs["a"][1], runs["b"][1], metric, allow_benchmark_mismatch=allow_benchmark_mismatch
        )
    if reference is not None:
        if tolerance is None:
            if not binary:
                raise ValueError(f"{metric!r} is not a rate: give --tolerance on its own scale for the verdict")
            tolerance = stats.TOLERANCE
        if report["a"]["ci95"] is None:
            raise ValueError(f"A has fewer than two values of {metric!r}: no interval to judge a reference by")
        report["reference"] = reference_verdict(report["a"], parse_reference(reference, binary), tolerance)
    return report


def print_report(report: dict[str, Any]) -> None:
    """Rich rendering of :func:`build_report`: runs, warnings, the difference, per-task deltas, the verdict."""
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console(highlight=False)
    binary = report["binary"]

    def val(x: float) -> str:
        return f"{x:.1%}" if binary else f"{x:.3f}"

    def delta(x: float, unit: bool = True) -> str:
        return (f"{x * 100:+.1f}" + (" pp" if unit else "")) if binary else f"{x:+.3f}"

    def ci(lo_hi: list[float] | None) -> str:
        if lo_hi is None:
            return "n/a"
        lo, hi = lo_hi
        if lo == hi:  # a bootstrap of identical values: no information about the spread
            return "n/a (no variation)"
        return f"[{lo * 100:.1f}, {hi * 100:.1f}]" if binary else f"[{lo:.3f}, {hi:.3f}]"

    def table(title: str, columns: list[str], rows: list[list[str]], first_flex: bool = False) -> Table:
        """Numbers keep their full width; with ``first_flex`` the first (task) column takes what is left (8-40
        characters) and ellipsizes.  Rich alone would squeeze the numeric columns to nothing below ~130 columns."""
        widths = [max(len(c) for c in [col] + [r[i] for r in rows]) for i, col in enumerate(columns)]
        if first_flex:
            border = 3 * len(columns) + 1  # a rule and two padding spaces per column, plus the closing rule
            widths[0] = max(8, min(40, widths[0], console.width - border - sum(widths[1:])))
        t = Table(title=title, title_justify="left")
        for i, col in enumerate(columns):
            t.add_column(
                col, justify="left" if i == 0 else "right", no_wrap=True, overflow="ellipsis", width=widths[i]
            )
        for r in rows:
            t.add_row(*map(escape, r))
        return t

    metric = report["metric"]
    names = [n for n in ("a", "b") if n in report]
    for name in names:
        run = report[name]
        console.print(escape(f"{name.upper()}: {run['path']} ({run['benchmark']})"), soft_wrap=True)
    rows = [
        [n.upper(), val(r["mean"]), ci(r["ci95"]), str(r["n"]), str(r["num_errors"])]
        for n in names
        if (r := report[n])
    ]
    console.print(table(f"{metric} (95% CI)", ["run", metric, "95% CI", "n", "errors"], rows))

    cmp = report.get("comparison")
    for w in report["warnings"] + (cmp["warnings"] if cmp else []):
        console.print(f"[yellow]⚠ {escape(w)}[/yellow]", soft_wrap=True)

    if cmp:
        if cmp["method"] == "paired":
            head = (
                f"Paired on ({', '.join(cmp['pair_key'])}): {cmp['paired']} pairs over {cmp['tasks_compared']} tasks"
            )
        else:
            head = (
                f"Independent samples over {cmp['tasks_compared']} shared tasks: "
                f"A n={cmp['n_a']}, B n={cmp['n_b']} ({cmp['paired']} episodes pair)"
            )
        console.print(head, soft_wrap=True)
        means = ""
        if (cmp["n_a"], cmp["n_b"]) != (report["a"]["n"], report["b"]["n"]):  # compared subset ≠ whole runs
            means = f" ({val(cmp['mean_a'])} vs {val(cmp['mean_b'])})"
        line = f"A − B = [bold]{delta(cmp['difference'])}[/bold]{means}, 95% CI {ci(cmp['ci95'])} ({cmp['interval']})"
        console.print(line, soft_wrap=True)
        if mc := cmp.get("mcnemar"):
            console.print(
                f"Exact McNemar: A-only {mc['a_only']}, B-only {mc['b_only']}, p = {mc['p_value']:.3g}",
                soft_wrap=True,
            )
        if len(cmp["tasks"]) > _MAX_TASK_ROWS:
            console.print(f"{len(cmp['tasks'])} tasks; per-task rows are in --json")
        elif len(cmp["tasks"]) > 1:
            with_p = any("p_value" in row for row in cmp["tasks"])
            columns = ["task", "n", "A", "B", "A − B", "95% CI"] + (["p"] * with_p)
            # LIBERO-style names share long prefixes ("pick up the black bowl ..."): show it once, in the title,
            # so a narrow task column still tells the rows apart.
            prefix = os.path.commonprefix([row["task"] for row in cmp["tasks"]])
            prefix = prefix[: prefix.rfind(" ") + 1] if len(prefix) >= 12 else ""
            task_rows = []
            for row in cmp["tasks"]:
                n = str(row["n_a"]) if row["n_a"] == row["n_b"] else f"{row['n_a']}/{row['n_b']}"
                name = "…" + row["task"][len(prefix) :] if prefix else row["task"]
                cells = [name, n, val(row["mean_a"]), val(row["mean_b"]), delta(row["difference"], False)]
                cells.append(ci(row["ci95"]))
                if with_p:
                    cells.append(f"{row['p_value']:.3g}")
                task_rows.append(cells)
            title = "Per task" + (", A − B in pp" if binary else "") + (f" (tasks: {prefix}…)" if prefix else "")
            console.print(table(title, columns, task_rows, first_flex=True))
            if with_p:
                console.print("p: exact McNemar per task, not adjusted for multiple comparisons", style="dim")

    if ref := report.get("reference"):
        a = report["a"]
        tol = f"{ref['tolerance'] * 100:g} pp" if binary else f"{ref['tolerance']:g}"
        console.print(
            f"Reference {val(ref['value'])}: A = {val(a['mean'])} {ci(a['ci95'])}, "
            f"Δ {delta(ref['difference'])} → {ref['symbol']} {ref['label']} (tolerance {tol})",
            soft_wrap=True,
        )
