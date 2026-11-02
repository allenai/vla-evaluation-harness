"""Analyze one vla-eval run (arm C or L) and print rtc.md's tables from this directory's data.

usage: python report.py analyze <recording.sqlite> <d> <s> [--live] [-o run.json]   # -> sync/ or live/<method>_d<d>.json
       python report.py tables                                                      # every table in rtc.md
Criteria: (1) the level-average diff lies within its 95% CI; (2) <= 2 of 12 levels outside their 95% CI (Newcombe)
and, for 2048-rollout arms, none off by more than 5 pp.
"""

import argparse
import collections
import json
import math
import pathlib
import sqlite3

import numpy as np

HERE = pathlib.Path(__file__).parent
Z = 1.959964
LEVELS = "grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle \
chain_lander catcher_v3 trampoline car_launch".split()
# Fig. 5 averages read from the vector PDF (±0.003): d sweep (s = max(d, 1)) and d = 1 s sweep (s = 1..7)
FIG5_D = {
    "rtc": [0.922, 0.893, 0.843, 0.747, 0.618],
    "rtc_hard": [0.916, 0.849, 0.817, 0.732, 0.619],
    "bid": [0.921, 0.877, 0.801, 0.673, 0.552],
    "naive": [0.915, 0.753, 0.708, 0.584, 0.493],
}
FIG5_S = {
    "rtc": [0.893, 0.878, 0.863, 0.845, 0.834, 0.821, 0.796],
    "rtc_hard": [0.848, 0.866, 0.856, 0.850, 0.830, 0.821, 0.796],
    "naive": [0.753, 0.812, 0.787, 0.784, 0.753, 0.754, 0.716],
}


def analyze(db: str, d: int, s: int, live: bool, hz: float) -> dict:
    """Per-level success and protocol instruments from one recording DB."""
    c = sqlite3.connect(db)
    lvl = collections.defaultdict(lambda: [0, 0])
    errors: collections.Counter = collections.Counter()
    steps_by_ep, rt = {}, collections.defaultdict(list)
    for sid, eid, ctx, status, metrics, steps in c.execute(
        "SELECT sid, eid, context, status, metrics, steps FROM episode_results"
    ):
        ctx = json.loads(ctx)
        if status == "error":
            errors[ctx["level"]] += 1
            continue
        lvl[ctx["level"]][0] += bool(json.loads(metrics).get("success"))
        lvl[ctx["level"]][1] += 1
        steps_by_ep[(sid, eid)] = steps
        for k, v in (ctx.get("rt_metrics") or {}).items():
            rt[k].append(v)

    inst: collections.Counter = collections.Counter()
    eff: collections.Counter = collections.Counter()  # effective delay - d, per boundary
    lag, margin_lo, margin_hi, infer_ms = [], [], [], []
    rows = collections.defaultdict(dict)
    for sid, eid, step, fields in c.execute("SELECT sid, eid, step_id, fields FROM step_rows"):
        rows[(sid, eid)][step] = json.loads(fields)
    for key, st in rows.items():
        n = steps_by_ep.get(key)
        if n is None:
            continue
        first = {}  # chunk k -> (first protocol step it was dispensed, its harness step)
        adm = {r["p"]: r["admitted_wall"] for _, r in sorted(st.items()) if r.get("p", -1) >= 0}  # latest admission
        infer_steps = {r["p"] if "p" in r else h for h, r in st.items() if "infer_k" in r}
        last_j = -1
        for step in range(n):
            r = st.get(step) or {}
            if r.get("pre_start"):
                inst["pre_start_ticks"] += 1
                continue
            if "action" not in r:
                inst["missing_bench_row"] += 1
                continue
            if "intended_action" not in r:
                if r.get("p", 0) >= 0:
                    inst["missing_server_row"] += 1
                continue
            j = r.get("p", step)  # protocol step (the server's), else the harness index
            if j < 0:
                continue
            inst["steps"] += 1
            last_j = max(last_j, j)
            k = r.get("chunk_k")
            if k is not None and k not in first:
                first[k] = (j, step)
            if not np.allclose(r["action"], r["intended_action"], atol=1e-6):
                inst["stale"] += 1
            elif "dispense_wall" in r and "t_wall" in r:
                lag.append(r["t_wall"] - r["dispense_wall"])
            if "infer_ms" in r:
                infer_ms.append(r["infer_ms"])
        inst["cadence_violation"] += len(infer_steps ^ {j for j in range(last_j + 1) if j % s == 0})
        for k, (j, h) in first.items():
            if k < 0:
                continue
            t = k * s
            eff[j - t - d] += 1
            lw = st[h].get("landed_wall")
            if live and lw is not None and d > 0:  # landed between the admissions of obs t+d-1 and obs t+d
                if t + d - 1 in adm:
                    margin_lo.append(lw - adm[t + d - 1])
                margin_hi.append(st[h]["admitted_wall"] - lw)
        expected = {k for k in range(last_j // s + 1) if k * s + d <= last_j}
        inst["boundary_never_dispensed"] += len(expected - {k for k in first if k >= 0})

    out: dict = {"levels": {}, "errors": dict(errors)}
    for name in sorted(lvl):
        k, m = lvl[name]
        out["levels"][name] = {"solved": k, "n": m, "p": k / m}
    out["avg"] = float(np.mean([v["p"] for v in out["levels"].values()])) if lvl else None
    out["n_total"] = sum(m for _, m in lvl.values())
    out["instruments"] = dict(inst)
    nb = sum(eff.values())
    out["eff_delay_hist"] = {str(k): v for k, v in sorted(eff.items())}
    out["eff_delay_exact_frac"] = eff[0] / nb if nb else None
    if inst["steps"]:
        out["stale_frac"] = inst["stale"] / inst["steps"]

    def pct(xs: list, scale: float = 1.0) -> dict:
        return {q: float(np.percentile(xs, x) * scale) for q, x in (("p50", 50), ("p99", 99), ("max", 100))}

    if lag:
        out["dispense_to_apply_ms"] = pct(lag, 1e3)
    if infer_ms:
        out["infer_ms"] = pct(infer_ms)
    if margin_lo:
        out["land_after_prev_obs_ms_min"] = float(np.min(margin_lo) * 1e3)
    if margin_hi:
        out["land_before_obs_ms_min"] = float(np.min(margin_hi) * 1e3)
    if rt.get("step_time_max"):
        out["eps_step_over_half_period"] = int(sum(x > 0.5 / hz for x in rt["step_time_max"]))
    for k in ("step_time_max", "stale_action_ratio"):
        if rt.get(k):
            out[f"rt_{k}_max"] = float(np.max(rt[k]))
            out[f"rt_{k}_mean"] = float(np.mean(rt[k]))
    return out


def wilson(p: float, n: int) -> tuple[float, float]:
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return c - h, c + h


def compare(ref: dict, arm: dict) -> dict:
    """``ref``/``arm``: level -> (p, n). Arm minus ref, per level (Newcombe) and on the 12-level average."""
    var, outside, worst = 0.0, [], 0.0
    for lv in LEVELS:
        (pr, nr), (px, nx) = ref[lv], arm[lv]
        diff = px - pr
        (lx, ux), (lr, ur) = wilson(px, nx), wilson(pr, nr)
        if not diff - math.hypot(px - lx, ur - pr) <= 0 <= diff + math.hypot(ux - px, pr - lr):
            outside.append(f"{lv} {diff * 100:+.1f}")
        worst = max(worst, abs(diff))
        var += pr * (1 - pr) / nr + px * (1 - px) / nx
    diff, ci, n = avg(arm) - avg(ref), Z * math.sqrt(var) / 12, min(arm[lv][1] for lv in LEVELS)
    c2 = len(outside) <= 2 and (worst <= 0.05 or n < 1024)  # the 5 pp cap is for the 2048-rollout arms
    return dict(n=n, diff=diff, ci=ci, outside=outside, worst=worst, c1=abs(diff) <= ci, c2=c2)


def load_ref(name: str) -> dict:
    """reference/<name>.jsonl -> (method, d, s) -> level -> (p, n)."""
    out: dict = collections.defaultdict(dict)
    for r in map(json.loads, open(HERE / "reference" / f"{name}.jsonl")):
        out[(r["method"], r["d"], r["s"])][r["level"]] = (r["solved"], r["n"])
    return out


def load_arm(arm: str, method: str, d: int) -> dict:
    """sync/ or live/<method>_d<d>.json -> level -> (p, n)."""
    return {k: (v["p"], v["n"]) for k, v in json.load(open(HERE / arm / f"{method}_d{d}.json"))["levels"].items()}


def avg(x: dict) -> float:
    return sum(x[lv][0] for lv in LEVELS) / 12


def table(title: str, head: str, rows: list[str]) -> None:
    print(f"{title}\n\n| {head} |\n|" + "---|" * (head.count("|") + 1))
    print("\n".join(f"| {r} |" for r in rows) + "\n")


def tables() -> None:
    r = {**load_ref("R_core"), **load_ref("R_bid"), **load_ref("R_horizon")}
    rv = {**load_ref("R_core_v3metric"), **load_ref("R_bid_v3metric")}
    ds = " | ".join(f"d={d}" for d in range(5))
    rows = [
        " | ".join([m, *(f"{avg(r[m, d, max(d, 1)]):.3f} / {f:.3f}" for d, f in enumerate(FIG5_D[m]))]) for m in FIG5_D
    ]
    table("R vs Fig. 5, d sweep (R / Fig. 5)", f"method | {ds}", rows)
    rows = [" | ".join([m, *(f"{avg(r[m, 1, s]):.3f} / {f:.3f}" for s, f in enumerate(FIG5_S[m], 1))]) for m in FIG5_S]
    table("R vs Fig. 5, d = 1 s sweep (R / Fig. 5)", "method | " + " | ".join(f"s={s}" for s in range(1, 8)), rows)
    rows = [
        " | ".join([m, *(f"{avg(rv[m, d, max(d, 1)]):.3f} / {avg(r[m, d, max(d, 1)]):.3f}" for d in range(5))])
        for m in FIG5_D
    ]
    table("R′ (v3 metric), d sweep (R′ / R)", f"method | {ds}", rows)

    sync = [(m, d) for m in ("naive", "rtc", "rtc_hard") for d in range(5)]
    live = sync[:10]
    head = "condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2)"

    def get(arm: str, m: str, d: int) -> dict:
        return rv[m, d, max(d, 1)] if arm == "R′" else load_arm(arm, m, d)

    ok = {True: "pass", False: "FAIL"}
    for title, conds, ref, arm in (
        ("C vs R′", sync, "R′", "sync"),
        ("L vs C", live, "sync", "live"),
        ("L vs R′", live, "R′", "live"),
    ):
        rows = []
        for m, d in conds:
            a, b = get(ref, m, d), get(arm, m, d)
            c = compare(a, b)
            rows.append(
                f"{m} d={d} | {c['n']} | {avg(a):.3f} | {avg(b):.3f} | {c['diff'] * 100:+.2f} ± {c['ci'] * 100:.2f} | "
                f"{', '.join(c['outside']) or '—'} | {c['worst'] * 100:.1f} | {ok[c['c1']]} | {ok[c['c2']]}"
            )
        table(title, head, rows)

    rows = []
    for m, d in live:
        j = json.load(open(HERE / "live" / f"{m}_d{d}.json"))
        off = ", ".join(f"{k}: {v}" for k, v in j["eff_delay_hist"].items() if k != "0") or "—"
        ins = j["instruments"]
        rows.append(
            f"{m} d={d} | {j['eff_delay_exact_frac'] * 100:.3f}% ({off}) | {ins.get('stale', 0)} "
            f"({j['stale_frac'] * 100:.3f}%) | {ins['pre_start_ticks'] / j['n_total']:.2f} | "
            f"{ins['cadence_violation']} | {j['rt_step_time_max_max'] * 1e3:.0f} | {j['eps_step_over_half_period']}"
        )
    head = "condition | delay = d (off by: count) | stale ticks | pre-start/ep | cadence misses | worst step ms | eps > Δt/2"
    table("Live instruments", head, rows)

    rows = []
    for m in ("naive", "rtc", "rtc_hard"):
        cells = []
        for d in range(5):
            arms = ["R′", "sync", "live"] if (m, d) in live else ["R′", "sync"]
            cells.append(" / ".join(f"{get(x, m, d)['trampoline'][0]:.3f}" for x in arms))
        rows.append(" | ".join([m, *cells]))
    table("trampoline (R′ / C / L)", f"method | {ds}", rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("db")
    a.add_argument("d", type=int)
    a.add_argument("s", type=int)
    a.add_argument("--live", action="store_true")
    a.add_argument("--hz", type=float, default=10.0)
    a.add_argument("-o", "--out")
    sub.add_parser("tables")
    args = p.parse_args()
    if args.cmd == "tables":
        tables()
    else:
        res = analyze(args.db, args.d, args.s, args.live, args.hz)
        print(json.dumps(res, indent=1))
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1)
