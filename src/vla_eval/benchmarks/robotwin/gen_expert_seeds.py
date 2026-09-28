"""Build the shipped expert-check lists (``expert_seeds/<task_config>.json``); runs inside the RoboTwin image.

    python -m vla_eval.benchmarks.robotwin.gen_expert_seeds check TASK CONFIG FIRST LAST OUT.jsonl
    python -m vla_eval.benchmarks.robotwin.gen_expert_seeds assemble CONFIG N OUT.json PART.jsonl...
    python -m vla_eval.benchmarks.robotwin.gen_expert_seeds verify CONFIG TASK K

``check`` runs the oracle on seeds FIRST..LAST-1 (one line per seed), so a task's seeds can be split over processes;
``assemble`` keeps each task's first N accepted seeds in seed order; ``verify`` re-runs the oracle on a task's first
K listed seeds and reports any it now rejects. Each task keeps the ``fingerprint`` (configs and task code) it was
checked with; ``get_tasks`` uses a list only while it matches.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from vla_eval.benchmarks.robotwin.benchmark import BUNDLED_SEED_BASE, EXPERT_SEEDS_DIR, RoboTwinBenchmark


def _benchmark(task: str, config: str) -> RoboTwinBenchmark:
    bench = RoboTwinBenchmark(task_name=task, task_config=config, seed=0)  # seeds from BUNDLED_SEED_BASE
    bench._init_robotwin()
    return bench


def _provenance() -> dict[str, str]:
    import torch

    out = {"image": os.environ.get("VLA_EVAL_IMAGE", "unknown")}
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    for name in ("sapien", "curobo", "warp"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # provenance is best effort
            out[name] = "unknown"
    return out


def check(task: str, config: str, first: int, last: int, out: str) -> None:
    out = os.path.abspath(out)  # _init_robotwin changes the working directory
    bench = _benchmark(task, config)
    env = bench._create_env()
    with open(out, "w") as f:
        f.write(json.dumps({"provenance": _provenance(), "fingerprint": bench.fingerprint()}) + "\n")
        for seed in range(first, last):
            info = bench._expert_check(env, seed, 0)
            row = {"task": task, "seed": seed, "ok": info is not None}
            if info is not None:
                row.update(bench._instructions(info, seed))
            f.write(json.dumps(row) + "\n")
            f.flush()


def _write(out: Path, provenance: list[dict], tasks: dict[str, dict]) -> None:
    lines = ",\n".join(f"  {json.dumps(t)}: {json.dumps(v, ensure_ascii=False)}" for t, v in tasks.items())
    out.write_text(f'{{"provenance": {json.dumps(provenance)},\n"tasks": {{\n{lines}\n}}}}\n')  # one line per task


def assemble(config: str, n: int, out: str, parts: list[str]) -> None:
    rows, provenance, fingerprints = {}, set(), {}
    for part in parts:
        lines = [json.loads(line) for line in Path(part).read_text().splitlines()]
        provenance.add(json.dumps(lines[0]["provenance"], sort_keys=True))
        for r in lines[1:]:
            rows[(r["task"], r["seed"])] = r
            fingerprints.setdefault(r["task"], set()).add(lines[0].get("fingerprint"))
    tasks: dict[str, dict] = {}
    for task in sorted({t for t, _ in rows}):
        seeds = sorted(s for t, s in rows if t == task)
        if seeds != list(range(BUNDLED_SEED_BASE, BUNDLED_SEED_BASE + len(seeds))):
            raise ValueError(f"{task}: checked seeds must run contiguously from {BUNDLED_SEED_BASE}")
        if len(fingerprints[task]) != 1:
            raise ValueError(f"{task}: parts were checked with different configs or code")
        accepted = [rows[task, s] for s in seeds if rows[task, s]["ok"]][:n]
        if len(accepted) < n:
            raise ValueError(f"{task}: {len(accepted)} accepted seeds, need {n}")
        episodes = [{k: r[k] for k in ("seed", "seen", "unseen")} for r in accepted]
        tasks[task] = {"fingerprint": fingerprints[task].pop(), "episodes": episodes}
    _write(Path(out), [json.loads(p) for p in sorted(provenance)], tasks)


def verify(config: str, task: str, k: int) -> None:
    listed = json.loads((EXPERT_SEEDS_DIR / f"{config}.json").read_text())["tasks"][task]["episodes"][:k]
    bench = _benchmark(task, config)
    env = bench._create_env()
    rejected = [e["seed"] for e in listed if bench._expert_check(env, e["seed"], 0) is None]
    print(json.dumps({"task": task, "config": config, "checked": len(listed), "rejected": rejected, **_provenance()}))


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "check":
        check(a[0], a[1], int(a[2]), int(a[3]), a[4])
    elif cmd == "assemble":
        assemble(a[0], int(a[1]), a[2], a[3:])
    elif cmd == "verify":
        verify(a[0], a[1], int(a[2]))
    else:
        raise SystemExit(__doc__)
