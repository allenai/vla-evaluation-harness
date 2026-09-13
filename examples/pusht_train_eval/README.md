# Push-T: evaluate while training

A self-contained project that trains LeRobot's Diffusion Policy on Push-T and calls vla-eval
between optimizer steps. Copy this directory into your own repo; it depends on `vla-eval` as
a package, not on this checkout.

```bash
cp -r examples/pusht_train_eval ~/my-pusht && cd ~/my-pusht
# delete the [tool.uv.sources] block in pyproject.toml (it points at the vla-eval checkout)
uv sync
uv run train.py                       # 20k steps, eval every 2k, ~1 h on one H100
uv run train.py --steps 50000         # ~2 h
uv run train.py --docker              # benchmark in its pinned image instead of in-process
```

Three files: `pyproject.toml` (deps), `eval.yaml` (the benchmark to run; the Push-T adapter
ships with vla-eval), `train.py` (LeRobot's Push-T training example plus the eval call).

## What the eval call does

```python
results = vla_eval.evaluate(
    PolicyServer(policy, preprocess, postprocess, device),   # wraps the live model
    "eval.yaml",
    docker=False,                                             # in-process
    no_save=True,                                             # in-memory results
    benchmark_overrides={"episodes_per_task": 20},
)
results[0]["mean_success"]
```

`evaluate` starts a WebSocket model server on a background thread of this process, so the
policy is served from the training GPU without a checkpoint round trip, then runs the
benchmark against it and returns one result dict per benchmark entry (the shape
`vla-eval merge` writes). `PolicyServer` is the only glue you write: observation dict in,
action array out, plus the two spec declarations the harness checks before running.

## Measured

One H100, 20 eval episodes per point (about +-20 pp of binomial noise), eval seeds disjoint
from the training episodes.

| Steps | Success | Mean coverage | Wall clock |
|---:|---:|---:|---:|
| 2k | 0% | 0.12 | 6 min |
| 10k | 10% | 0.59 | 31 min |
| 16k | 35% | 0.64 | 48 min |
| 20k | 20% | 0.66 | 60 min |
| 50k | 40% | 0.77 | 125 min |

LeRobot trains its published checkpoint for 200k steps and reports 65% success. Serving that
checkpoint (`lerobot/diffusion_pusht`) through the same `PolicyServer` gives 60% over 50
episodes, which is how the serving path was validated.

## Notes

- Results go to `outputs/curve.jsonl`; log them to your own tracker from there. vla-eval
  creates no tracking run of its own.
- In a distributed run, call `evaluate` on rank 0 only and barrier afterwards.
- Do not pass `watchdog_timeout_s`: the stall watchdog `os._exit`s the whole process.
