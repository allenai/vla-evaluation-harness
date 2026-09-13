# Push-T: your own benchmark, evaluated by vla-eval while training

Push-T is not one of vla-eval's benchmarks. This directory shows what a project does to use
vla-eval on an environment of its own: put an adapter, a YAML naming it, and a Dockerfile in
`benchmark/`, then call `evaluate` from the training loop. Copy the directory anywhere; it depends on `vla-eval` as a package
(pinned to a git commit until 0.6.0, the first release with `vla_eval.evaluate`, is on PyPI).

```bash
cp -r examples/pusht_train_eval ~/my-pusht && cd ~/my-pusht
uv sync
uv run train.py                       # adapter runs in this process; 20k steps, ~1 h on one H100
uv run train.py --docker              # adapter runs in the image benchmark/eval.yaml builds (or --runtime charliecloud)
```

Both paths use the same YAML and the same adapter; the flag only decides where the adapter runs. Use the image when the env's dependencies should stay out of the training env.

| File | Role |
|---|---|
| `benchmark/pusht.py` | The adapter: a `StepBenchmark` subclass over `gym-pusht` (reset, step, make_obs, result, specs). 123 lines |
| `benchmark/eval.yaml` | Names the adapter by import string (`benchmark.pusht:PushTBenchmark`), the episode budget, and the image for `--docker` with `build: .` |
| `benchmark/Dockerfile` | `python:3.12-slim` + `vla-eval` + `gym-pusht` + the adapter; built on first `--docker` run |
| `train.py` | LeRobot's Push-T Diffusion Policy training example plus the `evaluate` call. 150 lines |
| `pyproject.toml` | `vla-eval`, `lerobot`, `gym-pusht` |

## The eval call

```python
results = vla_eval.evaluate(
    PolicyServer(policy, preprocess, postprocess, device),   # wraps the live model
    "benchmark/eval.yaml",
    docker=False, no_save=True,
    benchmark_overrides={"episodes_per_task": 20},
)
results[0]["mean_success"]
```

`evaluate` starts a model server on a background thread of this process (the policy is served
from the training GPU, no checkpoint round trip), imports the adapter named in the YAML, runs
the episodes, and returns one result dict per benchmark entry. `PolicyServer` is the other
piece you write: observation dict in, action array out, plus the two spec declarations the
harness checks against the adapter before running.

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

LeRobot trains its published checkpoint for 200k steps and reports 65%. Serving that checkpoint
(`lerobot/diffusion_pusht`) through the same `PolicyServer` gives 60% over 50 episodes, which is
how the serving path was validated.

## Notes

- Results go to `outputs/curve.jsonl`; log them to your own tracker from there.
- In a distributed run, call `evaluate` on rank 0 only and barrier afterwards.
- Do not pass `watchdog_timeout_s`: the stall watchdog `os._exit`s the whole process.
