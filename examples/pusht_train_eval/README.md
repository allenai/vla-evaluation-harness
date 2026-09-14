# Push-T: train with LeRobot, evaluate with vla-eval

A minimal training project that calls `vla_eval.evaluate` every N steps on an environment
vla-eval does not ship. Everything specific to the environment lives in `benchmark/`; copy the
directory and replace that folder to do the same for your own simulator.

## Run

```bash
cp -r examples/pusht_train_eval ~/pusht && cd ~/pusht
uv sync                    # Python 3.12; downloads lerobot/pusht on first run
uv run train.py            # 20k steps, eval every 2k; about 1 h on one H100
uv run train.py --docker   # same, but the environment runs in a container
```

`--docker` builds `benchmark/Dockerfile` on first use and runs the environment there, so the
simulator's dependencies stay out of the training environment. `--runtime charliecloud` does
the same without a Docker daemon. Metrics go to `outputs/curve.jsonl`, the final checkpoint to
`outputs/policy/`.

## What you write for your own environment

| Piece | Here | What it does |
|---|---|---|
| Benchmark adapter | `benchmark/pusht.py` | `StepBenchmark` subclass: `get_tasks`, `reset`, `step`, `make_obs`, `get_step_result`, plus action/observation specs |
| Eval config | `benchmark/eval.yaml` | Names the adapter by import string, the episode budget and seed, and `docker.build` for the container path |
| Container image | `benchmark/Dockerfile` | `python:3.12-slim` + `vla-eval` + the simulator + the adapter. Only needed for `--docker` |
| Model server | `PolicyServer` in `train.py` | `PredictModelServer` subclass: observation dict in, action array out, plus the same two specs |

The two specs are compared before the first episode; a mismatch between what the policy emits
and what the environment expects is warned about up front instead of showing up as a 0% run.

## The call in the training loop

```python
results = vla_eval.evaluate(
    PolicyServer(policy, preprocess, postprocess, device),
    "benchmark/eval.yaml",
    docker=args.docker,
    benchmark_overrides={"episodes_per_task": 20},
)
success = results[0]["mean_success"]
```

The policy is served from the training process, so no checkpoint is written or reloaded for an
eval round. See [docs/python-api.md](../../docs/python-api.md) for the full argument list. In a
distributed run, call it on rank 0 only.

## Reference numbers

Diffusion Policy from scratch, one H100, 20 eval episodes per point, eval seeds disjoint from
the training data. With 20 episodes the success rate has roughly a 20 pp noise band.

| Steps | Success | Coverage | Wall clock |
|---:|---:|---:|---:|
| 10k | 10% | 0.59 | 31 min |
| 20k | 20% | 0.66 | 60 min |
| 50k | 40% | 0.77 | 125 min |

LeRobot's published checkpoint (`lerobot/diffusion_pusht`, 200k steps) scores 60% over 50
episodes through the same `PolicyServer`; the model card reports 65%.
