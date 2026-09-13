---
smoke_config: eval.yaml
---

# Push-T

2-D block pushing from Diffusion Policy, via `gym-pusht`. CPU only (pymunk + pygame), ~3 ms per step.
[Paper](https://arxiv.org/abs/2303.04137) | [GitHub](https://github.com/huggingface/gym-pusht)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/pusht:latest`

No GPU or Docker is required: `uv sync --extra pusht` then `vla-eval run --no-docker -c configs/benchmarks/pusht/eval.yaml`.
This is the benchmark used by [`examples/pusht_train_eval`](../../../examples/pusht_train_eval/) for training-time evaluation.

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Standard evaluation, seeds 0..49 | 1 | 50 |
| `realtime.yaml` | Same, in live (wall-clock) mode at 10 Hz | 1 | 50 |

## Interface

| | |
|---|---|
| `images["agentview"]` | 96x96x3 uint8 |
| `state` | agent xy, float32 (2,) |
| `actions` | absolute agent target xy in `[0, 512]`, shape (2,) |
| success | block/target coverage > 0.95 |
| metrics | `mean_success`, `mean_coverage` |
