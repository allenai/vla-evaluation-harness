# Reproduction Roadmap

Systematic verification that vla-eval reproduces published model scores.

## Stages

### Stage 1 — Breadth: many models, one benchmark

Verify as many supported models as possible on a single benchmark where most models have reported scores. LIBERO is the natural choice: 6+ models publish LIBERO numbers.

**Goal**: For each model, confirm that vla-eval produces scores within noise margin of the authors' reported numbers — or identify and document the gap.

**Status**: 3 reproduced, 2 with identified root causes, 1 aborted.

| Model | Reported | Reproduced | Verdict | Notes |
|-------|:--------:|:----------:|:-------:|-------|
| X-VLA (0.9B) | 98.1% | **97.8%** | Reproduced | |
| Pi0.5 | 96.9% | **96.4%** | Reproduced | Spatial suite 6pp low (image flip) |
| OpenVLA-OFT (joint) | ~96.8% | **97.0%** | Reproduced | |
| StarVLA (4 variants) | 95-97% | 0% | Not reproduced | Image ordering suspected |
| GR00T N1.6 | 97.0% | ~1% | Not reproduced | Missing `use_sim_policy_wrapper` |
| OpenVLA base (LoRA) | 76.5% | — | Aborted | Too slow without batch prediction |

Full details: [stage1-libero.md](stage1-libero.md)

### Stage 2 — Depth: top models, multiple benchmarks

Cross-validate 3-4 representative models across 3-4 representative benchmarks, confirming that the harness produces consistent results across the full evaluation matrix.

**Goal**: For each (model, benchmark) pair, reproduce published scores — establishing that the harness's decoupled architecture (model server + benchmark container) does not introduce systematic bias.

**Status**: DB-CogACT complete. Others pending Stage 1 resolution.

| Model | LIBERO | CALVIN | SimplerEnv | RoboTwin | Report |
|-------|:------:|:------:|:----------:|:--------:|--------|
| DB-CogACT | Reproduced | Reproduced | Reproduced | — | [db-cogact.md](db-cogact.md) |
| X-VLA | Reproduced (S1) | — | — | — | Pending |
| OFT-joint | Reproduced (S1) | — | — | — | Pending |
| Pi0.5 | Reproduced (S1) | — | — | — | Pending |

## Reference Data

- [reported-performance.md](reported-performance.md) — Officially reported scores from papers/model cards for all supported models. Not our measurements.

## Result Artifacts

Raw evaluation outputs (merged result JSONs, eval configs, server logs) are archived under [`data/`](data/).
