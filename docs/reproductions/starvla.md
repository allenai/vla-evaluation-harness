# StarVLA — Reproduction Report

VLA framework with pluggable action heads. [GitHub](https://github.com/starVLA/starVLA) (branch `starVLA`) |
No formal paper. Results from README + assets.

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO Spatial | **96.8%** (Qwen3-OFT) | 97.8% | Reproduced |
| CALVIN ABC→D | — | 3.79 (Q2.5-GR00T) | Not yet evaluated |
| SimplerEnv WidowX | **66.7%** (Qwen3-GR00T) | 65.3% | Reproduced |
| SimplerEnv WidowX | **64.6%** (Q2.5-FAST) | 58.6% | Reproduced |
| SimplerEnv WidowX | 20.8% (Qwen3-OFT) | 42.7% | Not reproduced (env issue) |
| RoboTwin | — | 50.4% (Qwen3-OFT) | Not yet evaluated |

### LIBERO — Spatial

| | |
|---|---|
| **Checkpoint** | `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` (official) |
| **Server config** | [`configs/model_servers/starvla/libero_qwen3_oft.yaml`](../../configs/model_servers/starvla/libero_qwen3_oft.yaml) |
| **Benchmark config** | [`configs/benchmarks/libero/spatial.yaml`](../../configs/benchmarks/libero/spatial.yaml) |
| **Results** | [`data/starvla-libero/`](data/starvla-libero/) |

10 tasks x 50 episodes. `unnorm_type: minmax`, `send_wrist_image: true`.

| Task | Reproduced | Reported |
|------|:----------:|:--------:|
| bowl between plate and ramekin | 100% | — |
| bowl from table center | 100% | — |
| bowl next to cookie box | 98% | — |
| bowl on cookie box | 98% | — |
| bowl on stove | 98% | — |
| bowl next to ramekin | 96% | — |
| bowl on wooden cabinet | 96% | — |
| bowl in top drawer | 94% | — |
| bowl next to plate | 94% | — |
| bowl on ramekin | 94% | — |
| **Average** | **96.8%** | **97.8%** |

Within binomial 95% CI (±1.9pp).

### SimplerEnv — WidowX VM

#### Qwen3-GR00T (Reproduced)

| | |
|---|---|
| **Checkpoint** | `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` (official) |
| **Server config** | [`configs/model_servers/starvla/groot_qwen3_simpler.yaml`](../../configs/model_servers/starvla/groot_qwen3_simpler.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/widowx_vm.yaml`](../../configs/benchmarks/simpler/widowx_vm.yaml) |
| **Results** | [`data/starvla/`](data/starvla/) |

4 tasks x 24 episodes. Adaptive ensemble (horizon=7, alpha=0.1). Euler-to-axisangle conversion.

| Task | Reproduced | Reported (avg of 4 runs) |
|------|:----------:|:------------------------:|
| Stack | 25.0% | 18.8% |
| Carrot | 58.3% | 59.4% |
| Spoon | 83.3% | 75.0% |
| Eggplant | 100% | 100% |
| **Average** | **66.7%** | **65.3%** |

#### Q2.5-FAST (Reproduced)

| | |
|---|---|
| **Checkpoint** | `StarVLA/Qwen-FAST-Bridge-RT-1` (official) |
| **Server config** | [`configs/model_servers/starvla/fast_simpler.yaml`](../../configs/model_servers/starvla/fast_simpler.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/widowx_vm.yaml`](../../configs/benchmarks/simpler/widowx_vm.yaml) |
| **Results** | [`data/starvla-fast/`](data/starvla-fast/) |

4 tasks x 24 episodes. Adaptive ensemble (horizon=7, alpha=0.1). Euler-to-axisangle conversion.

| Task | Reproduced | Reported |
|------|:----------:|:--------:|
| Stack | 37.5% | 36.5% |
| Carrot | 37.5% | 41.7% |
| Spoon | 95.8% | 71.9% |
| Eggplant | 87.5% | 84.4% |
| **Average** | **64.6%** | **58.6%** |

#### Qwen3-OFT (Not reproduced)

| | |
|---|---|
| **Checkpoint** | `StarVLA/Qwen3VL-OFT-Bridge-RT-1` (official) |
| **Server config** | [`configs/model_servers/starvla/oft_qwen3_simpler.yaml`](../../configs/model_servers/starvla/oft_qwen3_simpler.yaml) |

Result: 20.8% vs reported 42.7%. Running the author's evaluation logic directly
(no framework) in our environment produces the same low result, confirming the
gap is environmental, not a code bug. The Qwen3-GR00T variant reproduces
successfully with the same integration code, ruling out framework-level issues.
See [`data/starvla/reproduction_gap_investigation.md`](data/starvla/reproduction_gap_investigation.md) for details.

### Reported Scores by Variant

**SimplerEnv WidowX** (VM, from author's eval logs shipped with checkpoints):

| Variant | Spoon | Carrot | Stack | Eggplant | Avg |
|---------|:-----:|:------:|:-----:|:--------:|:---:|
| Qwen2.5-VL-FAST | 71.9 | 41.7 | 36.5 | 84.4 | 58.6 |
| Qwen2.5-VL-GR00T | 82.3 | 54.2 | 40.6 | 70.1 | 63.6 |
| Qwen3-VL-OFT | 90.3 | 38.5 | 9.7 | 100.0 | 59.6 |
| Qwen3-VL-GR00T | 83.0 | 59.4 | 18.8 | 100.0 | 65.3 |

**LIBERO** (all single-policy, 30K steps):

| Variant | Spatial | Object | Goal | Long | Avg |
|---------|:-------:|:------:|:----:|:----:|:---:|
| Qwen3-VL-OFT | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Qwen3-VL-GR00T | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |

**CALVIN ABC→D**: Q2.5-VL-GR00T = 3.79.

### Previous Reproduction Attempts

**StarVLA Q2.5-GR00T x LIBERO**: 29.6% spatial (reported 95.4%) — excluded.
Possible causes: wrong checkpoint (Bridge+RT-1 instead of LIBERO-finetuned?).

**Excluded variants**:
- StarVLA Q2.5-OFT / Q3-OFT: Supply <7 obs/s (chunk_size=1). 4-20 hours per suite.
- StarVLA Qwen3-PI: state_dict mismatch (36 vs 16 transformer blocks).

## Configuration Notes

- StarVLA is a **framework**, not a single model. Results vary by action head x backbone.
- SimplerEnv checkpoints trained on **Bridge + RT-1** data (not Bridge alone).
- LIBERO reported as single-policy (all 4 suites jointly), not per-suite finetuned.
- RoboTwin uses Protocol B (multi-task) — not comparable to Protocol A entries.

## Data

- [`data/starvla/`](data/starvla/) — Qwen3-GR00T SimplerEnv results + OFT investigation
- [`data/starvla-fast/`](data/starvla-fast/) — Q2.5-FAST SimplerEnv results
- [`data/starvla-libero/`](data/starvla-libero/) — Qwen3-OFT LIBERO-Spatial results
