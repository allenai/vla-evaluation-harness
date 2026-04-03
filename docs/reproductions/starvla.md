# StarVLA — Reproduction Report

VLA framework with pluggable action heads. [GitHub](https://github.com/starVLA/starVLA) (branch `starVLA`) |
No formal paper. Results from README + assets.

## Checkpoints

| Benchmark | Variant | Checkpoint | Source |
|-----------|---------|------------|--------|
| LIBERO | Multiple variants | Not yet identified on HF | unclear |
| CALVIN ABC→D | QwenGR00T (Q2.5-action) | `Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D` | community |
| SimplerEnv WidowX | Multiple variants | `StarVLA/Qwen-GR00T-Bridge-RT-1` etc. | official (Bridge+RT-1 trained) |
| RoboTwin | Qwen3-OFT | Multiple | official |

## Results Summary

| Benchmark | Reproduced | Best Reported | Verdict |
|-----------|:----------:|:-------------:|:-------:|
| LIBERO | — | 96.6% (Qwen3-OFT) | Not yet evaluated |
| CALVIN ABC→D | — | 3.79 (QwenGR00T Q2.5-action) | Not yet evaluated |
| SimplerEnv WidowX | — | 65.3% (Qwen3-GR00T) | Not yet evaluated |
| RoboTwin | — | 50.4% (Qwen3-OFT, Protocol B) | Not yet evaluated |

### Reported Scores by Variant

**LIBERO** (all single-policy, 30K steps):

| Variant | Spatial | Object | Goal | Long | Avg |
|---------|:-------:|:------:|:----:|:----:|:---:|
| Qwen2.5-VL-FAST | 97.3 | 97.2 | 96.1 | 90.2 | 95.2 |
| Qwen2.5-VL-OFT | 97.4 | 98.0 | 96.8 | 92.0 | 96.1 |
| Qwen2.5-VL-PI | 98.2 | 99.2 | 95.6 | 88.4 | 95.4 |
| Qwen2.5-VL-GR00T | 97.8 | 98.2 | 94.6 | 90.8 | 95.4 |
| Qwen3-VL-FAST | 97.3 | 97.4 | 96.3 | 90.6 | 95.4 |
| Qwen3-VL-OFT | 97.8 | 98.6 | 96.2 | 93.8 | 96.6 |
| Qwen3-VL-PI | 98.8 | 99.6 | 95.8 | 88.4 | 95.7 |
| Qwen3-VL-GR00T | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |

**SimplerEnv WidowX** (VM):

| Variant | Spoon | Carrot | Block | Eggplant | Avg |
|---------|:-----:|:------:|:-----:|:--------:|:---:|
| Qwen2.5-VL-FAST | 71.9 | 41.7 | 36.5 | 84.4 | 58.6 |
| Qwen2.5-VL-GR00T | 82.3 | 54.2 | 40.6 | 70.1 | 63.6 |
| Qwen3-VL-OFT | 90.3 | 38.5 | 9.7 | 100.0 | 59.6 |
| Qwen3-VL-GR00T | 83.0 | 59.4 | 18.8 | 100.0 | 65.3 |

**CALVIN ABC→D**: QwenGR00T (Q2.5-VL-3B-action) = 3.79. FAST/OFT: "will be released soon."

### Previous Reproduction Attempts

**StarVLA Q2.5-GR00T × LIBERO**: 29.6% spatial (reported 95.4%) — excluded.
Possible causes: wrong checkpoint (Bridge+RT-1 instead of LIBERO-finetuned?), model server bug, or missing configuration.

**Excluded variants**:
- StarVLA Q2.5-OFT / Q3-OFT: Supply <7 obs/s (chunk_size=1). 4-20 hours per suite.
- StarVLA Qwen3-PI: state_dict mismatch (36 vs 16 transformer blocks).

## Configuration Notes

- StarVLA is a **framework**, not a single model. Results vary by action head × backbone combination.
- `-action` suffix (e.g., `Qwen2.5-VL-3B-Instruct-Action`) denotes an action-specialized backbone variant.
- SimplerEnv checkpoints trained on **Bridge + RT-1** data (not Bridge alone).
- LIBERO reported as single-policy (all 4 suites jointly), not per-suite finetuned.
- RoboTwin uses Protocol B (multi-task) — not comparable to Protocol A entries (X-VLA, Pi0).

## Data

No evaluation data yet.
