# Officially Reported Performance of Supported Model Servers

> Last updated: 2026-03-24
>
> This document records **officially reported** benchmark scores for each model server
> supported in vla-eval. "Official" means numbers from the original authors' paper,
> GitHub repo, or HuggingFace model card — not third-party reproductions (unless noted).
>
> **Do not confuse with `leaderboard/data/results.json`** — that file is a separate,
> independently curated dataset of paper-reported scores across the entire VLA community.

## Table of Contents

- [Summary Table](#summary-table)
- [OpenVLA](#openvla)
- [OpenVLA-OFT](#openvla-oft)
- [CogACT](#cogact)
- [DB-CogACT (Dexbotic)](#db-cogact-dexbotic)
- [Pi0](#pi0)
- [GR00T](#groot)
- [StarVLA](#starvla)
- [X-VLA](#x-vla)
- [RTC](#rtc)
- [Notes on Evaluation Protocols](#notes-on-evaluation-protocols)

---

## Summary Table

| Model | Params | CALVIN | LIBERO | SE GR-VM | SE GR-VA | SE WidowX | RT2 Easy | RT2 Hard |
|---|---|---|---|---|---|---|---|---|
| OpenVLA | 7B | — | 76.5% (ft) | — | — | — | — | — |
| OpenVLA-OFT | 7B | — | 97.1% (ft) | — | — | — | — | — |
| CogACT (DiT-B) | 7B | — | — | 74.8% | 61.3% | 51.3% | — | — |
| DB-CogACT | 7B | 4.06 | 94.9% (ft) | — | — | 69.5% | 58.5%† | — |
| Pi0.5 | — | — | 96.9% (ft) | — | — | — | — | — |
| GR00T N1.6 | 2B | — | 97.0% (ft) | 67.7%‡ | — | 62.1%‡ | — | — |
| StarVLA Qwen2.5-FAST | 3B | — | 95.2 | — | — | 58.6% | — | — |
| StarVLA Qwen2.5-OFT | 3B | — | 96.1 | — | — | 44.2% | — | — |
| StarVLA Qwen2.5-PI | 3B | — | 95.4 | — | — | 62.5% | — | — |
| StarVLA Qwen2.5-GR00T | 3B | 3.79 | 95.4 | — | — | 63.6% | — | — |
| StarVLA Qwen3-FAST | 4B | — | 95.4 | — | — | 31.6% | — | — |
| StarVLA Qwen3-OFT | 4B | — | 96.6 | — | — | 59.6% | 50.4%§ | — |
| StarVLA Qwen3-PI | 4B | — | 95.7 | — | — | 60.9% | — | — |
| StarVLA Qwen3-GR00T | 4B | 3.76 | 96.5 | — | — | 65.3% | — | — |
| X-VLA | 0.9B | 4.43 | 98.1% (sh) | 80.4% | 75.7% | 95.8% | 70.0% | 39.0% |
| RTC | — | — | — | — | — | — | — | — |

SE = SimplerEnv, RT2 = RoboTwin 2.0.
(ft) = per-suite/per-task finetuned, (sh) = shared weights (single model for all suites).
† = 4 tasks only. ‡ = non-standard task set. § = Protocol B (multi-task), 48 tasks.
— = not reported.

---

## OpenVLA

**Paper**: [arxiv 2406.09246](https://arxiv.org/abs/2406.09246) | **Params**: 7B
**Weights**: `openvla/openvla-7b` (HuggingFace) — pretrained generalist checkpoint
**GitHub**: [openvla/openvla](https://github.com/openvla/openvla)

### LIBERO — Table 9 (Appendix E.2)

Per-suite finetuned (LoRA r=32, third-person image only, filtered dataset).

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | 84.7% |
| LIBERO-Object | 88.4% |
| LIBERO-Goal | 79.2% |
| LIBERO-10 | 53.7% |
| **Average** | **76.5%** |

**Reproduction notes:**
- Per-suite LoRA finetuning — no single shared-weights LIBERO result exists.
- Uses **filtered dataset** (unsuccessful demos removed). Results on unfiltered data would differ.
- **Third-person image only** — no wrist camera, no proprioception.
- `unnorm_key` must match dataset: `libero_spatial_no_noops`, `libero_object_no_noops`, etc.

### CALVIN

Not reported.

### SimplerEnv

Not reported. The authors stated in [GitHub issue #7](https://github.com/openvla/openvla/issues/7)
that they do not have fully vetted SimplerEnv results. Real-robot BridgeData V2 WidowX
results (70.6%) and Google Robot results (85.0%) are from physical hardware, not SimplerEnv sim.

### RoboTwin

Not reported. Paper predates RoboTwin 2.0.

---

## OpenVLA-OFT

**Paper**: [arxiv 2502.19645](https://arxiv.org/abs/2502.19645) | **Params**: 7B
**Weights (per-suite)**: `moojink/openvla-7b-oft-finetuned-libero-{suite}` (HuggingFace)
  — `libero-spatial`, `libero-object`, `libero-goal`, `libero-10`
**Weights (combined)**: `moojink/openvla-7b-oft-finetuned-libero-combined` (HuggingFace)
**GitHub**: [moojink/openvla-oft](https://github.com/moojink/openvla-oft)

### LIBERO — Table I

Per-suite finetuned, 500 trials/suite (10 tasks × 50 episodes).

**Primary result (3rd-person + wrist camera + proprioception, filtered dataset):**

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | 97.6% |
| LIBERO-Object | 98.4% |
| LIBERO-Goal | 97.9% |
| LIBERO-10 | 94.5% |
| **Average** | **97.1%** |

**Image + language only (filtered dataset):**

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | 96.2% |
| LIBERO-Object | 98.3% |
| LIBERO-Goal | 96.2% |
| LIBERO-10 | 90.7% |
| **Average** | **95.3%** |

**Shared-weights (single model, all 4 suites combined) — Appendix A-G1:**

Average **~96.8%** (from LIBERO.md in repo).

**Reproduction notes:**
- The 97.1% headline uses **3rd-person + wrist camera + proprioception** — reproducing
  with image-only input yields 95.3% instead.
- Uses **filtered dataset** (same as OpenVLA). Unfiltered dataset yields 94.5%.
- Per-suite `unnorm_key` required: `libero_spatial_no_noops`, etc.
- Continuous action head with L1 loss (Cont-L1 variant, not diffusion).
- Action chunk size = 10 (parallel decoding).

### CALVIN

Not reported.

### SimplerEnv

Not reported. The paper evaluates on real-robot WidowX only.

### RoboTwin

Not reported.

---

## CogACT

**Paper**: [arxiv 2411.19650](https://arxiv.org/abs/2411.19650) | **Params**: 7B (DiT-Base action head)
**Weights**: `CogACT/CogACT-Base` (HuggingFace) — also `CogACT-Small` (DiT-S), `CogACT-Large` (DiT-L)
**GitHub**: [microsoft/CogACT](https://github.com/microsoft/CogACT)

### SimplerEnv — Tables 1 & 2

**Google Robot Visual Matching (VM):**

| Task | Score |
|---|---|
| Pick Coke Can | 91.3% |
| Move Near | 85.0% |
| Open/Close Drawer | 71.8% |
| Place Apple in Drawer | 50.9% |
| **4-task Average** | **74.8%** |
| **3-task Average** (standard) | **82.7%** |

**Google Robot Variant Aggregation (VA):**

| Metric | Score |
|---|---|
| **4-task Average** | **61.3%** |

**WidowX Visual Matching (VM):**

| Task | Score |
|---|---|
| Put Spoon on Towel | 71.7% |
| Put Carrot on Plate | 50.8% |
| Stack Green Block | 15.0% |
| Put Eggplant in Basket | 67.5% |
| **Average** | **51.3%** |

**Reproduction notes:**
- Multi-dataset checkpoint — `unnorm_key` selects denormalization stats: `bridge_orig`
  (BridgeData V2 / WidowX), `fractal20220817_data` (RT-1 / Google Robot).
- DiT action head uses DDIM sampling: `cfg_scale=1.5`, `num_ddim_steps=10`.
- Action chunk = 16 steps (future_action_window_size=15 → 16 total).
- The 4-task VM average (74.8%) includes Place Apple in Drawer; the standard 3-task
  average (without it) is **82.7%**.

### CALVIN

Not reported in the original paper.

### LIBERO

Not reported in the original paper.

### RoboTwin

Not reported in the original paper.

---

## DB-CogACT (Dexbotic)

**Paper**: [arxiv 2510.23511](https://arxiv.org/abs/2510.23511) | **Params**: 7B (DexboticVLM: Qwen2.5 + CLIP + DiT)
**Weights (per-benchmark)**:
  - CALVIN: `Dexmal/calvin-db-cogact`
  - LIBERO: `Dexmal/libero-db-cogact`
  - SimplerEnv: `Dexmal/simpler-db-cogact`
  - RoboTwin: `Dexmal/robotwin-db-cogact` (per-task subdirectories, e.g. `/adjust_bottle`)
  - ManiSkill2: `Dexmal/maniskill2-db-cogact`
**GitHub**: [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic)

Architecture differs from original CogACT: replaces Llama2 backbone with Qwen2.5-based
DexboticVLM, pretrained on OXE subsets + private data. CogACT DiT action head retained.

### CALVIN (ABC→D) — Table 4

| Step | DB-CogACT | CogACT (repro) |
|---|---|---|
| 1 | 93.5% | 83.8% |
| 2 | 86.7% | 72.9% |
| 3 | 80.3% | 64.0% |
| 4 | 76.0% | 55.9% |
| 5 | 69.8% | 48.0% |
| **Avg Length** | **4.06** | **3.25** |

### LIBERO — Table 3

Per-suite finetuned.

| Suite | DB-CogACT | CogACT (repro) |
|---|---|---|
| LIBERO-Spatial | 93.8% | 97.2% |
| LIBERO-Object | 97.8% | 98.0% |
| LIBERO-Goal | 96.2% | 90.2% |
| LIBERO-10 | 91.8% | 88.8% |
| **Average** | **94.9%** | **93.6%** |

### SimplerEnv — WidowX VM only

| Task | DB-CogACT | CogACT (repro) |
|---|---|---|
| Put Spoon on Towel | 87.5% | 71.7% |
| Put Carrot on Plate | 65.3% | 50.8% |
| Stack Cube | 29.2% | 15.0% |
| Put Eggplant in Basket | 95.8% | 67.5% |
| **Average** | **69.5%** | **51.3%** |

Google Robot results not reported.

### RoboTwin 2.0 — Table 2 (Easy mode, 4 tasks only)

Likely Protocol A (single-task). No Hard scores reported.

| Task | DB-CogACT | CogACT (repro) |
|---|---|---|
| Adjust Bottle | 99% | 87% |
| Grab Roller | 89% | 72% |
| Place Empty Cup | 28% | 11% |
| Place Phone Stand | 18% | 5% |
| **Average** | **58.5%** | **43.75%** |

Note: Only 4 of 50 RoboTwin tasks evaluated. Not comparable to full 50-task results.

**Reproduction notes:**
- Each benchmark has a **separate checkpoint** — no shared-weights model.
- LIBERO uses per-suite `chunk_size_map`: Spatial=12, Object=16, Goal=16, Long=15.
- RoboTwin uses **per-task subdirectories** under the HF repo (e.g. `robotwin-db-cogact/adjust_bottle`).
  Must switch `model_path` per task. Uses 3 cameras: `head_camera`, `left_camera`, `right_camera`.
- `use_text_template: true` for CALVIN, LIBERO, SimplerEnv (wraps task instruction in a template).
- The "CogACT (repro)" baseline numbers in Dexbotic's tables are their own re-evaluation
  of the official CogACT checkpoint — not from the original CogACT paper.

---

## Pi0

**Paper**: [arxiv 2410.24164](https://arxiv.org/abs/2410.24164) (π0)
**Weights**: Distributed via GCS, not HuggingFace. Accessed through `openpi` config system.
  - LIBERO: `gs://openpi-assets/checkpoints/pi05_libero` (Pi0.5, not Pi0)
  - Config names: `pi0_fast_libero`, `pi05_libero`, `pi0_fast_droid`, `pi05_droid`, etc.
**GitHub**: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

The original Pi0 paper evaluates exclusively on real-robot tasks. No CALVIN, LIBERO,
SimplerEnv, or RoboTwin results are reported by the authors for Pi0.

### Pi0.5 — LIBERO (openpi repo)

From [openpi examples/libero/README.md](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md).
Checkpoint: `pi05_libero` (Pi0.5 finetuned, not Pi0 base).

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | 98.8% |
| LIBERO-Object | 98.2% |
| LIBERO-Goal | 98.0% |
| LIBERO-10 | 92.4% |
| **Average** | **96.85%** |

### Pi0 — RoboTwin 2.0 (third-party, by RoboTwin authors)

From RoboTwin 2.0 paper ([arxiv 2506.18088](https://arxiv.org/abs/2506.18088)), Table 5.
Protocol A (single-task, 50 clean demos/task, 50 tasks). **Not self-reported by Physical Intelligence.**

| Metric | Score |
|---|---|
| Easy | 46.4% |
| Hard | 16.3% |

**Reproduction notes:**
- The LIBERO result is **Pi0.5** (not Pi0 original). No Pi0 LIBERO numbers exist.
- openpi uses its own config/checkpoint system — not standard HF `from_pretrained`.
  Requires `openpi` package installation and GCS access.
- LIBERO eval uses 3rd-person image + wrist image + proprioceptive state (8-dim).
- `pi0_fast_libero` (Pi0-FAST) is a different, lower-performing variant.
- The RoboTwin result for Pi0 is **third-party** (by the RoboTwin 2.0 paper authors),
  not self-reported by Physical Intelligence.

### CALVIN / SimplerEnv

Not reported for either Pi0 or Pi0.5.

---

## GR00T

**Paper**: [arxiv 2503.14734](https://arxiv.org/abs/2503.14734) (GR00T N1) | **Params**: 2B
**Weights**:
  - Foundation: `nvidia/GR00T-N1.6-3B` (HuggingFace)
  - LIBERO finetuned: requires finetuning from foundation (see Isaac-GR00T examples)
  - SimplerEnv finetuned: `nvidia/GR00T-N1.6-bridge` (WidowX), `nvidia/GR00T-N1.6-fractal` (Google Robot)
**GitHub**: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

The N1 paper evaluates on RoboCasa, DexMG, and GR-1 humanoid tasks only. LIBERO and
SimplerEnv results exist only for **GR00T N1.6** (post-paper model) via the GitHub repo.

### GR00T N1.6 — LIBERO (GitHub, finetuned)

From [examples/LIBERO/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md).
20K gradient steps, batch 640. Per-suite finetuned.

| Suite | Trials | Success Rate |
|---|---|---|
| LIBERO-Spatial | 200 | 97.65% |
| LIBERO-Object | 200 | 98.45% |
| LIBERO-Goal | 200 | 97.50% |
| LIBERO-10 | 200 | 94.35% |
| **Average** | | **96.99%** |

### GR00T N1.6 — SimplerEnv (GitHub, finetuned)

From [examples/SimplerEnv/README.md](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md).
Task-specific finetuned checkpoints (`nvidia/GR00T-N1.6-bridge`, `nvidia/GR00T-N1.6-fractal`).

**WidowX (7 tasks, non-standard set — includes open/close drawer tasks not in the usual 4-task set):**

| Task | Score |
|---|---|
| Spoon on Towel | 64.5% |
| Carrot on Plate | 65.5% |
| Eggplant in Basket | 93.0% |
| Stack Cube | 5.5% |
| Eggplant in Sink | 40.0% |
| Close Drawer | 70.5% |
| Open Drawer | 95.5% |
| **Average (7 tasks)** | **62.1%** |

**Google Robot (6 tasks, non-standard set — includes place-in-drawer):**

| Task | Score |
|---|---|
| Pick Coke Can | 97.5% |
| Pick Object | 87.0% |
| Move Near | 75.5% |
| Open Drawer | 44.0% |
| Close Drawer | 87.5% |
| Place in Closed Drawer | 14.5% |
| **Average (6 tasks)** | **67.7%** |

Note: These task sets differ from the standard 4-task WidowX / 3-task Google Robot protocols
used by most papers. Direct comparison requires matching task subsets.

**Reproduction notes:**
- N1 (paper) ≠ N1.5 ≠ N1.6 — versions differ significantly. All benchmark numbers above
  are **N1.6** only.
- LIBERO finetuned checkpoints are not pre-released on HF — must finetune from
  `nvidia/GR00T-N1.6-3B` foundation model using Isaac-GR00T repo (20K steps, batch 640).
- `embodiment_tag` must be set per benchmark: `LIBERO_PANDA`, `OXE_GOOGLE`, `OXE_WIDOWX`.
- SimplerEnv task set is **non-standard** — includes extra tasks (open/close drawer for
  WidowX, place-in-closed-drawer for Google Robot) not in the community 4/3-task protocols.
  Comparing the 4-task WidowX subset (Spoon 64.5, Carrot 65.5, Eggplant 93.0, Block 5.5 =
  **57.1% avg**) would be more fair.
- Action chunk size = 16.

### CALVIN / RoboTwin

Not reported.

---

## StarVLA

**No formal paper.** Results from [github.com/starVLA/starVLA](https://github.com/starVLA/starVLA) README (branch `starVLA`).
All variants use Qwen2.5-VL or Qwen3-VL as the vision-language backbone with different action heads.
**Weights (SimplerEnv / Bridge+RT-1 trained)**:
  - `StarVLA/Qwen-FAST-Bridge-RT-1` (QwenFAST, Qwen2.5-VL)
  - `StarVLA/Qwen-GR00T-Bridge-RT-1` (QwenGR00T, Qwen2.5-VL)
  - `StarVLA/Qwen-OFT-Bridge-RT-1` (QwenOFT, Qwen2.5-VL)
  - `StarVLA/Qwen-PI-Bridge-RT-1` (QwenPI, Qwen2.5-VL)
  - `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` (QwenGR00T, Qwen3-VL)
  - `StarVLA/Qwen3VL-OFT-Bridge-RT-1` (QwenOFT, Qwen3-VL)
**Weights (CALVIN)**:
  - `Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D`
**Weights (Bridge-only, no RT-1)**:
  - `StarVLA/Qwen-GR00T-Bridge`
**GitHub**: [starVLA/starVLA](https://github.com/starVLA/starVLA) (branch `starVLA`)

### CALVIN (ABC→D) — from `assets/calvin.png`

| Variant | Backbone | Avg Length |
|---|---|---|
| QwenPI | Qwen2.5-VL-3B-action | 3.58 |
| QwenPI | Qwen3-VL-4B | 3.47 |
| QwenGR00T | Qwen2.5-VL-3B | 3.70 |
| **QwenGR00T** | **Qwen2.5-VL-3B-action** | **3.79** |
| QwenGR00T | Qwen3-VL-4B | 3.65 |
| QwenGR00T | Qwen3-VL-4B-action | 3.76 |

FAST/OFT CALVIN results noted as "will be released soon."

### LIBERO — from `assets/starvla_LIBERO.png`

All models trained for 30K steps, single policy for all 4 suites.

| Variant | Spatial | Object | Goal | Long | **Avg** |
|---|---|---|---|---|---|
| Qwen2.5-VL-FAST | 97.3 | 97.2 | 96.1 | 90.2 | 95.2 |
| Qwen2.5-VL-OFT | 97.4 | 98.0 | 96.8 | 92.0 | 96.1 |
| Qwen2.5-VL-PI | 98.2 | 99.2 | 95.6 | 88.4 | 95.4 |
| Qwen2.5-VL-GR00T | 97.8 | 98.2 | 94.6 | 90.8 | 95.4 |
| Qwen3-VL-FAST | 97.3 | 97.4 | 96.3 | 90.6 | 95.4 |
| **Qwen3-VL-OFT** | **97.8** | **98.6** | **96.2** | **93.8** | **96.6** |
| Qwen3-VL-PI | 98.8 | 99.6 | 95.8 | 88.4 | 95.7 |
| Qwen3-VL-GR00T | 97.8 | 98.8 | 97.4 | 92.0 | 96.5 |

### SimplerEnv — WidowX VM only, from `assets/starvla_simpleEnv.png`

No Google Robot results published.

| Variant | Steps | Spoon | Carrot | Block | Eggplant | **Avg** |
|---|---|---|---|---|---|---|
| Qwen2.5-VL-FAST | 10K | 71.9 | 41.7 | 36.5 | 84.4 | 58.6 |
| Qwen2.5-VL-OFT | 10K | 39.6 | 40.3 | 10.4 | 86.5 | 44.2 |
| Qwen2.5-VL-PI | 30K | 82.3 | 60.4 | 35.4 | 71.9 | 62.5 |
| Qwen2.5-VL-GR00T | 30K | 82.3 | 54.2 | 40.6 | 70.1 | 63.6 |
| Qwen3-VL-FAST | 15K | 18.8 | 31.3 | 4.2 | 71.9 | 31.6 |
| Qwen3-VL-OFT | 65K | 90.3 | 38.5 | 9.7 | 100.0 | 59.6 |
| Qwen3-VL-PI | 40K | 78.1 | 46.9 | 30.2 | 88.5 | 60.9 |
| **Qwen3-VL-GR00T** | **20K** | **83.0** | **59.4** | **18.8** | **100.0** | **65.3** |

### RoboTwin 2.0 — Protocol B (multi-task), from `examples/Robotwin/README.md`

| Config | Tasks | Easy | Hard |
|---|---|---|---|
| Qwen3-VL-OFT (50 demos/task) | 48 | 50.4% | — |
| Qwen3-VL-OFT (50 + 500 DR demos/task) | 50 | 88.2% | 88.3% |

Protocol B = multi-task joint training. Not comparable to Protocol A results (X-VLA, Pi0).

**Reproduction notes:**
- StarVLA is a **toolbox/framework**, not a single model. Results vary significantly
  across action heads (FAST, OFT, PI, GR00T) and backbones (Qwen2.5-VL vs Qwen3-VL).
- `-action` suffix on backbone (e.g. `Qwen2.5-VL-3B-Instruct-Action`) denotes an
  action-specialized finetuned backbone variant — distinct from the base instruct model.
- SimplerEnv checkpoints are trained on **Bridge + RT-1** data (not Bridge alone).
  The Bridge-only variant (`Qwen-GR00T-Bridge`) reportedly achieves 71.4% on WidowX.
- CALVIN checkpoint exists only for QwenGR00T (Qwen2.5-VL-3B-action). FAST/OFT CALVIN
  results marked "will be released soon."
- LIBERO reported as single-policy (all 4 suites jointly), not per-suite finetuned.
- No Google Robot SimplerEnv results published.
- RoboTwin uses Protocol B (multi-task) — not comparable to Protocol A entries.
- Training compute varies: FAST/OFT 10K steps (3h on 16×A100), PI/GR00T 30K steps (18h).

---

## X-VLA

**Paper**: [arxiv 2510.10274](https://arxiv.org/abs/2510.10274) | **Params**: 0.9B (Florence-2-large)
**Weights (per-benchmark, full finetune)**:
  - LIBERO: `2toINF/X-VLA-Libero` (single model, all 4 suites)
  - CALVIN: `2toINF/X-VLA-Calvin-ABC_D`
  - SimplerEnv/WidowX: `2toINF/X-VLA-WidowX`
**Weights (PEFT/LoRA, lower performance)**:
  - `2toINF/X-VLA-libero-spatial-peft`, `-object-peft`, `-goal-peft`, `-long-peft`
  - PEFT avg ~93% vs full-finetune 98.1%
**GitHub**: [2toinf/X-VLA](https://github.com/2toinf/X-VLA)

X-VLA uses **shared weights** — a single model across all benchmarks, with lightweight
domain-specific soft prompts (~0.04% params per embodiment).

### CALVIN (ABC→D) — Table 2

Finetuned on CALVIN.

| Chain length | Success Rate |
|---|---|
| 1 | 97.1% |
| 2 | 92.6% |
| 3 | 88.5% |
| 4 | 84.4% |
| 5 | 78.8% |
| **Avg Length** | **4.43** |

### LIBERO — Table 2

**Shared weights** — single model jointly trained on all 4 suites (not per-suite finetuned).

| Suite | Success Rate |
|---|---|
| LIBERO-Spatial | 98.2% |
| LIBERO-Object | 98.6% |
| LIBERO-Goal | 97.8% |
| LIBERO-10 | 97.6% |
| **Average** | **98.1%** |

### SimplerEnv — Table 2, Appendix Table 12

**Google Robot Visual Matching (VM):**

| Task | Score |
|---|---|
| Pick Coke Can | 98.3% |
| Move Near | 97.1% |
| Open/Close Drawer | 69.5% |
| Place Apple in Drawer | 56.5% |
| **4-task Average** | **80.4%** |
| **3-task Average** (standard) | **88.3%** |

**Google Robot Variant Aggregation (VA):**

| Task | Score |
|---|---|
| Pick Coke Can | 85.5% |
| Move Near | 79.8% |
| Open/Close Drawer | 61.9% |
| Place Apple in Drawer | 75.7% |
| **4-task Average** | **75.7%** |
| **3-task Average** (standard) | **75.7%** |

**WidowX Visual Matching (VM):**

| Task | Score |
|---|---|
| Spoon on Towel | 100.0% |
| Carrot on Plate | 91.7% |
| Stack Block | 95.8% |
| Put Eggplant in Basket | 95.8% |
| **Average** | **95.8%** |

### RoboTwin 2.0 — Table 16

50 tasks. Likely Protocol A (single-task, 50 clean demos/task).

| Metric | Score |
|---|---|
| Easy | 70.0% |
| Hard | 39.0% |

**Reproduction notes:**
- LIBERO uses **shared weights** (single checkpoint for all 4 suites), unlike most other
  models that use per-suite finetuning. This makes X-VLA's 98.1% especially notable.
- Each benchmark requires a different `domain_id`: LIBERO=3, CALVIN=2.
  `benchmark_profile` sets action space normalization.
- Uses flow matching with `denoising_steps=10`.
- Chunk size differs per benchmark: LIBERO=30, CALVIN=20.
- GitHub README shows slightly higher SimplerEnv numbers than paper (VM 83.5% vs 80.4%,
  VA 76.4% vs 75.7%) — likely a post-publication re-evaluation. Paper numbers are canonical.
- RoboTwin protocol not explicitly stated but consistent with Protocol A (single-task,
  50 clean demos/task). 50 tasks evaluated.

---

## RTC

**Paper**: [arxiv 2506.07339](https://arxiv.org/abs/2506.07339) (Real-Time Chunking)
**Weights**: No public model weights. RTC is an inference algorithm applied on top of
  existing diffusion/flow-based VLA checkpoints (e.g. π0.5). Kinetix BC checkpoints
  are trained locally from the RTC repo.
**GitHub**: [real-time-chunking/rtc-kinetix](https://github.com/real-time-chunking/rtc-kinetix) (Kinetix experiments)

RTC is an **inference-time algorithm** for diffusion/flow-based VLAs, not a standalone model.
It is evaluated exclusively on:
- **Kinetix** (12 dynamic tasks, state-based, force control)
- Real-world bimanual manipulation (on top of π0.5)

### CALVIN / LIBERO / SimplerEnv / RoboTwin

Not reported. RTC does not evaluate on these benchmarks.

---

## Notes on Evaluation Protocols

### LIBERO
- **Per-suite finetuned** (most models): separate model per suite. Higher scores.
- **Shared weights** (X-VLA): single model for all suites. More challenging.
- Standard: 4-suite average (Spatial, Object, Goal, 10). 50 episodes/task.

### CALVIN
- Standard: ABC→D split, 1000 eval chains. Metric = avg completed subtasks (0–5).

### SimplerEnv
- Three independent dimensions: Google Robot VM, Google Robot VA, WidowX VM.
- **3-task standard** for Google Robot: Pick Coke Can, Move Near, Open/Close Drawer.
  Some papers report a 4th task (Place Apple in Drawer) — check task count before comparing.
- WidowX: 4-task standard (Spoon, Carrot, Block, Eggplant).

### RoboTwin 2.0
- **Protocol A** (official): Single-task, 50 clean demos/task. Hard = OOD.
- **Protocol B** (Motus-style): Multi-task, 50 clean + 500 DR demos/task. Hard = in-distribution.
- Protocols are **not comparable**. Always check which protocol was used.
- v1 and v2 are separate benchmarks — do not mix.
