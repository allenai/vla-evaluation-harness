# DB-CogACT Reproduction Report

DB-CogACT (dexbotic fine-tuned CogACT 7B) evaluated across four benchmarks.

---

## Model Info

| Field | Value |
|-------|-------|
| **Model** | DB-CogACT (dexbotic fine-tuned CogACT) |
| **Architecture** | CogACT 7B (diffusion action head, `cfg_scale=1.5`, `num_ddim_steps=10`) |
| **Loading** | `CogACTForCausalLM.from_pretrained(..., torch_dtype=bfloat16, low_cpu_mem_usage=True, device_map={"": "cuda:0"})` |
| **Action ensemble** | Disabled |

### Common Code Modifications

**Action indexing bug fix (critical)**: `actions[0]` used only the first action of a chunk → ~60% success rate. Fixed to return `np.array(actions, dtype=np.float32)` for the full chunk. Affected all benchmarks.

**numpy bool serialization fix**: `json.dumps(default=str)` converted numpy `False` to string `"False"` (truthy in Python), inflating success rates. Fixed by normalizing `success` to Python `bool` in `ResultCollector.record()`.

**transformers version**: Requires `transformers==4.46.3` (v5.x causes meta device issue and `Qwen2Model` labels error).

---

## LIBERO

| Field | Value |
|-------|-------|
| **Status** | `complete` |
| **Date** | 2026-02-27 |
| **Harness commit** | [`e37fbe0`](../../commit/e37fbe0) |
| **Benchmark** | LIBERO (Spatial / Object / Goal / 10) — 10 tasks × 50 episodes × 4 suites = 2 000 episodes |
| **Hardware** | Model server: 1 × H100; Benchmark: separate CPU node |
| **Seed** | 7 |
| **Action space** | 7D (6D delta pose + 1D gripper), chunk sizes: Spatial 12 / Object 16 / Goal 16 / 10 15 |

### How to Reproduce

```bash
# 1. Start model server (max_batch_size=16 enables batch mode)
srun $SINGLE_GPU --pty \
  vla-eval serve --config configs/model_servers/dexbotic_cogact_libero.yaml

# 2. Run all 4 suites in parallel (50 shards + auto-merge)
./scripts/run_sharded.sh -c configs/libero_all.yaml -n 50
```

### Results

| Suite | Score | Reference | Diff | Verdict |
|-------|:-----:|:---------:|:----:|:-------:|
| Spatial | **95.2%** | 93.8% | +1.4 pp | ✅ Reproduced |
| Object | **98.6%** | 97.8% | +0.8 pp | ✅ Reproduced |
| Goal | **95.2%** | 96.2% | −1.0 pp | ✅ Reproduced |
| 10 | **89.6%** | 91.8% | −2.2 pp | ✅ Reproduced |

All suites reproduced within ±2.2 pp. Wall-clock: **~18 min** for 2 000 episodes (50 shards, 1 × H100, batch inference, ~47× throughput).

<details>
<summary>Per-task breakdown</summary>

**LIBERO-Spatial — 95.2%**

| Task | Rate | N |
|------|:----:|:---:|
| pick up the black bowl from table center and place it on the plate | 100.0% | 50/50 |
| pick up the black bowl on the ramekin and place it on the plate | 100.0% | 50/50 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 98.0% | 49/50 |
| pick up the black bowl next to the cookie box and place it on the plate | 98.0% | 49/50 |
| pick up the black bowl between the plate and the ramekin and place it on the plate | 96.0% | 48/50 |
| pick up the black bowl on the stove and place it on the plate | 96.0% | 48/50 |
| pick up the black bowl next to the plate and place it on the plate | 94.0% | 47/50 |
| pick up the black bowl next to the ramekin and place it on the plate | 92.0% | 46/50 |
| pick up the black bowl on the cookie box and place it on the plate | 92.0% | 46/50 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 86.0% | 43/50 |
| **Overall** | **95.2%** | **476/500** |

**LIBERO-Object — 98.6%**

| Task | Rate | N |
|------|:----:|:---:|
| pick up the alphabet soup and place it in the basket | 100.0% | 50/50 |
| pick up the bbq sauce and place it in the basket | 100.0% | 50/50 |
| pick up the butter and place it in the basket | 100.0% | 50/50 |
| pick up the chocolate pudding and place it in the basket | 100.0% | 50/50 |
| pick up the ketchup and place it in the basket | 100.0% | 50/50 |
| pick up the orange juice and place it in the basket | 100.0% | 50/50 |
| pick up the cream cheese and place it in the basket | 98.0% | 49/50 |
| pick up the salad dressing and place it in the basket | 98.0% | 49/50 |
| pick up the milk and place it in the basket | 96.0% | 48/50 |
| pick up the tomato sauce and place it in the basket | 94.0% | 47/50 |
| **Overall** | **98.6%** | **493/500** |

**LIBERO-Goal — 95.2%**

| Task | Rate | N |
|------|:----:|:---:|
| open the middle drawer of the cabinet | 100.0% | 50/50 |
| push the plate to the front of the stove | 100.0% | 50/50 |
| put the bowl on the stove | 100.0% | 50/50 |
| put the bowl on top of the cabinet | 100.0% | 50/50 |
| put the wine bottle on top of the cabinet | 100.0% | 50/50 |
| turn on the stove | 100.0% | 50/50 |
| put the bowl on the plate | 94.0% | 47/50 |
| put the cream cheese in the bowl | 92.0% | 46/50 |
| put the wine bottle on the rack | 86.0% | 43/50 |
| open the top drawer and put the bowl inside | 80.0% | 40/50 |
| **Overall** | **95.2%** | **476/500** |

**LIBERO-10 — 89.6%**

| Task | Rate | N |
|------|:----:|:---:|
| put both the cream cheese box and the butter in the basket | 100.0% | 50/50 |
| put the yellow and white mug in the microwave and close it | 100.0% | 50/50 |
| pick up the book and place it in the back compartment of the caddy | 98.0% | 49/50 |
| put the black bowl in the bottom drawer of the cabinet and close it | 98.0% | 49/50 |
| turn on the stove and put the moka pot on it | 98.0% | 49/50 |
| put both the alphabet soup and the cream cheese box in the basket | 90.0% | 45/50 |
| put both the alphabet soup and the tomato sauce in the basket | 84.0% | 42/50 |
| put the white mug on the left plate and put the yellow and white mug on the right plate | 84.0% | 42/50 |
| put both moka pots on the stove | 72.0% | 36/50 |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | 72.0% | 36/50 |
| **Overall** | **89.6%** | **448/500** |

</details>

<details>
<summary>Timing breakdown</summary>

| Suite | Episodes | Wall-clock | Sum of `elapsed_sec` | Avg per episode |
|-------|:--------:|:----------:|:--------------------:|:---------------:|
| Spatial | 500 | 4.3 min | 187.8 min | 22.5 s |
| Object | 500 | 3.9 min | 179.0 min | 21.5 s |
| Goal | 500 | 3.8 min | 149.0 min | 17.9 s |
| 10 | 500 | 7.7 min | 320.8 min | 38.5 s |
| **Total** | **2 000** | **~18 min** | **836.6 min** | **25.1 s** |

836.6 min sequential → ~18 min parallel (~47× throughput) via batch inference (`max_batch_size=16`).

</details>

### Discussion

- All 4 suites reproduced within ±2.2 pp of reference scores.
- LIBERO-10 delta (−2.2 pp): hardest tasks — "put both moka pots on the stove" (72%) and "put the white mug on the plate..." (72%) — account for most of the gap.
- numpy bool bug: previous results (100% on 3 suites) were inflated. Fixed in this run.

---

## CALVIN

| Field | Value |
|-------|-------|
| **Status** | `complete` |
| **Date** | 2026-02-28 |
| **Harness commit** | [`703e5aa`](../../commit/703e5aa) |
| **Benchmark** | CALVIN (ABC→D, 1000 sequences × 5 chained subtasks) |
| **Hardware** | Model server: 1 × H100; Benchmark: separate CPU node |
| **Action space** | 7D (6D delta pose + 1D gripper), chunk size 7, max 360 steps/subtask |

### How to Reproduce

```bash
# 1. Start model server
srun $SINGLE_GPU --pty \
  vla-eval serve --config configs/model_servers/dexbotic_cogact_calvin.yaml

# 2. Run sharded evaluation (16 parallel containers + auto-merge)
./scripts/run_sharded.sh -c configs/calvin_eval.yaml -n 16
```

### Results

| Step | Score | Reference | Diff | Verdict |
|------|:-----:|:---------:|:----:|:-------:|
| 1/5 | **93.3%** | 93.5% | −0.2 pp | ✅ Reproduced |
| 2/5 | **86.3%** | 86.7% | −0.4 pp | ✅ Reproduced |
| 3/5 | **81.5%** | 80.3% | +1.2 pp | ✅ Reproduced |
| 4/5 | **75.6%** | 76.0% | −0.4 pp | ✅ Reproduced |
| 5/5 | **68.4%** | 69.8% | −1.4 pp | ✅ Reproduced |
| **Avg Len** | **4.051** | **4.063** | **−0.012** | **✅ Reproduced** |

All steps reproduced within ±1.4 pp. Wall-clock: **~33 min** for 1000 sequences (16 shards, ~16× throughput).

### Discussion

- Avg Len 4.051 vs reference 4.063 (−0.012).
- 5/5 delta (−1.4 pp) within expected variance for 1000 sequences.
- PyBullet CPU rendering is the per-container bottleneck.

---

## SimplerEnv

| Field | Value |
|-------|-------|
| **Status** | `complete` |
| **Date** | 2026-03-05 |
| **Harness commit** | [`cb486d6`](../../commit/cb486d6) |
| **Benchmark** | SimplerEnv (WidowX, 4 tasks × 24 episodes × 3 seeds = 288 episodes) |
| **Hardware** | Model server: 1 × H100; Benchmark: separate GPU node (Vulkan) |
| **Action space** | 7-DOF delta pose + gripper, chunk size 5, max 120 steps |

### How to Reproduce

```bash
# 1. Start model server
srun $SINGLE_GPU --pty \
  vla-eval serve --config configs/model_servers/dexbotic_cogact_simpler.yaml

# 2. Run 3-seed evaluation (16 shards × 3 seeds + auto-merge)
scripts/run_simpler_seeds.sh
```

### Benchmark-Specific Fixes

**`is_done()` semantics (critical)**: In SimplerEnv, `terminated` is a transient success signal, not a final verdict. Fixed to only end episode on `truncated=True`. Impact: Stack Green Block 75% → 29.2% (matches reference).

**Image resize interpolation**: Switched from `PIL.BILINEAR` to `cv2.INTER_AREA` to match reference preprocessing.

**Vulkan in Docker**: Requires `NVIDIA_DRIVER_CAPABILITIES=all` and host Vulkan ICD mount.

### Results (3-Seed Average)

| Task | Score | Reference | Diff | Verdict |
|------|:-----:|:---------:|:----:|:-------:|
| Stack Green Block | **25.00%** | 29.17% | −4.17 pp | ⚠️ Lower |
| Put Carrot on Plate | **72.22%** | 65.28% | +6.94 pp | ⚠️ Higher |
| Put Spoon on Towel | **94.44%** | 87.50% | +6.94 pp | ⚠️ Higher |
| Put Eggplant in Basket | **97.22%** | 95.83% | +1.39 pp | ✅ Close |
| **Average** | **72.22%** | **69.45%** | **+2.77 pp** | **✅** |

Wall-clock: **~8.5 min** total for 288 episodes (3 seeds × 16 shards, ~12× throughput).

<details>
<summary>Per-seed breakdown</summary>

| Task | Seed 0 | Seed 2 | Seed 4 | 3-Seed Avg |
|------|:------:|:------:|:------:|:----------:|
| Stack Green Block | 29.2% | 25.0% | 20.8% | 25.00% |
| Put Carrot on Plate | 62.5% | 79.2% | 75.0% | 72.22% |
| Put Spoon on Towel | 95.8% | 95.8% | 91.7% | 94.44% |
| Put Eggplant in Basket | 95.8% | 100.0% | 95.8% | 97.22% |
| **Overall** | **70.8%** | **75.0%** | **70.8%** | **72.22%** |

</details>

### Discussion

- Reproduced within expected variance (+2.77 pp overall). Exact match not achievable — CogACT's diffusion head uses stochastic noise without seed control.
- Seed variance is significant (Stack Green Block: 20.8%–29.2%), confirming multi-seed averaging is essential.

---

## Changelog

| Date | Benchmark | Change |
|------|-----------|--------|
| 2026-03-05 | SimplerEnv | Re-run with `cv2.INTER_AREA` resize; 3-seed evaluation |
| 2026-02-28 | CALVIN | Re-evaluation with 16-shard parallel run |
| 2026-02-27 | LIBERO | Re-evaluation with batch server, 50-shard parallel, numpy bool fix |
| 2026-02-22 | All | Initial evaluations |
