# Dexbotic — Reproduction Report

DB-CogACT: CogACT 7B fine-tuned by [Dexbotic](https://github.com/Dexmal/dexbotic).
Paper: [arxiv 2510.23511](https://arxiv.org/abs/2510.23511) (DexboticVLM) / [arxiv 2411.19650](https://arxiv.org/abs/2411.19650) (CogACT).

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **94.7%** | 94.9% | Reproduced |
| CALVIN ABC→D | **4.02** | 4.06 | Reproduced |
| SimplerEnv WidowX | **63.5%** | 69.5% | Approximate (−6pp) |

### LIBERO

| | |
|---|---|
| **Checkpoint** | `Dexmal/libero-db-cogact` (official) |
| **Server config** | [`configs/model_servers/db_cogact/libero.yaml`](../../configs/model_servers/db_cogact/libero.yaml) |
| **Benchmark config** | [`configs/benchmarks/libero/all.yaml`](../../configs/benchmarks/libero/all.yaml) |
| **Results** | [`data/dbcogact-libero/`](data/dbcogact-libero/) |

4 suites × 10 tasks × 50 episodes = 2000 episodes. Seed=7.

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 93.8% | 93.8% |
| Object | 98.4% | 97.8% |
| Goal | 96.0% | 96.2% |
| Long | 90.8% | 91.8% |
| **Average** | **94.7%** | **94.9%** |

Pipeline audit: All 18 verification items match. No discrepancies found.

### CALVIN (ABC→D)

| | |
|---|---|
| **Checkpoint** | `Dexmal/calvin-db-cogact` (official) |
| **Server config** | [`configs/model_servers/db_cogact/calvin.yaml`](../../configs/model_servers/db_cogact/calvin.yaml) |
| **Benchmark config** | [`configs/benchmarks/calvin/eval.yaml`](../../configs/benchmarks/calvin/eval.yaml) |
| **Results** | [`data/dbcogact-calvin/`](data/dbcogact-calvin/) |

1000 sequences × 5 chained subtasks, max 360 steps/subtask.

| Task chain | Reproduced | Reported |
|-----------|:----------:|:--------:|
| 1/5 | 93.3% | 93.5% |
| 2/5 | 86.3% | 86.7% |
| 3/5 | 80.5% | 80.3% |
| 4/5 | 74.4% | 76.0% |
| 5/5 | 67.8% | 69.8% |
| **Avg Length** | **4.02** | **4.06** |

Pipeline audit: All items match. No discrepancies.

### SimplerEnv — WidowX VM

| | |
|---|---|
| **Checkpoint** | `Dexmal/simpler-db-cogact` (official) |
| **Server config** | [`configs/model_servers/db_cogact/simpler.yaml`](../../configs/model_servers/db_cogact/simpler.yaml) |
| **Benchmark config** | [`configs/benchmarks/simpler/widowx_vm.yaml`](../../configs/benchmarks/simpler/widowx_vm.yaml) |
| **Results** | [`data/dexbotic/`](data/dexbotic/) |

4 tasks × 24 episodes. `chunk_size=5` (matching reference `replan_step`).

| Task | Reproduced | Reported |
|------|:----------:|:--------:|
| Stack | 41.7% | — |
| Carrot | 33.3% | — |
| Spoon | 83.3% | — |
| Eggplant | 95.8% | — |
| **Average** | **63.5%** | **69.5%** |

~6pp gap. Per-task reported scores not available (only overall 69.5% on HF).

### RoboTwin 2.0

| | |
|---|---|
| **Checkpoint** | `Dexmal/robotwin-db-cogact/{task}` (official, per-task) |
| **Server config** | [`configs/model_servers/db_cogact/robotwin2.yaml`](../../configs/model_servers/db_cogact/robotwin2.yaml) |
| **Benchmark config** | [`configs/benchmarks/robotwin/eval.yaml`](../../configs/benchmarks/robotwin/eval.yaml) |
| **Results** | — (not yet evaluated) |

Status: Not yet evaluated. Pipeline verified — code-complete, config adjustments needed (test_num: 1→100, skip_expert_check: true→false).
Reported: 58.5% avg on 4 tasks (adjust_bottle 99%, grab_roller 89%, place_empty_cup 28%, place_phone_stand 18%).

## Configuration Notes

- Architecture: DexboticVLM (Qwen2.5 + CLIP + CogACT DiT action head, cfg_scale=1.5, num_ddim_steps=10).
- Separate checkpoint per benchmark — no shared-weights model.
- LIBERO uses per-suite `chunk_size_map`: Spatial=12, Object=16, Goal=16, Long=15.
- `use_text_template: true` for CALVIN, LIBERO, SimplerEnv.
- RoboTwin uses per-task subdirectories and 3 cameras (head, left, right).
- Requires `transformers==4.46.3` (v5.x causes meta device issue).
- Action indexing bug fix: `actions[0]` → full chunk. See [common-pitfalls.md](common-pitfalls.md#8-serialization--data-bugs).
- numpy bool serialization fix. See [common-pitfalls.md](common-pitfalls.md#8-serialization--data-bugs).
- SimplerEnv image resize (224×224) handled internally by `process_images()`.

