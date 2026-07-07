# LeRobot Bridge Reproduction Report

Any 🤗 [LeRobot](https://github.com/huggingface/lerobot) `PreTrainedPolicy` checkpoint served
through [`configs/model_servers/lerobot/`](../../configs/model_servers/lerobot/), pinned at
lerobot [v0.6.0](https://github.com/huggingface/lerobot/releases/tag/v0.6.0). Published numbers
come from LeRobot's policy docs at the same tag.

## Results Summary

| Model | Suite | Reproduced | Reported | Verdict |
|-------|-------|:----------:|:--------:|:-------:|
| π₀.₅ | LIBERO Object | **100%** (100 eps) | 99.0 | Reproduced |
| GR00T N1.7 | LIBERO Object | **99.0%** (100 eps) | 81 | Reproduced (above reported) |
| MolmoAct2 | LIBERO Goal | **97.0%** (100 eps) | 98.0 | Reproduced |
| VLA-JEPA | LIBERO-10 | **96.0%** (100 eps) | 93.0 | Reproduced |
| LingBot-VA | LIBERO-10 | 10/10 (10 eps) | none published | Validated, no reference |
| FastWAM | LIBERO-10 | 0% | 94.0 | Upstream defect (see below) |

Protocol for the 100-episode rows: 10 tasks × 10 episodes, `seed: 7`, model server on one H100
GPU, benchmark in the standard `libero:latest` container. `num_steps_wait: 10` except MolmoAct2
(50, per its LeRobot doc). Per-episode artifacts were not persisted for these runs; per-task
tables below are from the run summaries.

### π₀.₅: LIBERO Object

| | |
|---|---|
| **Checkpoint** | `lerobot/pi05_libero_finetuned` |
| **Server config** | [`configs/model_servers/lerobot/pi05_libero.yaml`](../../configs/model_servers/lerobot/pi05_libero.yaml) |
| **Benchmark config** | [`configs/benchmarks/libero/object.yaml`](../../configs/benchmarks/libero/object.yaml) |

100/100 success (LeRobot reports 99.0, OpenPI 98.2). Run on an H100 at bridge integration time;
re-validated at the v0.6.0 pin with a 10-episode check (10/10).

### GR00T N1.7: LIBERO Object

| | |
|---|---|
| **Checkpoint** | `nvidia/gr00t17-lerobot-libero_object-640` (LeRobot's eval-ready conversion) |
| **Server config** | [`configs/model_servers/lerobot/groot_n17.yaml`](../../configs/model_servers/lerobot/groot_n17.yaml) |
| **Benchmark config** | 10 tasks × 10 episodes on `libero_object` |

The checkpoint bakes the converter machine's local `base_model_path`; the config overrides it
back to `nvidia/GR00T-N1.7-3B` (gated behind `nvidia/Cosmos-Reason2-2B` access on HF).

| Task | Reproduced |
|------|:----------:|
| alphabet soup in basket | 100% |
| cream cheese in basket | 100% |
| salad dressing in basket | 100% |
| bbq sauce in basket | 100% |
| ketchup in basket | 100% |
| tomato sauce in basket | 90% |
| butter in basket | 100% |
| milk in basket | 100% |
| chocolate pudding in basket | 100% |
| orange juice in basket | 100% |
| **Average** | **99.0%** (reported 81) |

### MolmoAct2: LIBERO Goal

| | |
|---|---|
| **Checkpoint** | `allenai/MolmoAct2-LIBERO` (original-format HF release, loaded via `checkpoint_path` + `norm_tag=libero`) |
| **Server config** | [`configs/model_servers/lerobot/molmoact2_libero.yaml`](../../configs/model_servers/lerobot/molmoact2_libero.yaml) |
| **Benchmark config** | 10 tasks × 10 episodes on `libero_goal`, `num_steps_wait: 50` |

| Task | Reproduced |
|------|:----------:|
| open the middle drawer | 100% |
| bowl on the stove | 100% |
| wine bottle on top of cabinet | 100% |
| open top drawer, bowl inside | 90% |
| bowl on top of cabinet | 100% |
| push plate to front of stove | 100% |
| cream cheese in the bowl | 100% |
| turn on the stove | 80% |
| bowl on the plate | 100% |
| wine bottle on the rack | 100% |
| **Average** | **97.0%** (reported 98.0) |

### VLA-JEPA: LIBERO-10 (Long)

| | |
|---|---|
| **Checkpoint** | `lerobot/VLA-JEPA-LIBERO` |
| **Server config** | [`configs/model_servers/lerobot/vla_jepa_libero.yaml`](../../configs/model_servers/lerobot/vla_jepa_libero.yaml) |
| **Benchmark config** | 10 tasks × 10 episodes on `libero_10` |

| Task | Reproduced |
|------|:----------:|
| alphabet soup + tomato sauce in basket | 100% |
| cream cheese box + butter in basket | 100% |
| turn on stove, moka pot on it | 100% |
| black bowl in bottom drawer, close it | 100% |
| both mugs on plates | 100% |
| book in caddy back compartment | 100% |
| mug on plate + pudding right of plate | 100% |
| alphabet soup + cream cheese box in basket | 100% |
| both moka pots on stove | 70% |
| yellow/white mug in microwave, close it | 90% |
| **Average** | **96.0%** (reported 93.0) |

### LingBot-VA: LIBERO-10 (Long)

`lerobot/lingbot_va_libero_long`, 10 tasks × 1 episode: 10/10. LeRobot publishes no score for
this checkpoint, so there is no reference to reproduce against. Inference must run through the
bridge's `use_select_action` path: the policy feeds executed-step keyframes back into its KV
cache, which chunked inference starves (0/10 without it). ~21 s per 16-step chunk refill on an
A100.

### FastWAM: upstream defect

`ZibinDong/fastwam_libero_uncond_2cam224` scores 0/10 through the bridge, and the defect is
upstream: `lerobot-eval` at v0.6.0 with the documented reproduction command also scores 0/20,
and the checkpoint cannot even load without overriding its baked-in local
`tokenizer_model_id: "."`. The published 94.0 on LIBERO-10 is not reproducible at the tag.
