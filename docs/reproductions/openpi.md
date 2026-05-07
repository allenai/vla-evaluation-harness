# openpi — Reproduction Report

Physical Intelligence's open-source VLA models. [GitHub](https://github.com/Physical-Intelligence/openpi) |
[Paper (Pi0)](https://arxiv.org/abs/2410.24164) | [Paper (Pi0.5)](https://arxiv.org/abs/2504.16054) | 3B params (PaliGemma).

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **97.7%** | 96.9% | Reproduced |
| CALVIN | — | — | No checkpoint |
| SimplerEnv | — | — | No checkpoint / no self-reported score |

### LIBERO

| | |
|---|---|
| **Checkpoint** | `s3://openpi-assets/checkpoints/pi05_libero` (official Pi0.5) |
| **Server config** | [`configs/model_servers/pi0/libero.yaml`](../../configs/model_servers/pi0/libero.yaml) |
| **Benchmark config** | [`configs/benchmarks/libero/all.yaml`](../../configs/benchmarks/libero/all.yaml) |
| **Results** | [`data/pi05-libero/`](data/pi05-libero/) |

4 suites × 10 tasks × 50 episodes. Pi0.5 fine-tuned (`pi05_libero`).

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 98.0% | 98.8% |
| Object | 99.6% | 98.2% |
| Goal | 98.6% | 98.0% |
| Long | 94.6% | 92.4% |
| **Average** | **97.7%** | **96.9%** |

Pipeline audit: All 18 items match. No discrepancies.
- Uses `states` (from `raw_obs`), 8D `[pos3, axisangle3, gripper2]`.
- `send_wrist_image=True`, `send_state=True`, `image_resolution=224`.
- OpenPI handles all normalization/denormalization internally via its config system.
- Our reproduced score (97.7%) exceeds reported (96.9%), within normal variance.

### CALVIN / SimplerEnv

No checkpoints publicly available in openpi. Available checkpoints are:
- Base models: `pi0_base`, `pi0_fast_base`, `pi05_base` (OXE pretrained)
- Fine-tuned: `pi0_fast_droid`, `pi0_droid`, `pi0_aloha_*` (3 variants), `pi05_libero`, `pi05_droid`

`pi0_base` was pretrained on OXE (10k+ hours, including Bridge/WidowX data), so it could be fine-tuned for SimplerEnv in principle. However:
- Physical Intelligence has never self-reported SimplerEnv scores, nor released a Bridge/WidowX fine-tuned checkpoint.
- Third-party LoRA fine-tunes exist (`HaomingSong/openpi0-bridge-lora`, from the SimplerEnv-OpenVLA project — **not** an official Pi checkpoint), but others have failed to reproduce their reported scores ([SimplerEnv-OpenVLA #13](https://github.com/DelinQu/SimplerEnv-OpenVLA/issues/13), [#28](https://github.com/DelinQu/SimplerEnv-OpenVLA/issues/28): 0% or 30-40% success).
- A user fine-tuned `pi05_base` on Bridge data (80K steps, 32×H100) and evaluated on SimplerEnv — results were below expectations ([openpi #799](https://github.com/Physical-Intelligence/openpi/issues/799)).
- `pi0_fast_libero` is a different, lower-performing variant from `pi05_libero`.

## Configuration Notes

- openpi uses its own config/checkpoint system — not standard HF `from_pretrained`. Requires `openpi` package + GCS/S3 access.
- LIBERO eval uses 3rd-person image + wrist image + proprioceptive state (8-dim).
- `chunk_size=10`, `action_ensemble="newest"`.