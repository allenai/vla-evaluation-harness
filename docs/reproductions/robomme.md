# MME-VLA — Reproduction Report (RoboMME)

Memory-augmented VLA models from the RoboMME benchmark. [GitHub](https://github.com/RoboMME/robomme_policy_learning) |
[Paper](https://arxiv.org/abs/2603.04639) | [Weights](https://huggingface.co/Yinpei/mme_vla_suite) | pi0.5-based (JAX).

## Results Summary

| Variant | Memory Type | Reproduced | Reported | Verdict |
|---------|-------------|:----------:|:--------:|:-------:|
| π₀.5 baseline | None | **25.5%** (counting) | 22.72% (counting) | ✅ |
| FrameSamp+Modul | Perceptual | — | 44.51% | ⬜ |
| FrameSamp+Context | Perceptual | — | 30.68% | ⬜ |
| FrameSamp+Expert | Perceptual | — | 36.25% | ⬜ |
| TokenDrop+Modul | Perceptual | — | 38.04% | ⬜ |
| TokenDrop+Context | Perceptual | — | 34.50% | ⬜ |
| TokenDrop+Expert | Perceptual | — | 34.86% | ⬜ |
| SimpleSG+QwenVL | Symbolic | — | 29.00% | ⬜ |
| GroundSG+QwenVL | Symbolic | — | 32.70% | ⬜ |
| TTT+Context | Recurrent | — | 22.28% | ⬜ |
| TTT+Modul | Recurrent | — | 21.97% | ⬜ |
| TTT+Expert | Recurrent | — | 22.35% | ⬜ |
| RMT+Context | Recurrent | — | 19.46% | ⬜ |
| RMT+Modul | Recurrent | — | 20.17% | ⬜ |
| RMT+Expert | Recurrent | — | 18.15% | ⬜ |

Reported scores are averages over 9 runs (3 checkpoints × 3 seeds, 50 episodes/task).

### π₀.5 Baseline (no memory)

| | |
|---|---|
| **Checkpoint** | [`Yinpei/pi05_baseline`](https://huggingface.co/Yinpei/pi05_baseline) |
| **Server config** | [`configs/model_servers/mme_vla/pi05_baseline.yaml`](../../configs/model_servers/mme_vla/pi05_baseline.yaml) |
| **Benchmark config** | [`configs/benchmarks/robomme/counting.yaml`](../../configs/benchmarks/robomme/counting.yaml) |
| **Results** | [`results/robomme-pi05-baseline/`](../../results/robomme-pi05-baseline/) |

Counting suite: 4 tasks × 50 episodes = 200 total. No memory — ignores conditioning video.

| Task | Reproduced | Reported |
|------|:----------:|:--------:|
| BinFill | 22.0% | — |
| PickXtimes | 36.0% | — |
| SwingXtimes | 40.0% | — |
| StopCube | 4.0% | — |
| **Average** | **25.5%** | **22.72%** |

Single run (1 seed). Paper reports 9-run average (3 ckpts × 3 seeds).
Difference (+2.8pp) is not statistically significant (z=0.94, p=0.348). Reproduced.

### FrameSamp+Modul (best non-oracle)

| | |
|---|---|
| **Checkpoint** | [`Yinpei/mme_vla_suite/perceptual-framesamp-modul`](https://huggingface.co/Yinpei/mme_vla_suite) |
| **Server config** | [`configs/model_servers/mme_vla/framesamp_modul.yaml`](../../configs/model_servers/mme_vla/framesamp_modul.yaml) |
| **Benchmark config** | [`configs/benchmarks/robomme/eval.yaml`](../../configs/benchmarks/robomme/eval.yaml) |
| **Results** | `data/mme-vla-robomme/` (pending) |

Uses conditioning video via `add_buffer` to populate perceptual memory before inference.

## Configuration Notes

- **Framework**: All models use the `mme_vla_suite` package (extended OpenPI, JAX-based). Requires `mme-vla-suite` pip dependency, NOT the standard `openpi` package.
- **State truncation**: RoboMME benchmark sends 9D state (8 joint + 1 gripper). Models expect 8D. The model server truncates `state[:8]` automatically.
- **Video history**: Conditioning video (motion planning demo) is sent on the first observation only via `video_history`. Memory-augmented models convert this to an `add_buffer` call. The baseline ignores it.
- **Prompt casing**: Task descriptions are lowercased to match the upstream training convention.
- **Action space**: 8D absolute joint angles (7 arm joints + 1 gripper). `chunk_size=10`.
- **Image resolution**: 224×224 (SigLIP2 encoder input).
- **Evaluation protocol**: Paper uses 50 episodes per task, max 1300 steps, `test` split.

## Per-Suite Reported Scores (%)

From Table 3 of the paper (averaged over 9 runs):

| Variant | Counting | Permanence | Reference | Imitation | **Average** |
|---------|:--------:|:----------:|:---------:|:---------:|:-----------:|
| π₀.5 baseline | 22.72 | 13.67 | 11.39 | 23.94 | **17.93** |
| FrameSamp+Modul | 47.89 | 36.00 | 41.83 | 52.33 | **44.51** |
| TokenDrop+Modul | 44.06 | 29.00 | 33.50 | 45.61 | **38.04** |
| GroundSG+QwenVL | 53.89 | 18.67 | 26.33 | 31.89 | **32.70** |

## Known Limitations

- Some recurrent checkpoints (TTT, RMT variants) may not be fully released on HuggingFace yet.
- Video history frames do not include per-frame proprioceptive state; zero-filled instead. This matches the default config (`use_state_emb=False`) but may slightly affect variants trained with state embeddings.
- The `mme_vla_suite` package pins a specific JAX version with CUDA 12. Ensure the host GPU driver is compatible.
