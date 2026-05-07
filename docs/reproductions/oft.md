# OpenVLA-OFT — Reproduction Report

OpenVLA with Optimized Fine-Tuning. [GitHub](https://github.com/moojink/openvla-oft) |
[Paper](https://arxiv.org/abs/2502.19645) | 7B params (OpenVLA + parallel decoding + action chunking).

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **96.7%** | 97.1% | Reproduced |

### LIBERO

| | |
|---|---|
| **Checkpoint** | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` (joint, all 4 suites) |
| **Server configs** | [`configs/model_servers/oft/libero_joint_*.yaml`](../../configs/model_servers/oft/) |
| **Benchmark configs** | [`configs/benchmarks/libero/spatial.yaml`](../../configs/benchmarks/libero/spatial.yaml), etc. |
| **Results** | [`data/oft-libero-joint/`](data/oft-libero-joint/) |

4 suites × 10 tasks × 50 episodes = 2000 episodes. Joint checkpoint (300k steps, all suites combined).

| Suite | Reproduced | Reported |
|-------|:----------:|:--------:|
| Spatial | 96.6% | 97.7% |
| Object | 97.6% | 98.0% |
| Goal | 97.4% | 96.1% |
| Long | 95.4% | 95.3% |
| **Average** | **96.7%** | **96.8%** |

Reported scores from Table XIV of the paper (arxiv 2502.19645), joint checkpoint row.

### Per-suite vs Joint checkpoint

Per-suite checkpoints (50k steps each) produced significantly lower scores on
Goal (45.8%) and Long (55.6%), likely due to insufficient training. The joint
checkpoint (300k steps) matches the paper's reported scores.

## Pipeline Notes

Key fix: **`quat_to_axisangle` antipodal normalization** — our implementation
normalized quaternions with `w < 0` by flipping the sign (antipodal mapping),
producing axis-angle values in `[0, π]`. The OFT reference (robosuite-style)
does not flip, producing values in `[0, 2π]`. Since training data was generated
with the robosuite convention, our antipodal version produced out-of-distribution
proprio states. Impact: Goal 83.4% → 97.4%, Long 55.8% → 95.4%.

This fix is OFT-specific (`quat_no_antipodal=True` in `get_observation_params`).
Other models (OpenVLA, Pi0, X-VLA) are unaffected — either they don't use
proprio state (OpenVLA) or they were reproduced successfully with the antipodal
version.

## Configuration Notes

- Joint checkpoint: single model for all 4 LIBERO suites, `unnorm_key` set per suite.
- `chunk_size=8` (matches `NUM_ACTIONS_CHUNK`; config previously said 10).
- `center_crop=True` handled internally by OFT's `prepare_images_for_vla()`.
- JPEG roundtrip + Lanczos3 resize also handled internally by OFT.
- `num_images_in_input=2` (3rd-person + wrist camera).
- `use_proprio=True` (8D state: pos3 + axisangle3 + gripper_qpos2).
- `env_seed=0` (matching OpenVLA/OFT reference).
