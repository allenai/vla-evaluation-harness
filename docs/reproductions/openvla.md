# OpenVLA — Reproduction Report

Open-source VLA model. [GitHub](https://github.com/openvla/openvla) |
[Paper](https://arxiv.org/abs/2406.09246) | 7B params (Prismatic VLM + action head).

## Results Summary

| Benchmark | Reproduced | Reported | Verdict |
|-----------|:----------:|:--------:|:-------:|
| LIBERO | **76.2%** | 76.5% | Reproduced |

### LIBERO

| | |
|---|---|
| **Checkpoints** | Per-suite fine-tuned (LoRA), see below |
| **Server configs** | [`configs/model_servers/openvla/libero_*.yaml`](../../configs/model_servers/openvla/) |
| **Benchmark configs** | [`configs/benchmarks/libero/spatial.yaml`](../../configs/benchmarks/libero/spatial.yaml), etc. |
| **Results** | [`data/openvla-libero/`](data/openvla-libero/) |

4 suites × 10 tasks × 50 episodes = 2000 episodes. Per-suite fine-tuned checkpoints (LoRA).

| Suite | Checkpoint | Reproduced | Reported |
|-------|-----------|:----------:|:--------:|
| Spatial | `openvla/openvla-7b-finetuned-libero-spatial` | 86.6% | 84.7% |
| Object | `openvla/openvla-7b-finetuned-libero-object` | 86.6% | 88.4% |
| Goal | `openvla/openvla-7b-finetuned-libero-goal` | 79.0% | 79.2% |
| Long | `openvla/openvla-7b-finetuned-libero-10` | 52.8% | 53.7% |
| **Average** | | **76.2%** | **76.5%** |

Per-suite reported scores from Table I of the paper (arxiv 2406.09246).

## Pipeline Notes

Matching the reference eval required aligning image preprocessing and env setup.
The reference (`openvla/experiments/robot/`) applies JPEG roundtrip, Lanczos3
resize to 224×224, center crop (scale=0.9), and `env.seed(0)`.

- **Center crop** — isolated impact ~3pp (73.3% → 76.2%).
- **JPEG roundtrip, env_seed=0, Lanczos resize** — applied together; individual
  contributions not isolated. Combined with center crop: 71.0% → 76.2%.

Without any of these fixes, the average was 71.0%.

## Configuration Notes

- `chunk_size=1` (no action chunking).
- Per-suite `unnorm_key` required: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.
- Gripper: RLDS format `[0=close, 1=open]` converted to robosuite `[-1=open, +1=close]` via `-(2x - 1)`.
- Action output: 7D `[pos3, axisangle3, gripper]`.
- Each suite requires a different model checkpoint — cannot evaluate all suites with a single model.
- 7B model requires ~15 GB GPU memory (bfloat16).
- `transformers==4.40.1` required (trust_remote_code model).
- Single image input (no wrist camera, no proprioceptive state).
