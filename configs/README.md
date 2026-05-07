# Evaluation Configs

Quick reference for running evaluations. For reproduction scores, see [docs/reproductions/](../docs/reproductions/).

## Quick Start

```bash
# 1. Start a model server (terminal 1)
vla-eval serve --config configs/model_servers/xvla/libero.yaml

# 2. Run an evaluation (terminal 2)
vla-eval run --config configs/benchmarks/libero/all.yaml
```

## Benchmarks

Benchmark names link to their config directory with available YAML files and usage details.

| Benchmark | | Paper | Docker Image | Python | Description |
|-----------|:--:|:-----:|:------------:|:------:|-------------|
| [LIBERO](benchmarks/libero/) | ![✓](https://img.shields.io/badge/✓-teal) | [2310.07899](https://arxiv.org/abs/2310.07899) | [`libero`](https://ghcr.io/allenai/vla-evaluation-harness/libero) | 3.8 | Tabletop manipulation, 4 suites (MuJoCo) |
| [LIBERO-Pro](benchmarks/libero_pro/) | ![◇](https://img.shields.io/badge/◇-blue) | [2310.07899](https://arxiv.org/abs/2310.07899) | [`libero-pro`](https://ghcr.io/allenai/vla-evaluation-harness/libero-pro) | 3.8 | Extended harder tasks |
| [LIBERO-Plus](benchmarks/libero_plus/) | ![✓](https://img.shields.io/badge/✓-teal) | [2310.07899](https://arxiv.org/abs/2310.07899) | [`libero-plus`](https://ghcr.io/allenai/vla-evaluation-harness/libero-plus) | 3.8 | Extended task set |
| [LIBERO-Mem](benchmarks/libero_mem/) | ![◇](https://img.shields.io/badge/◇-blue) | [2310.07899](https://arxiv.org/abs/2310.07899) | [`libero-mem`](https://ghcr.io/allenai/vla-evaluation-harness/libero-mem) | 3.8 | Memory-augmented tasks |
| [SimplerEnv](benchmarks/simpler/) | ![✓](https://img.shields.io/badge/✓-teal) | [2405.05941](https://arxiv.org/abs/2405.05941) | [`simpler`](https://ghcr.io/allenai/vla-evaluation-harness/simpler) | 3.10 | Google Robot + WidowX real2sim (SAPIEN) |
| [CALVIN](benchmarks/calvin/) | ![✓](https://img.shields.io/badge/✓-teal) | [2112.03227](https://arxiv.org/abs/2112.03227) | [`calvin`](https://ghcr.io/allenai/vla-evaluation-harness/calvin) | 3.8 | Chained 5-subtask sequences (PyBullet) |
| [RoboTwin](benchmarks/robotwin/) | ![◇](https://img.shields.io/badge/◇-blue) | [2409.02920](https://arxiv.org/abs/2409.02920) | [`robotwin`](https://ghcr.io/allenai/vla-evaluation-harness/robotwin) | 3.10 | Dual-arm manipulation (SAPIEN) |
| [ManiSkill2](benchmarks/maniskill2/) | ![◇](https://img.shields.io/badge/◇-blue) | [2302.04659](https://arxiv.org/abs/2302.04659) | [`maniskill2`](https://ghcr.io/allenai/vla-evaluation-harness/maniskill2) | 3.10 | Generalizable manipulation (SAPIEN) |
| [RoboMME](benchmarks/robomme/) | ![✓](https://img.shields.io/badge/✓-teal) | [2603.04639](https://arxiv.org/abs/2603.04639) | [`robomme`](https://ghcr.io/allenai/vla-evaluation-harness/robomme) | 3.11 | Multi-modal evaluation (MuJoCo) |
| [Kinetix](benchmarks/kinetix/) | ![◇](https://img.shields.io/badge/◇-blue) | [2410.23208](https://arxiv.org/abs/2410.23208) | [`kinetix`](https://ghcr.io/allenai/vla-evaluation-harness/kinetix) | 3.11 | Physics-based 2D manipulation (JAX) |
| [MolmoSpaces-Bench](benchmarks/molmospaces/) | ![✓](https://img.shields.io/badge/✓-teal) | [2603.16861](https://arxiv.org/abs/2603.16861) | [`molmospaces`](https://ghcr.io/allenai/vla-evaluation-harness/molmospaces) | 3.11 | Spatial reasoning (AI2-THOR) |
| [RLBench](benchmarks/rlbench/) | ![◇](https://img.shields.io/badge/◇-blue) | [1909.12271](https://arxiv.org/abs/1909.12271) | 🔒 `rlbench` | 3.8 | Vision-guided manipulation (CoppeliaSim) |
| [RoboCasa](benchmarks/robocasa/) | ![◇](https://img.shields.io/badge/◇-blue) | [2406.02523](https://arxiv.org/abs/2406.02523) | [`robocasa`](https://ghcr.io/allenai/vla-evaluation-harness/robocasa) | 3.11 | Kitchen manipulation (MuJoCo) |
| [VLABench](benchmarks/vlabench/) | ![◇](https://img.shields.io/badge/◇-blue) | [2502.09858](https://arxiv.org/abs/2502.09858) | [`vlabench`](https://ghcr.io/allenai/vla-evaluation-harness/vlabench) | 3.10 | Language-conditioned manipulation (SAPIEN) |
| [MIKASA-Robo](benchmarks/mikasa/) | ![◇](https://img.shields.io/badge/◇-blue) | [2502.07007](https://arxiv.org/abs/2502.07007) | [`mikasa-robo`](https://ghcr.io/allenai/vla-evaluation-harness/mikasa-robo) | 3.10 | Robot manipulation (MuJoCo) |
| [RoboCerebra](benchmarks/robocerebra/) | ![◇](https://img.shields.io/badge/◇-blue) | [2502.02853](https://arxiv.org/abs/2502.02853) | [`robocerebra`](https://ghcr.io/allenai/vla-evaluation-harness/robocerebra) | 3.8 | Cognitive manipulation (MuJoCo) |
| [BEHAVIOR-1K](benchmarks/behavior1k/) | ![◇](https://img.shields.io/badge/◇-blue) | [2403.09227](https://arxiv.org/abs/2403.09227) | 🔒 `behavior1k` | 3.10 | Household activities (OmniGibson) |

<sub>![✓](https://img.shields.io/badge/✓-teal) [reproduced](../docs/reproductions/) · ![◇](https://img.shields.io/badge/◇-blue) integrated, awaiting first reproduction · 🔒 license-restricted (local build only)</sub>

## Model Servers

Model names link to their server config directory. For reproduction scores, see [docs/reproductions/](../docs/reproductions/).

| Model | | Paper | Codebase | Supported Benchmarks | Reproduction |
|-------|:--:|:-----:|----------|---------------------|:------------:|
| [OpenVLA](model_servers/openvla/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2406.09246](https://arxiv.org/abs/2406.09246) | [openvla/openvla](https://github.com/openvla/openvla) | [LIBERO](model_servers/openvla/libero_spatial.yaml), [SimplerEnv GR](model_servers/openvla/simpler_google_robot.yaml) | [report](../docs/reproductions/openvla.md) |
| [π₀ / π₀-FAST](model_servers/pi0/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2410.24164](https://arxiv.org/abs/2410.24164) | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | [LIBERO](model_servers/pi0/libero.yaml) | [report](../docs/reproductions/openpi.md) |
| [GR00T N1.6](model_servers/groot/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2503.14734](https://arxiv.org/abs/2503.14734) | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | [LIBERO](model_servers/groot/libero.yaml), [SimplerEnv](model_servers/groot/simpler_widowx.yaml) | [report](../docs/reproductions/groot.md) |
| [OFT](model_servers/oft/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2502.19645](https://arxiv.org/abs/2502.19645) | [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | [LIBERO](model_servers/oft/libero_spatial.yaml) | [report](../docs/reproductions/oft.md) |
| [X-VLA](model_servers/xvla/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2510.10274](https://arxiv.org/abs/2510.10274) | [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | [LIBERO](model_servers/xvla/libero.yaml), [CALVIN](model_servers/xvla/calvin.yaml), [SimplerEnv](model_servers/xvla/simpler_widowx.yaml), [RoboTwin](model_servers/xvla/robotwin.yaml) | [report](../docs/reproductions/xvla.md) |
| [CogACT](model_servers/cogact/) | ![◇](https://img.shields.io/badge/◇-blue) | [2411.19650](https://arxiv.org/abs/2411.19650) | [microsoft/CogACT](https://github.com/microsoft/CogACT) | [SimplerEnv](model_servers/cogact/cogact.yaml) | [report](../docs/reproductions/cogact.md) |
| [VLANeXt](model_servers/vlanext/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2602.18532](https://arxiv.org/abs/2602.18532) | [DravenALG/VLANeXt](https://github.com/DravenALG/VLANeXt) | [LIBERO](model_servers/vlanext/libero_spatial.yaml) | [PR #34](https://github.com/allenai/vla-evaluation-harness/pull/34) |
| [DB-CogACT](model_servers/db_cogact/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2510.23511](https://arxiv.org/abs/2510.23511) | [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic) | [LIBERO](model_servers/db_cogact/libero.yaml), [CALVIN](model_servers/db_cogact/calvin.yaml), [SimplerEnv](model_servers/db_cogact/simpler.yaml), [RoboTwin](model_servers/db_cogact/robotwin2.yaml), [ManiSkill2](model_servers/db_cogact/maniskill2.yaml) | [report](../docs/reproductions/dexbotic.md) |
| [starVLA](model_servers/starvla/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2604.05014](https://arxiv.org/abs/2604.05014) | [starVLA/starVLA](https://github.com/starVLA/starVLA) | [LIBERO](model_servers/starvla/libero_qwen3_oft.yaml), [SimplerEnv](model_servers/starvla/groot_simpler.yaml) | [report](../docs/reproductions/starvla.md) |
| [RTC](model_servers/rtc/) | ![◇](https://img.shields.io/badge/◇-blue) | [2506.07339](https://arxiv.org/abs/2506.07339) | [Physical-Intelligence/rtc](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | [Kinetix](model_servers/rtc/kinetix.yaml) | [report](../docs/reproductions/rtc.md) |
| [MolmoBot](model_servers/molmobot/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2603.16861](https://arxiv.org/abs/2603.16861) | [allenai/MolmoBot](https://github.com/allenai/MolmoBot) | [MolmoSpaces](model_servers/molmobot/droid.yaml) | [report](../docs/reproductions/molmobot.md) |
| [MME-VLA](model_servers/mme_vla/) | ![✓](https://img.shields.io/badge/✓-8B5CF6) | [2603.04639](https://arxiv.org/abs/2603.04639) | [RoboMME/robomme_policy_learning](https://github.com/RoboMME/robomme_policy_learning) | [RoboMME](model_servers/mme_vla/pi05_baseline.yaml) | [report](../docs/reproductions/robomme.md) |

<sub>![✓](https://img.shields.io/badge/✓-8B5CF6) [reproduced](../docs/reproductions/) · ![◇](https://img.shields.io/badge/◇-blue) integrated, awaiting first reproduction</sub>

## Config Schemas

### Benchmark configs (`benchmarks/<name>/*.yaml`)

Used by `vla-eval run`.

```yaml
server:
  url: "ws://localhost:8000"
docker:
  image: ghcr.io/allenai/vla-evaluation-harness/<name>:latest
output_dir: "./results"
benchmarks:
  - benchmark: "vla_eval.benchmarks.<name>.benchmark:ClassName"
    episodes_per_task: 50
    params: { ... }
```

### Server configs (`model_servers/<name>/*.yaml`)

Used by `vla-eval serve`.

```yaml
extends: _base.yaml          # optional inheritance
script: "src/vla_eval/model_servers/mymodel.py"
args:
  model_path: org/model-name
  chunk_size: 16
```

The `extends` mechanism deep-merges `args` from a base file; `script` is inherited if omitted. Model servers declare their observation requirements via the HELLO handshake, so the benchmark is auto-configured without manual flags.
