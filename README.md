# vla-evaluation-harness

[![CI](https://github.com/allenai/vla-evaluation-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/vla-evaluation-harness/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/vla-eval.svg)](https://pypi.python.org/pypi/vla-eval)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker Images](https://img.shields.io/badge/Docker_Images-ghcr.io-2496ED.svg?logo=docker)](https://ghcr.io/allenai/vla-evaluation-harness)

| | |
|:--|:--|
| **Benchmarks** | [![LIBERO](https://img.shields.io/badge/LIBERO-✓-teal)](configs/libero_all.yaml) [![SimplerEnv](https://img.shields.io/badge/SimplerEnv-✓-teal)](configs/simpler_all_tasks.yaml) [![CALVIN](https://img.shields.io/badge/CALVIN-✓-teal)](configs/calvin_eval.yaml) [![ManiSkill2](https://img.shields.io/badge/ManiSkill2-✓-teal)](configs/maniskill2_eval.yaml) [![LIBERO-Pro](https://img.shields.io/badge/LIBERO--Pro-✓-teal)](configs/libero_pro_eval.yaml) [![RoboCasa](https://img.shields.io/badge/RoboCasa-✓-teal)](configs/robocasa_eval.yaml) [![VLABench](https://img.shields.io/badge/VLABench-✓-teal)](configs/vlabench_eval.yaml) [![MIKASA-Robo](https://img.shields.io/badge/MIKASA--Robo-✓-teal)](configs/mikasa_eval.yaml) [![RoboTwin](https://img.shields.io/badge/RoboTwin-✓-teal)](configs/robotwin_eval.yaml) [![RLBench](https://img.shields.io/badge/RLBench-✓-teal)](configs/rlbench_eval.yaml) [![RoboCerebra](https://img.shields.io/badge/RoboCerebra-✓-teal)](configs/robocerebra_eval.yaml) [![LIBERO-Mem](https://img.shields.io/badge/LIBERO--Mem-✓-teal)](configs/libero_mem.yaml) ![BEHAVIOR-1K](https://img.shields.io/badge/BEHAVIOR--1K-soon-lightgrey) [![Kinetix](https://img.shields.io/badge/Kinetix-✓-teal)](configs/kinetix_eval.yaml) ![FurnitureBench](https://img.shields.io/badge/FurnitureBench-soon-lightgrey) |
| **Models (official)** | [![OpenVLA](https://img.shields.io/badge/OpenVLA-✓-8B5CF6)](configs/model_servers/openvla.yaml) [![π₀](https://img.shields.io/badge/π₀-✓-8B5CF6)](configs/model_servers/pi0_libero.yaml) [![π₀-FAST](https://img.shields.io/badge/π₀--FAST-✓-8B5CF6)](configs/model_servers/pi0_libero.yaml) [![GR00T N1.6](https://img.shields.io/badge/GR00T_N1.6-✓-8B5CF6)](configs/model_servers/groot.yaml) [![OFT](https://img.shields.io/badge/OFT-✓-8B5CF6)](configs/model_servers/oft_libero.yaml) [![X-VLA](https://img.shields.io/badge/X--VLA-✓-8B5CF6)](configs/model_servers/xvla_libero.yaml) [![CogACT](https://img.shields.io/badge/CogACT-✓-8B5CF6)](configs/model_servers/cogact.yaml) [![RTC](https://img.shields.io/badge/RTC-✓-8B5CF6)](configs/model_servers/rtc_kinetix.yaml) ![MemVLA](https://img.shields.io/badge/MemVLA-soon-lightgrey) |
| **Models ([dexbotic](https://github.com/dexmal/dexbotic))** ![stars](https://img.shields.io/github/stars/dexmal/dexbotic?style=social) | [![DB-CogACT](https://img.shields.io/badge/DB--CogACT-✓-8B5CF6)](configs/model_servers/dexbotic_cogact_libero.yaml) |
| **Models ([starVLA](https://github.com/starVLA/starVLA))** ![stars](https://img.shields.io/github/stars/starVLA/starVLA?style=social) | [![QwenGR00T](https://img.shields.io/badge/QwenGR00T-✓-8B5CF6)](configs/model_servers/starvla_groot_simpler.yaml) [![QwenOFT](https://img.shields.io/badge/QwenOFT-✓-8B5CF6)](configs/model_servers/starvla_oft_simpler.yaml) [![QwenPI](https://img.shields.io/badge/QwenPI-✓-8B5CF6)](configs/model_servers/starvla_pi_simpler.yaml) [![QwenFAST](https://img.shields.io/badge/QwenFAST-✓-8B5CF6)](configs/model_servers/starvla_fast_simpler.yaml) |

**One framework to evaluate any VLA model on any robot simulation benchmark.**

### Why vla-evaluation-harness?

| | |
|:--|:--|
| **Batch Parallel Evaluation** | Episode sharding + batched GPU inference → **47× throughput** (2 000 LIBERO episodes in 18 min on 1× H100). [Details](#batch-parallel-evaluation) |
| **Zero Setup** | Benchmarks in Docker, model servers as single-file [uv scripts](https://docs.astral.sh/uv/guides/scripts/) — no dependency conflicts. |
| **AI-Assisted Integration** | Built-in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills for [adding benchmarks](.claude/skills/add-benchmark/) and [model servers](.claude/skills/add-model-server/) — scaffold new integrations in minutes, not hours. |

---

## Motivation

VLA models are evaluated on LIBERO, CALVIN, SimplerEnv, ManiSkill, and others — but each benchmark has its own dependencies, observation format, and evaluation protocol. In practice, every research team ends up maintaining private eval forks per benchmark. Results diverge. Bug fixes don't propagate. No one tests under real-time conditions where the environment keeps moving during inference.

**vla-evaluation-harness** integrates the model once, integrates the benchmark once, and the full cross-evaluation matrix fills itself.

**How**: our abstraction layer fully decouples models from benchmarks.

- Benchmarks run inside **Docker** — no dependency hell, exact reproducibility.
- Model servers are standalone **[uv scripts](https://docs.astral.sh/uv/guides/scripts/)** with inline dependency declarations — zero manual setup.

See [Architecture](docs/architecture.md) for how the pieces connect.

---

## Installation

```bash
pip install vla-eval
```

Or from source:

```bash
git clone https://github.com/allenai/vla-evaluation-harness.git
cd vla-evaluation-harness
uv sync --python 3.11 --all-extras --dev
```

---

## Quick Start

Two terminals: one for the model server (GPU), one for the benchmark client.

```bash
# Terminal 1 — model server (runs on host with GPU)
vla-eval serve --config configs/model_servers/dexbotic_cogact_libero.yaml

# Terminal 2 — run evaluation (benchmark runs in Docker by default)
vla-eval run --config configs/libero_smoke_test.yaml
```

Results are saved to `results/` as JSON. The benchmark runs inside Docker by default — pass `--no-docker` for local development.

### Smoke Tests

Before running full evaluations, verify your setup with the built-in smoke tests:

```bash
vla-eval test --list                                    # show what's available + readiness
vla-eval test -c configs/model_servers/cogact.yaml      # smoke-test a model server
vla-eval test -c configs/libero_smoke_test.yaml         # smoke-test a benchmark (Docker)
vla-eval test                                           # run all available tests
```

### Full Evaluation

For full evaluation (10 tasks × 50 episodes):

```bash
vla-eval run --config configs/libero_spatial.yaml
```

See [Reproduction Reports](docs/reproductions/README.md) for verified scores and per-model details.

> **Need faster runs?** See [Batch Parallel Evaluation](#batch-parallel-evaluation) — **2 000 LIBERO episodes in ~18 min** (47× vs sequential).

---

## Batch Parallel Evaluation

A full evaluation takes hours sequentially. Two layers of parallelism bring this down to minutes:

<p align="center">
  <img src=".github/assets/speedup_comparison.png" alt="Wall-clock evaluation time: sequential vs batch parallel across LIBERO (47×), CALVIN (16×), SimplerEnv (12×)" width="700">
</p>

**Episode sharding** splits `(task, episode)` pairs across N independent processes ([RFC-0006](docs/rfcs/0006-episode-sharding.md)). Each shard connects to the same model server, where a [`BatchPredictModelServer`](docs/rfcs/0007-batch-predict-model-server.md) **batches their inference requests** into a single forward pass. The two axes multiply together.

### Episode Sharding (environment parallelism)

```bash
# Option A: use the helper script (launches all shards + auto-merges)
./scripts/run_sharded.sh -c configs/libero_spatial.yaml -n 50

# Option B: manual launch
vla-eval run -c configs/libero_spatial.yaml --shard-id 0 --num-shards 4 &
vla-eval run -c configs/libero_spatial.yaml --shard-id 1 --num-shards 4 &
# ... (each shard is a separate process)
wait
vla-eval merge -c configs/libero_spatial.yaml -o results/libero_spatial.json
```

Each shard gets a deterministic slice via round-robin. Results merge with episode-level deduplication — if a shard fails, re-run only that shard.

### Batch Model Server (GPU parallelism)

Enable batching in the model server config by setting `max_batch_size > 1`:

```yaml
args:
  max_batch_size: 16    # max observations per GPU forward pass (>1 enables batching)
  max_wait_time: 0.05   # seconds to wait before dispatching a partial batch
```

### Tuning & Combined Effect

We tune parallelism via a demand/supply methodology: **demand λ(N)** measures environment throughput as a function of shards, **supply μ(B)** measures model throughput as a function of batch size. The operating point satisfies λ(N) < 80% · μ(B\*) to prevent queue buildup.

<p align="center">
  <img src=".github/assets/demand_supply.png" alt="Demand/supply throughput for LIBERO + CogACT on H100" width="700">
</p>

Sharding and batching multiply together (DB-CogACT 7B, LIBERO Spatial, 1× H100-80GB):

| | Sequential | Batch Parallel (50 shards, B=16) |
|:--|:---:|:---:|
| Wall-clock | ~14 h | **~18 min** |
| Throughput | ~11 obs/s | ~486 obs/s |

**2 000 episodes, 47× faster.** The included benchmarking tools (`experiments/bench_demand.py`, `experiments/bench_supply.py`) measure λ and μ for any model + benchmark combination. See the [Tuning Guide](docs/tuning-guide.md) for worked examples and `max_wait_time` derivation.

---

## Docker Images

All benchmark environments are packaged as standalone Docker images based on `base`.

| Image | Size | Benchmark | Python | Base |
|-------|------|-----------|--------|------|
| [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) | 3.3 GB | — | 3.10 | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` |
| [`rlbench`](https://ghcr.io/allenai/vla-evaluation-harness/rlbench) | 4.7 GB | RLBench | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`simpler`](https://ghcr.io/allenai/vla-evaluation-harness/simpler) | 4.9 GB | SimplerEnv | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero`](https://ghcr.io/allenai/vla-evaluation-harness/libero) | 6.0 GB | LIBERO | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero-pro`](https://ghcr.io/allenai/vla-evaluation-harness/libero-pro) | 6.2 GB | LIBERO-Pro | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robocerebra`](https://ghcr.io/allenai/vla-evaluation-harness/robocerebra) | 6.3 GB | RoboCerebra | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`calvin`](https://ghcr.io/allenai/vla-evaluation-harness/calvin) | 9.5 GB | CALVIN | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`kinetix`](https://ghcr.io/allenai/vla-evaluation-harness/kinetix) | 9.5 GB | Kinetix | 3.11 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`maniskill2`](https://ghcr.io/allenai/vla-evaluation-harness/maniskill2) | 9.8 GB | ManiSkill2 | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`mikasa-robo`](https://ghcr.io/allenai/vla-evaluation-harness/mikasa-robo) | 10.1 GB | MIKASA-Robo | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero-mem`](https://ghcr.io/allenai/vla-evaluation-harness/libero-mem) | 11.3 GB | LIBERO-Mem | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`vlabench`](https://ghcr.io/allenai/vla-evaluation-harness/vlabench) | 17.7 GB | VLABench | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robotwin`](https://ghcr.io/allenai/vla-evaluation-harness/robotwin) | 28.6 GB | RoboTwin 2.0 | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robocasa`](https://ghcr.io/allenai/vla-evaluation-harness/robocasa) | 35.6 GB | RoboCasa | 3.11 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |

**Pull** (recommended):

```bash
docker pull ghcr.io/allenai/vla-evaluation-harness/libero:latest
```

**Build locally** (see [docker/build.sh](docker/build.sh)):

```bash
docker/build.sh          # build all (base first, then benchmarks)
docker/build.sh libero   # build one
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Component descriptions, protocol, episode flow, configuration |
| [Contributing](CONTRIBUTING.md) | Dev setup, adding benchmarks/models, PR workflow |
| [Reproduction Reports](docs/reproductions/README.md) | Per-model evaluation results and reproducibility verdicts |
| [RFCs](docs/rfcs/README.md) | Design proposals with rationale and status tracking |
| [Design Philosophy](docs/design-philosophy.md) | Freshness, Convenience, Layered Abstraction, Quality, Reproducibility, Openness |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup and PR workflow.

PRs for any 🔜 item in the support matrix are welcome.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{choi2026vlaeval,
  title={vla-eval: A Unified Evaluation Harness for Vision-Language-Action Models},
  author={Choi, Suhwan and Lee, Yunsung and Park, Yubeen and Kim, Chris Dongjoo and Krishna, Ranjay and Fox, Dieter and Yu, Youngjae},
  journal={arXiv preprint arXiv:2603.13966},
  year={2026}
}
```

## License

Apache 2.0
