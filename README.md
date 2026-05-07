# vla-evaluation-harness

[![CI](https://github.com/allenai/vla-evaluation-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/vla-evaluation-harness/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/vla-eval.svg)](https://pypi.python.org/pypi/vla-eval)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker Images](https://img.shields.io/badge/Docker_Images-ghcr.io-2496ED.svg?logo=docker)](https://ghcr.io/allenai/vla-evaluation-harness)

| | |
|:--|:--|
| **Benchmarks** | [![LIBERO](https://img.shields.io/badge/LIBERO-✓-teal)](configs/benchmarks/libero/) [![SimplerEnv](https://img.shields.io/badge/SimplerEnv-✓-teal)](configs/benchmarks/simpler/) [![CALVIN](https://img.shields.io/badge/CALVIN-✓-teal)](configs/benchmarks/calvin/) [![ManiSkill2](https://img.shields.io/badge/ManiSkill2-◇-blue)](configs/benchmarks/maniskill2/) [![LIBERO-Pro](https://img.shields.io/badge/LIBERO--Pro-◇-blue)](configs/benchmarks/libero_pro/) [![LIBERO-Plus](https://img.shields.io/badge/LIBERO--Plus-✓-teal)](configs/benchmarks/libero_plus/) [![RoboCasa](https://img.shields.io/badge/RoboCasa-◇-blue)](configs/benchmarks/robocasa/) [![VLABench](https://img.shields.io/badge/VLABench-◇-blue)](configs/benchmarks/vlabench/) [![MIKASA-Robo](https://img.shields.io/badge/MIKASA--Robo-◇-blue)](configs/benchmarks/mikasa/) [![RoboTwin](https://img.shields.io/badge/RoboTwin-◇-blue)](configs/benchmarks/robotwin/) [![RLBench](https://img.shields.io/badge/RLBench-◇-blue)](configs/benchmarks/rlbench/) [![RoboCerebra](https://img.shields.io/badge/RoboCerebra-◇-blue)](configs/benchmarks/robocerebra/) [![LIBERO-Mem](https://img.shields.io/badge/LIBERO--Mem-◇-blue)](configs/benchmarks/libero_mem/) [![BEHAVIOR-1K](https://img.shields.io/badge/BEHAVIOR--1K-◇-blue)](configs/benchmarks/behavior1k/) [![Kinetix](https://img.shields.io/badge/Kinetix-◇-blue)](configs/benchmarks/kinetix/) [![RoboMME](https://img.shields.io/badge/RoboMME-✓-teal)](configs/benchmarks/robomme/) [![MolmoSpaces-Bench](https://img.shields.io/badge/MolmoSpaces--Bench-✓-teal)](configs/benchmarks/molmospaces/) ![FurnitureBench](https://img.shields.io/badge/FurnitureBench-·-lightgrey) |
| **Models (official)** | [![OpenVLA](https://img.shields.io/badge/OpenVLA-✓-8B5CF6)](configs/model_servers/openvla/) [![π₀](https://img.shields.io/badge/π₀-✓-8B5CF6)](configs/model_servers/pi0/) [![π₀-FAST](https://img.shields.io/badge/π₀--FAST-✓-8B5CF6)](configs/model_servers/pi0/) [![GR00T N1.6](https://img.shields.io/badge/GR00T_N1.6-✓-8B5CF6)](configs/model_servers/groot/) [![OFT](https://img.shields.io/badge/OFT-✓-8B5CF6)](configs/model_servers/oft/) [![X-VLA](https://img.shields.io/badge/X--VLA-✓-8B5CF6)](configs/model_servers/xvla/) [![CogACT](https://img.shields.io/badge/CogACT-◇-blue)](configs/model_servers/cogact/) [![RTC](https://img.shields.io/badge/RTC-◇-blue)](configs/model_servers/rtc/) [![VLANeXt](https://img.shields.io/badge/VLANeXt-✓-8B5CF6)](configs/model_servers/vlanext/) [![MolmoBot](https://img.shields.io/badge/MolmoBot-✓-8B5CF6)](configs/model_servers/molmobot/) ![MemVLA](https://img.shields.io/badge/MemVLA-·-lightgrey) |
| **Models ([dexbotic](https://github.com/dexmal/dexbotic))** ![stars](https://img.shields.io/github/stars/dexmal/dexbotic?style=social) | [![DB-CogACT](https://img.shields.io/badge/DB--CogACT-✓-8B5CF6)](configs/model_servers/db_cogact/) |
| **Models ([starVLA](https://github.com/starVLA/starVLA))** ![stars](https://img.shields.io/github/stars/starVLA/starVLA?style=social) | [![QwenGR00T](https://img.shields.io/badge/QwenGR00T-✓-8B5CF6)](configs/model_servers/starvla/) [![QwenOFT](https://img.shields.io/badge/QwenOFT-✓-8B5CF6)](configs/model_servers/starvla/) [![QwenPI](https://img.shields.io/badge/QwenPI-◇-blue)](configs/model_servers/starvla/) [![QwenFAST](https://img.shields.io/badge/QwenFAST-✓-8B5CF6)](configs/model_servers/starvla/) |

<sub>✓ [reproduced](docs/reproductions/)  |  ◇ integrated, awaiting first reproduction  |  · planned</sub>

**One framework to evaluate any VLA model on any robot simulation benchmark.**

### Why vla-evaluation-harness?

| | |
|:--|:--|
| **Batch Parallel Evaluation** | Episode sharding + batched GPU inference → **47× throughput** (2 000 LIBERO episodes in 18 min on 1× H100). [Details](#batch-parallel-evaluation) |
| **Zero Setup** | Benchmarks in Docker, model servers as single-file [uv scripts](https://docs.astral.sh/uv/guides/scripts/) — no dependency conflicts. |
| **AI-Assisted Integration** | Built-in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skills for [adding benchmarks](.claude/skills/add-benchmark/) and [model servers](.claude/skills/add-model-server/) — scaffold new integrations in minutes, not hours. |
| **[Leaderboard](https://allenai.github.io/vla-evaluation-harness/leaderboard/)** | The largest unified VLA comparison — 1,885 models × 18 benchmarks, aggregated from 1,755 papers. |

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
vla-eval serve --config configs/model_servers/db_cogact/libero.yaml

# Terminal 2 — run evaluation (benchmark runs in Docker by default)
vla-eval run --config configs/benchmarks/libero/smoke_test.yaml
```

Results are saved to `results/` as JSON. The benchmark runs inside Docker by default — pass `--no-docker` for local development.

### Full Evaluation

For full evaluation (10 tasks × 50 episodes):

```bash
vla-eval run --config configs/benchmarks/libero/spatial.yaml
```

See [Reproduction Reports](docs/reproductions/) for verified scores and per-model details.

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
./scripts/run_sharded.sh -c configs/benchmarks/libero/spatial.yaml -n 50

# Option B: manual launch
vla-eval run -c configs/benchmarks/libero/spatial.yaml --shard-id 0 --num-shards 4 &
vla-eval run -c configs/benchmarks/libero/spatial.yaml --shard-id 1 --num-shards 4 &
# ... (each shard is a separate process)
wait
vla-eval merge -c configs/benchmarks/libero/spatial.yaml -o results/libero_spatial.json
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
| [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) | 3.3 GB | — | — | `nvidia/cuda:12.1.1-runtime-ubuntu22.04` |
| `rlbench` 🔒 | 4.7 GB | RLBench | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`simpler`](https://ghcr.io/allenai/vla-evaluation-harness/simpler) | 4.9 GB | SimplerEnv | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero`](https://ghcr.io/allenai/vla-evaluation-harness/libero) | 6.0 GB | LIBERO | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero-pro`](https://ghcr.io/allenai/vla-evaluation-harness/libero-pro) | 6.2 GB | LIBERO-Pro | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robocerebra`](https://ghcr.io/allenai/vla-evaluation-harness/robocerebra) | 6.4 GB | RoboCerebra | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`calvin`](https://ghcr.io/allenai/vla-evaluation-harness/calvin) | 9.6 GB | CALVIN | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`maniskill2`](https://ghcr.io/allenai/vla-evaluation-harness/maniskill2) | 9.8 GB | ManiSkill2 | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`kinetix`](https://ghcr.io/allenai/vla-evaluation-harness/kinetix) | 10.0 GB | Kinetix | 3.11 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`mikasa-robo`](https://ghcr.io/allenai/vla-evaluation-harness/mikasa-robo) | 10.1 GB | MIKASA-Robo | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero-mem`](https://ghcr.io/allenai/vla-evaluation-harness/libero-mem) | 11.3 GB | LIBERO-Mem | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`libero-plus`](https://ghcr.io/allenai/vla-evaluation-harness/libero-plus) | 14.8 GB | LIBERO-Plus | 3.8 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robomme`](https://ghcr.io/allenai/vla-evaluation-harness/robomme) | 17.0 GB | RoboMME | 3.11 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`vlabench`](https://ghcr.io/allenai/vla-evaluation-harness/vlabench) | 17.7 GB | VLABench | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| `behavior1k` 🔒 | 23.6 GB | BEHAVIOR-1K | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robotwin`](https://ghcr.io/allenai/vla-evaluation-harness/robotwin) | 28.6 GB | RoboTwin 2.0 | 3.10 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`molmospaces`](https://ghcr.io/allenai/vla-evaluation-harness/molmospaces) | 31.4 GB | MolmoSpaces-Bench | 3.11 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |
| [`robocasa`](https://ghcr.io/allenai/vla-evaluation-harness/robocasa) | 35.6 GB | RoboCasa | 3.11 | [`base`](https://ghcr.io/allenai/vla-evaluation-harness/base) |

<sub>🔒 = build-locally only; the Dockerfile gates the build behind a licence opt-in (`docker/build.sh <name> --accept-license <name>`) and the image isn't published to ghcr.io.</sub>

**Pull** (recommended):

```bash
docker pull ghcr.io/allenai/vla-evaluation-harness/libero:latest
```

**Build locally** (see [docker/build.sh](docker/build.sh)):

```bash
docker/build.sh                                           # build all (gated images skipped)
docker/build.sh libero                                    # build one
docker/build.sh behavior1k --accept-license behavior1k    # build a gated image
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Component descriptions, protocol, episode flow, configuration |
| [Contributing](CONTRIBUTING.md) | Dev setup, adding benchmarks/models, PR workflow |
| [Reproduction Reports](docs/reproductions/) | Per-model evaluation results and reproducibility verdicts |
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
