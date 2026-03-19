# Contributing

## Dev Environment

```bash
# Clone and install (requires uv: https://docs.astral.sh/uv/)
git clone https://github.com/allenai/vla-evaluation-harness.git
cd vla-evaluation-harness
uv sync --python 3.11 --all-extras --dev
```

## Running Tests

```bash
make test              # unit tests (pytest)
make smoke             # smoke tests across all CLI commands (vla-eval test)
```

### Smoke Tests

`vla-eval test` is a unified CLI for smoke-testing model servers and benchmarks:

```bash
vla-eval test --list                                    # show available tests + prerequisites
vla-eval test --validate                                # validate all benchmark config import strings
vla-eval test --server                                  # smoke-test all model servers
vla-eval test --benchmark                               # smoke-test all benchmarks
vla-eval test -c configs/model_servers/cogact.yaml      # smoke-test a specific config
vla-eval test --dry-run                                 # preview what would run
vla-eval test                                           # run all available tests
```

Server tests require `uv` + model weights + GPU. Benchmark tests require Docker + the benchmark image (pulled via `docker pull`). Unavailable tests are auto-skipped.

## Linting, Formatting & Type Checking

```bash
make lint              # ruff check --fix + ruff format (auto-fix)
make format            # ruff format only
make check             # lint + format + ty check (no auto-fix, CI-style)
```

Ruff and ty config are in `pyproject.toml` — line length is **119**.

## CI

Every PR triggers lint, type-check, and test jobs automatically (`.github/workflows/ci.yml`).

## Project Structure

```
src/vla_eval/
├── cli/              # CLI entry point (argparse)
├── benchmarks/       # Benchmark adapters (LIBERO, LIBERO-Pro, CALVIN, ManiSkill2, SimplerEnv, RoboCasa, VLABench, MIKASA-Robo, RoboTwin, RLBench, RoboCerebra)
├── model_servers/    # Model server ABCs, utilities, and implementations
├── runners/          # Episode execution loops (sync, async)
├── results/          # Result collection and shard merging
├── protocol/         # msgpack message definitions
├── orchestrator.py   # Top-level evaluation orchestrator
├── connection.py     # WebSocket client with retry/reconnect
├── config.py         # Typed dataclasses (ServerConfig, DockerConfig, EvalConfig)
└── registry.py       # Lazy import registry for benchmarks/servers
```

## Adding a Benchmark

1. Create `src/vla_eval/benchmarks/<name>/benchmark.py`
2. Subclass `StepBenchmark` from `benchmarks/base.py`
3. Implement the 6 required methods: `get_tasks()`, `reset()`, `step()`, `make_obs()`, `check_done()`, `get_step_result()`
4. Optionally override `get_metadata()` to set defaults like `max_steps`
5. Reference via import string in config YAML (e.g. `benchmark: "vla_eval.benchmarks.<name>.benchmark:MyBenchmark"`)
6. Add a config YAML in `configs/`
7. Add a Dockerfile in `docker/Dockerfile.<name>`
8. Register the name in the `BENCHMARKS` array in `docker/build.sh` and the `IMAGES` array in `docker/push.sh`
9. Smoke-test: `vla-eval test -c configs/<name>.yaml` (runs 1 episode with an EchoModelServer — no real model or GPU needed, but requires Docker + the benchmark image)

See `benchmarks/libero/` for a complete reference implementation.

## Adding a Model Server

1. Create `src/vla_eval/model_servers/<name>.py`
2. Subclass `ModelServer` from `model_servers/base.py` (or `PredictModelServer` from `model_servers/predict.py` for the common blocking-inference pattern)
3. Implement `predict(obs, ctx) -> dict` (`PredictModelServer`) or `on_observation(obs, ctx)` (`ModelServer`)
4. Reference via import string in config YAML
5. Add a config YAML in `configs/model_servers/`
6. Smoke-test: `vla-eval test -c configs/model_servers/<name>.yaml` (launches the server, sends dummy observations from a StubBenchmark, checks for actions — requires `uv` + GPU + model weights but no simulation environment)

See `model_servers/dexbotic/cogact.py` for a complete reference implementation.

## Config Conventions

YAML configs are parsed into typed dataclasses in `config.py`. When adding config fields:

- Add the field to the appropriate dataclass with a default value
- Update `from_dict()` if the field needs special handling
- Keep `params` as `dict[str, Any]` — it's benchmark-specific and passed through as-is

## PR Workflow

1. Branch from `main`
2. Make changes, add tests if applicable
3. Update relevant documentation (`README.md` badges, `docs/reproductions/README.md` index, etc.)
4. Run `make check && make test`
5. Open a PR — CI will run automatically
