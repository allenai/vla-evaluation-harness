# Contributing

## Ways to Contribute

We'd love your help — especially with reproducibility, which we care about most.

1. **Reproduction reports** — If you've run an evaluation and can compare against published scores, we'd appreciate a report. See [db-cogact.md](docs/reproductions/db-cogact.md) for an example.

2. **Bug reports and fixes** — If something doesn't look right during evaluation, please let us know. Filing an issue or opening a PR helps a lot.

3. **New benchmark integrations** — Adding a benchmark lets everyone reproduce results on it. See [Adding a Benchmark](#adding-a-benchmark) for the step-by-step guide.

4. **New model server integrations** — Adding a model server opens up new evaluation combinations. See [Adding a Model Server](#adding-a-model-server) for the guide.

5. **Leaderboard data** — Missing scores, corrections, or protocol notes are all helpful. See [leaderboard/CONTRIBUTING.md](leaderboard/CONTRIBUTING.md).

We're also open to contributions that improve the harness itself — new evaluation metrics, video recording, visualization tools, and similar enhancements are all welcome.

---

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
├── benchmarks/       # Benchmark adapters (LIBERO + LIBERO-Pro/Plus/Mem, CALVIN, ManiSkill2, SimplerEnv, RoboCasa, VLABench, MIKASA-Robo, RoboTwin, RLBench, RoboCerebra, RoboMME, MolmoSpaces, Kinetix, BEHAVIOR-1K)
├── model_servers/    # Model server ABCs, utilities, and implementations
├── runners/          # Episode execution loops (sync, live)
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

1. Create `src/vla_eval/model_servers/<name>.py` as a **uv script** with [PEP 723](https://peps.python.org/pep-0723/) inline metadata
2. Subclass `PredictModelServer` from `model_servers/predict.py` (or `ModelServer` from `model_servers/base.py` for advanced async use cases)
3. Implement `predict(obs, ctx) -> dict`. Load the model inside `__init__` (no lazy-load hook)
4. Use `run_server()` for the CLI entrypoint — **do not write manual argparse**:
   ```python
   if __name__ == "__main__":
       from vla_eval.model_servers.serve import run_server
       run_server(MyModelServer)
   ```
5. Add config YAML(s) in `configs/model_servers/<name>/` (subdirectory per model; use `extends: _base.yaml` for shared settings)
6. Smoke-test: `vla-eval test -c configs/model_servers/<name>/<name>.yaml`

See `model_servers/cogact.py` for a complete reference implementation.

## Config Conventions

YAML configs are parsed into typed dataclasses in `config.py`. When adding config fields:

- Add the field to the appropriate dataclass with a default value
- Update `from_dict()` if the field needs special handling
- Keep `params` as `dict[str, Any]` — it's benchmark-specific and passed through as-is

## PR Workflow

1. Branch from `main`
2. Make changes, add tests if applicable
3. Update relevant documentation (`README.md` badges, `docs/reproductions/`, etc.)
4. Run `make check && make test`
5. Open a PR — CI will run automatically
