# CLAUDE.md

This file provides context for AI coding assistants working on this repository.

## Project Overview

**vla-evaluation-harness** (`vla-eval`) is a unified evaluation framework for Vision-Language-Action (VLA) models across 11+ robot simulation benchmarks. Models integrate once, benchmarks integrate once, and the full cross-evaluation matrix works automatically.

Core design: Model server communicates with benchmark (Docker container, with optional GPU access for rendering) via WebSocket + msgpack binary protocol. This decouples model dependencies from benchmark dependencies entirely.

## Commands

```bash
# Setup
uv sync --python 3.11 --all-extras --dev

# Quality (CI runs these on every PR)
make lint          # ruff check --fix + ruff format
make check         # ruff check + ruff format --check + ty check (no auto-fix)
make test          # uv run pytest

# Single test
uv run pytest tests/test_protocol.py -v
uv run pytest tests/test_protocol.py::test_name -v

# Smoke tests (model servers, benchmarks, config validation)
vla-eval test                                       # validate configs only (fast, default)
vla-eval test --all                                 # run all categories (validate + server + benchmark)
vla-eval test --list                                # show available tests + prerequisites
vla-eval test --server                              # smoke-test all model servers
vla-eval test --benchmark                           # smoke-test all benchmarks
vla-eval test -c configs/model_servers/cogact.yaml  # smoke-test a specific config
make smoke                                          # shortcut for vla-eval test --all
```

Line length is **119** (configured in pyproject.toml for ruff and ty).

## Architecture

```
CLI (cli/main.py)
 └─ Orchestrator (orchestrator.py)
     ├─ Benchmark (benchmarks/base.py)  ── runs inside Docker container
     │   └─ EpisodeRunner (runners/)    ── sync or live
     │       └─ Connection (connection.py) ←─ WebSocket/msgpack ─→ ModelServer (model_servers/base.py)
     ├─ ResultCollector (results/collector.py)  ── in-memory aggregation for stdout summary
     └─ RecordingStore (recording.py) ── SQLite (one per eval) capturing step rows + per-episode results
```

### Key design decisions

- **Episode-level error isolation**: One episode failure never aborts the entire evaluation.
- **anyio-based async**: asyncio-compatible, not trio. Use anyio primitives for new async code.
- **Parallel evaluation**: Environment parallelism via episode sharding + inference parallelism via batch forward passes.
- **Recording goes through SQLite**: per-episode step rows + episode results + per-eval metadata live in `<output_dir>/recording-<eval_id>.sqlite`. WAL mode + `json_patch` UPSERT lets shards AND the model server (e.g. reflex-train) write concurrently and field-union step rows for the same `(sid, eid, step_id)`. `vla-eval merge` materialises the human-readable per-episode jsonl + per-benchmark aggregate JSON from that DB.

### Recording flow

`vla-eval run` writes raw rows to `<output_dir>/recording-<eval_id>.sqlite`:

- Recording is configured at the **benchmark config's top-level `recording:` key** (sibling of `params`). The orchestrator reads this dict (`output_dir`, `filename_stem`, `record_video`, `record_step`, `video_fps`) and builds the recorder. Benchmarks are not involved in recording policy; they only decide *what* to record (which obs frame, which step fields).
- Benchmark calls `recorder.record_video(frame)` / `recorder.record_step(row)`; the recorder buffers per-episode and flushes in one transaction at episode end.
- `filename_stem` (default `"ep{episode_idx:04d}_{status}"`) is a `str.format` template against the task dict's serializable fields + `{status}`. The orchestrator validates it against the first task at startup so a typo fails fast.
- Model server (optional, used by external callers like reflex-train) receives `(sid, eid, eval_id, db_path)` in the `EPISODE_START` WS payload and opens a `vla_eval.recording.StepRecorder` to push its own step rows. Field-union with the benchmark's rows is automatic via SQLite `json_patch`.
- `vla-eval merge -c <config> [--eval-id <id>]` reads the DB and emits per-episode jsonl + a `BenchmarkResult`-shaped aggregate JSON. Single-shard `vla-eval run` invokes this inline; sharded runs delegate to `scripts/run_sharded.sh` which calls `vla-eval merge` once after `wait`.
- `vla-eval run --no-save` skips recording entirely (in-memory only).

### Tracking

Optional sibling to recording: top-level `tracking.report_to: wandb` (or list / `"all"`) in the eval YAML mirrors aggregate metrics to wandb/trackio. Backend settings come from native env vars (`WANDB_*`, `TRACKIO_*`); the harness only injects `eval_id` + `resume="allow"` so live and merge paths converge on the same run. Per-episode tracking fires on the live path only; sharded mode defers aggregate emission to `vla-eval merge`. See the README "Observability" section.

Read `CONTRIBUTING.md` before any integration work (adding benchmarks/model servers, PR workflow).
