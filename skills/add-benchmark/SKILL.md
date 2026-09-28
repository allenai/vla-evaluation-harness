---
name: add-benchmark
description: Add or improve a simulation benchmark in vla-eval, including evaluation protocol, assets, adapter lifecycle, and runtime validation.
---

# Add a benchmark

Work from the target harness checkout; paths below are relative to it. Read
`AGENTS.md`, `CONTRIBUTING.md`, and a comparable adapter/config. For an existing
integration, use the affected sections and preserve unrelated behavior.

## 1. Evaluation protocol

Trace the upstream evaluator: task selection → initialization → rollout → success
check → cleanup. Establish:

- Version, task/variation set, initial-state selection, and seeds.
- Robot/controller, observations, action format, and control frequency.
- Warm-up, task-specific horizons, termination, success, and aggregation.

Inspect helper side effects: an expert rollout may select solvable states and
initialize fields needed by success checks. Preserve those functions without
blindly repeating expensive work. Bound setup retries; report failures instead of
silently changing the evaluated population. Document intentional protocol deviations.

## 2. Runtime and assets

| Material | Placement and setup |
| --- | --- |
| Simulator dependencies | `docker/Dockerfile.<name>`; register in `docker/build.sh` and `docker/push.sh`. Pin compatible revisions and verify actual imports. |
| External scenes/data | Persistent cache via existing download mechanisms; record source, version, required contents, and acquisition command. |
| Generated files | Writable scratch/output paths, separate from immutable assets; libraries may try to write beside installed assets. |

Use `vla_eval.dirs.assets_cache("<benchmark>")`. Its root follows
`VLA_EVAL_ASSETS_CACHE` → `VLA_EVAL_HOME/assets` →
`${XDG_CACHE_HOME:-~/.cache}/vla-eval/assets`. Config mounts must map the host cache
to the simulator's container path with the same precedence.

For a working mount/download example, inspect
`configs/benchmarks/behavior1k/eval.yaml` and `Behavior1KBenchmark._ensure_assets`
in `src/vla_eval/benchmarks/behavior1k/benchmark.py`.

- Reuse complete caches; coordinate lazy downloads or prepare them before shards.
- Use `ensure_license` for gated assets. User acceptance is passed through
  `--accept-license <id>` or `VLA_EVAL_ACCEPTED_LICENSES`; do not assume consent.
- Check the intended runtime's write permissions, including read-only images.
- Follow `docs/render-backends.md`; verify real camera output before declaring
  renderer support. Successful environment creation is insufficient.

## 3. Benchmark adapter

Implement `StepBenchmark` under `src/vla_eval/benchmarks/<name>/`, following the
current `src/vla_eval/benchmarks/base.py` contracts. Keep simulator imports lazy.
Add configs under `configs/benchmarks/<name>/` per `configs/README.md`.

| Boundary | Responsibility |
| --- | --- |
| Observations | Named cameras, language, optional state; preserve image content. Policy resize/crop belongs with model preprocessing. |
| Requested inputs | Forward model-requested wrist/state settings through wrappers; explicit benchmark config takes precedence. Verify delivery to the server. |
| Actions | Map controller, absolute/delta convention, frame, rotation, units, and gripper semantics. Apply each conversion in one layer. |
| Results | Declare specs and metric aggregation; supply recording data through the harness, keeping recording policy out of the adapter. |

## 4. Reset, termination, and episode identity

- Match upstream reuse/reconstruction and cleanup, including interrupted episodes.
  Check reset correctness and resource accumulation before optimizing env creation.
- Keep task/variation/episode identity independent of shard assignment. Compare
  representative initial images/states across resets; one seeded RNG may not cover
  robot, scene, and object initialization. Document residual nondeterminism.
- Reconcile task-specific limits with the runner cap to avoid truncating valid
  episodes. Separate simulation-step limits from wall-clock watchdogs.
- Preserve success predicates and distinguish task failure from infrastructure error.

## 5. Validation and handoff

Run `make check`, `make test`, and `uv run vla-eval test --validate`; confirm the
config appears in `uv run vla-eval test --list`. When runtime execution is authorized:

```bash
uv run vla-eval test -c configs/benchmarks/<name>/<config>.yaml --dev
```

Use `--dev` for local adapter edits or rebuild the image. Exercise representative
task paths and consecutive episodes. Inspect images/state, action effects,
termination, resource growth, and shard coverage; compare upstream behavior using
a released baseline or replay when available.

Host `tests/` is for simulator-independent harness logic. Do not add host pytest
cases that import container-only benchmark adapters, install simulator dependencies
on the host to run them, or fake those dependencies through `sys.modules`.
Validate adapter code through smoke/integration runs inside the real benchmark image.
An echo-policy smoke proves plumbing, not simulator fidelity or reproduced performance.

Deliver asset instructions, runnable configs, supported runtime/rendering, protocol
deviations, and separate statements of structural, runtime, and reproduction checks.
