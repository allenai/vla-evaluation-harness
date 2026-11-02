---
name: add-model-server
description: Add or improve a VLA model integration in vla-eval, covering checkpoint setup, inference parity, per-episode state, and serving validation.
---

# Add a model server

Work from the target harness checkout; paths below are relative to it. Read
`AGENTS.md`, `CONTRIBUTING.md`, `configs/README.md`, and a comparable server.
For an existing integration, focus on the changed behavior.

## 1. Integration scope

Identify the checkpoint, upstream evaluator, target embodiment/benchmark, required
inputs, action format, and temporal behavior. Checkpoint variants may differ.

Choose an existing bridge config, a focused bridge extension, or a dedicated server
based on what is missing. Use `PredictModelServer` for blocking inference,
`DaemonModelServer` for a per-episode real-time loop (`run(ep)`; RTC streaming is
`examples/live_servers/rtc.py`), and `ModelServer` for custom async behavior; use
`run_server()` for the CLI.

## 2. Dependencies and model assets

Resolve weights/revision, architecture config, tokenizer, image processor, adapters,
normalization statistics/dataset keys, and auxiliary models.

- Reuse Hub caches or explicit local paths; document downloads and overrides.
  Verify requested files/revisions, not merely the existence of any snapshot.
- Fail on missing checkpoint/suite assets; do not select another checkpoint or
  invent processor defaults. Expose supported local tokenizer/processor overrides.
- Use a PEP 723 uv script with an editable harness dependency at the correct relative
  path, commit-pinned git dependencies, and `exclude-newer` set to the verification
  date. Verify resolved imports and real inference in that isolated environment.
- Prefer installable upstream packages; use the existing pinned-clone/cache pattern
  when packaging is unavailable. Never clone over a user-provided checkout.

Load the policy once during initialization. Put checkpoint configs under
`configs/model_servers/<name>/`, using `extends: _base.yaml` for shared settings.

## 3. Inference adapter

Trace observation → preprocessing → policy → decoded action → benchmark input.
Assign each transform to one layer: camera order/resize/crop/aspect ratio, pixel and
state normalization, action denormalization, frame/rotation/units, and gripper mapping.
Inspect upstream processors before adding transforms they already perform.

| Interface | Contract |
| --- | --- |
| `get_observation_params()` | Request benchmark-supported inputs such as wrist/state; verify actual delivery and config overrides. |
| `get_observation_spec()` / `get_action_spec()` | Declare input/output semantics; shape checks alone cannot establish compatibility. |
| `predict(obs, row)` | One observation of one episode (`row.id`, `row.step`, `row.is_first`, `row.episode`); return `{"actions": array}` shaped `(action_dim,)` or `(chunk_size, action_dim)`. |
| `predict_batch(obs, rows)` | Optional: observation columns (a row per episode) to action columns; implement when the model infers a batch faster than rows one by one, and set `max_batch_size`. |
| `reset_episodes(rows)` / `finish_episodes(rows, results)` | Batched, blocking; `rows.episodes[i].config` is the `EPISODE_START` payload; `results[i]["status"]` says how it ended. The episode begins once `reset_episodes` returns: warm up (compile etc.) there or in `__init__`, so the first inference runs at steady-state latency. |

Read current contracts in `src/vla_eval/model_servers/base.py` and `predict.py`;
`cogact.py` provides a concrete server/batching example. Do not fabricate or pad
missing inputs unless that behavior belongs to the upstream policy.

## 4. State and concurrency

- Identify ownership of history, recurrent state, action queues, and ensembling.
  Preserve replanning frequency; cached chunks may bypass observations needed by
  policies that update on every executed step.
- Keep mutable state keyed by `row.id` (ids are reused): reset it in `reset_episodes`.
  Global state → `max_rows=1`; a policy that must see every observation (e.g. a state history with buffered
  chunks) subclasses `ModelServer` and keeps its own chunk queue, as `vlanext.py` does.
- Verify batching with mixed episodes, episode lengths, and resets. Compare single
  and batched behavior with controlled inputs/randomness, allowing justified
  stochastic or numerical differences rather than requiring bitwise identity.

## 5. Validation and handoff

Run `make check`, `make test`, and config validation. Check smoke discovery with
`uv run vla-eval test --list`. With weights and allocated compute:

```bash
uv run vla-eval test -c configs/model_servers/<name>/<config>.yaml
```

Compare upstream and adapter on the same observation/history: inspect intermediate
tensors and decoded actions to locate the first divergence. Follow with a real
rollout, consecutive episodes, and concurrent episodes where supported. If upstream
also fails, report that limitation instead of tuning toward a published score.

Host `tests/` is for model-independent harness logic. Do not add host pytest cases
that require a server's isolated model dependencies, install those dependencies into
the host test environment, or fake them through `sys.modules`. Validate model-specific
code with real loading/inference checks in the server's PEP 723 uv environment.
Distinguish loading, inference, rollout, and reproduction support in the documentation.

After correctness checks, measure warm latency, throughput, and memory on representative
inputs, separately from download/load/compile costs. Record and validate changes to
precision, inference steps, or action horizon: they may change the policy.
Deliver acquisition instructions, configs, supported inputs/benchmarks, state/batching
behavior, and completed or unavailable checks.
