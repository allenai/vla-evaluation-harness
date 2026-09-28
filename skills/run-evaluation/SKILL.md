---
name: run-evaluation
description: Run, debug, and scale VLA evaluations in vla-eval while preserving evaluation conditions and verifying complete results.
---

# Run an evaluation

Work from the target harness checkout. Read `AGENTS.md`, `configs/README.md`, and
the selected model/benchmark notes. Discover configs with
`rg --files configs/model_servers configs/benchmarks`. Use the relevant sections;
a validated setup need not be re-integrated.

## 1. Evaluation plan

Specify checkpoint/revision, benchmark version, tasks/variations, episodes, seeds,
and initialization. Match observations, controller/actions, horizon, and replanning
to the intended comparison; record deliberate differences.

Define expected work, recording needs, and a distinct output directory. Separate
smoke-run completion from reproduction evidence. `docs/reproductions/running-guide.md`
contains examples, not universal hardware allocations or verdict thresholds.

## 2. Preflight

- Verify checkpoint/auxiliary files, assets, image, and code actually executed.
  Use `--dev` for local benchmark edits or rebuild; keep model dependencies isolated.
- Resolve asset roots through `src/vla_eval/dirs.py` and config mounts. Check paths
  on the consuming host; server and benchmark may not share storage. Reuse caches,
  coordinate acquisition, and provide writable scratch/output space and disk capacity.
- Use allocated CPU/GPU resources and the permitted runtime. Consult
  `docs/runtimes.md` or `docs/render-backends.md` when changing either; follow local policies.
- Check server readiness via HTTP 200 from `/health` and connectivity from the
  benchmark runtime. Loading and first-inference compilation are separate phases;
  allow justified warm-up without masking a stall.

## 3. Baseline run

For an unfamiliar setup, run a small representative workload in separate terminals
or managed processes:

```bash
# Server process; keep running during evaluation.
uv run vla-eval serve -c <model-config>

# Benchmark process, after the health check succeeds.
uv run vla-eval run -c <benchmark-config> \
  --server-url ws://<server-host>:<port> --output-dir <run-directory>
```

Set actual task/episode counts through supported config/CLI options. Inspect images,
state, actions, reset, termination, and recordings across relevant task paths and
consecutive episodes.

Diagnose the first failing boundary: setup → observations → preprocessing/inference
→ action application → success/termination → recording. Compare with upstream;
normal process exit, low scores, or matching shapes alone do not locate the problem.
For matched runs, compare representative initial states rather than assuming equal
seeds guarantee equal starts. Record remaining nondeterminism.

## 4. Throughput tuning

Optimize valid completed episodes per time/resource under the same conditions.
Read `docs/tuning-guide.md` and check current controls before tuning.

| Step | What to measure or change |
| --- | --- |
| Locate the bottleneck | Environment demand and serving capacity; rendering, CPU contention, GPU memory, latency, recording I/O. Use `experiments/bench_demand.py` and `experiments/bench_supply.py` for dedicated probes. |
| Increase concurrency | Sweep independent shards and supported batch sizes incrementally; bound resources/threads and tune batch waiting against latency. Leave headroom. |
| Judge the result | Total wall time, valid-episode throughput, worker imbalance, and errors. Stop scaling when throughput plateaus or latency/errors grow. |

Separate cold start from steady state, requests from model forward passes under
chunking, and environment setup from rollout. Failures can run to the horizon:
changes in success mix may distort timing comparisons. Inspect current work
assignment rather than assuming fixed episode counts per shard.

Preserve episode identity, reset, and session isolation. Do not silently change
inputs, precision, inference steps, action horizon, control frequency, or episode
limits for speed. In live/latency-sensitive evaluation, serving delays and the
runner's timing/hold-action semantics are part of the measured conditions.

## 5. Full run and recovery

Prefer the existing helper when suitable; inspect its current options. Set
`server.url` in the benchmark config, then choose a measured shard count:

```bash
uv run bash scripts/run_sharded.sh -c <benchmark-config> \
  -n <shards> -o <run-directory>
```

The helper shares an eval ID and exports after workers exit. Manual shards must
share `--eval-id` and recording destination; independent evaluations use distinct
destinations. Shared SQLite recording requires working filesystem locks and sync.

Monitor expected work, episode errors, and worker exits. Episode failures need not
stop the process. Before retrying, inspect completed/unfinished work and current
queue/recording semantics to avoid duplicate execution or overwritten results.
Preserve diagnostic logs and stop only this run's jobs/processes.

## 6. Results and reporting

The authoritative recording is `recording-<eval-id>.sqlite`. Single-shard runs export
automatically; for manual export:

```bash
uv run vla-eval export <recording.sqlite> -o <export-directory>
```

Check expected versus recorded task/episode identities, missing/duplicate/errored
episodes, worker completion, aggregation, and the score denominator. An export may
be a snapshot of an incomplete run. Distinguish task failures from infrastructure
errors and disclose their treatment.

Report scores, episode/error counts, deviations, checkpoint and code/image provenance,
resource use, and recording/log locations. Separate smoke validation from reproduction;
executed results do not belong in the paper-reported leaderboard.
