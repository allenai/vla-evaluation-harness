# RFC-0006: Episode Sharding for Parallel Evaluation

- **Author:** @MilkClouds
- **Status:** Implemented
- **Type:** Standards Track
- **Created:** 2026-02-23
- **Requires:** RFC-0004
- **Superseded-By:** —

## Summary

Add episode-level sharding so that N independent OS processes can evaluate disjoint subsets of episodes in parallel, sharing a single model server. A `merge` command combines shard results into a unified report. This is the primary mechanism for reducing wall-clock evaluation time.

## Motivation

LIBERO Spatial: 10 tasks × 50 episodes × ~220 steps = ~110,000 steps, executed sequentially. A single evaluation run takes hours. The bottleneck is the serial alternation of CPU simulation and GPU inference — while one is working, the other is idle.

Parallelizing at the episode level is the most impactful optimization because:
1. Episodes are independent — no shared state between them.
2. Multiple simulation processes can overlap their inference wait time with other simulations.
3. The model server already accepts multiple WebSocket clients concurrently.

### Why shell-level processes, not in-process multiprocessing?

Some benchmark simulators (notably LIBERO/robosuite) have internal state that conflicts with Python `multiprocessing` (OpenGL contexts, MuJoCo shared memory, global state in `robosuite.utils`). Shell-level process isolation (`cmd &`) avoids these issues entirely — each process gets its own address space with no shared state.

## Design

### CLI Interface

```bash
# Run 4 shards in parallel
vla-eval run -c libero_spatial.yaml --shard-id 0 --num-shards 4 &
vla-eval run -c libero_spatial.yaml --shard-id 1 --num-shards 4 &
vla-eval run -c libero_spatial.yaml --shard-id 2 --num-shards 4 &
vla-eval run -c libero_spatial.yaml --shard-id 3 --num-shards 4 &
wait

# Merge results
vla-eval merge results/libero_spatial_shard*.json -o results/libero_spatial.json
```

Without `--shard-id`/`--num-shards`, behavior is unchanged (single process, all episodes).

### Shard Assignment

Work items are the flat list of `(task, episode_idx)` pairs. Round-robin assignment distributes them evenly:

```python
work_items = [(task, ep) for task in tasks for ep in range(episodes_per_task)]
my_items = [w for i, w in enumerate(work_items) if i % num_shards == shard_id]
```

Round-robin (not contiguous blocks) because different tasks may have different max_steps, and round-robin naturally balances load across shards.

### Result File Naming

Shard results use deterministic filenames (no timestamp) so re-runs overwrite previous attempts:

```
results/{benchmark_name}_shard{id}of{total}.json
```

Non-sharded runs keep the existing `{name}_{mode}_{timestamp}.json` format.

### Shard Metadata in Result JSON

Each shard result includes shard provenance:

```json
{
  "benchmark": "libero_spatial",
  "mode": "sync",
  "shard": {"id": 0, "total": 4},
  "partial": false,
  "tasks": [ ... ],
  "overall_success_rate": 0.92
}
```

The `"shard"` field is absent in non-sharded runs.

### Merge Command

`vla-eval merge <glob> [-o output.json]` performs:

1. **Load** all shard JSON files.
2. **Validate** consistency: same benchmark name, same shard total, no duplicate shard IDs.
3. **Detect missing shards**: compare found shard IDs against `range(total)`.
4. **Merge episodes** by task name. Deduplicate by `episode_id` (last-write-wins for re-runs).
5. **Aggregate** metrics: per-task success rate, overall success rate.
6. **Report** coverage and save merged result.

Output example:
```
All 4 shards complete. Coverage: 500/500 episodes (100.0%)
Overall success rate: 94.2% (471/500)
Saved to: results/libero_spatial.json
```

Partial example:
```
⚠ Missing shards: [2] (expected 0..3)
Coverage: 375/500 episodes (75.0%)
Merged result (PARTIAL): 93.1% (349/375)
Saved to: results/libero_spatial.json

To complete: vla-eval run -c libero_spatial.yaml --shard-id 2 --num-shards 4
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Shard process crashes | No result file (or partial). `merge` detects missing shard, reports coverage gap, suggests re-run command. |
| Some episodes fail within a shard | Existing per-episode error isolation (RFC-0004). Episodes recorded with `failure_reason`. Shard file marked `partial: true`. |
| Re-run a failed shard | Deterministic filename overwrites previous attempt. `merge` deduplicates by `episode_id` (last-write-wins). |
| All shards succeed | Clean merge, full coverage. |

## Interaction with Docker

Sharding args pass through to the Docker container transparently:

```bash
docker run ghcr.io/allenai/vla-evaluation-harness/libero:latest run --no-docker --config /tmp/config.yaml --shard-id 0 --num-shards 4
```

The `_run_via_docker` function forwards `--shard-id` and `--num-shards` to the container command.

## Interaction with Model Server

N shard processes connect to the same model server URL. The server already handles multiple WebSocket clients via `websockets.serve` (one coroutine per connection). `PredictModelServer` dispatches each request to `run_in_executor`, so N concurrent requests run in N threads.

This is **not** true GPU batching — requests are serialized on the GPU. True batching requires `BatchPredictModelServer` (RFC-0003, future work). However, even without batching, sharding provides significant speedup because simulation time in one process overlaps with inference time in another.

## Implementation Scope

| Change | File | ~Lines |
|--------|------|--------|
| CLI args `--shard-id`, `--num-shards` | `cli/main.py` | ~15 |
| Shard filtering + deterministic filenames | `orchestrator.py` | ~30 |
| `vla-eval merge` command | `cli/main.py` + `results/merge.py` | ~100 |
| Docker arg forwarding | `cli/main.py` | ~5 |
| Tests | `tests/` | ~80 |

Total: ~230 lines of new/changed code.

## Future Work

- **`vla-eval run-parallel`**: Convenience wrapper that spawns N shard processes + auto-merges. Not needed for v1 — shell scripts suffice.



