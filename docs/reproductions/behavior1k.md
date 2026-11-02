# BEHAVIOR-1K — Reproduction Status

[Challenge site](https://behavior.stanford.edu/challenge/) |
[Evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html) |
[2026 leaderboard](https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard) |
100 long-horizon household tasks (B100) on R1Pro / OmniGibson

> **Protocol version:** the default is now the 2026 challenge stack: BEHAVIOR-1K `v3.9.4` (the
> challenge's post-release), Isaac Sim 5.1.0, Python 3.11, B100. The image is built FROM upstream's
> official `stanfordvl/behavior:3.9.4`, pinned by digest. The 2025 stack (v3.7.2, Isaac Sim 4.5.0,
> B50) is no longer supported, and **the 2025 results previously published on this page do not apply to the current
> adapter** (different task set, instance split, assets, simulator and timeouts). They are kept below
> only as an archive.

## Status

| Check | State |
|---|---|
| Adapter, Docker recipe, configs on the 2026 stack | ✅ |
| Image builds FROM `stanfordvl/behavior:3.9.4` (Charliecloud) | ✅ 2026-10-31 |
| End-to-end run in Isaac Sim (demo replay) | ✅ renders on H100; 1/2 demos succeed (below) |
| Zero-action baseline run | ✅ 0/2, as expected |
| Trained-VLA reproduction against published 2026 scores | ⬜ not attempted |

## Protocol (2026)

The adapter drives upstream's own `omnigibson.eval.evaluator.BatchedEvaluator` with one logical
environment per task, so scene setup, instance loading, timeouts and metrics are upstream's code; only
the policy transport differs (harness WebSocket instead of upstream's openpi-style server).

| | 2025 stack (dropped) | 2026 stack (default) | Upstream source (v3.9.4) |
|---|---|---|---|
| Tasks | B50 | B100 (`B100_task_misc.csv`) | `omnigibson/eval/utils/eval_utils.py` |
| Instances | TRO instances 1-10 (train ids) | test ids 301-340; public 301-320, hidden 321-340; report indices 0-9, 1 rollout each | `eval_utils.py`, `evaluator.py:resolve_instance_ids`, `docs/challenge/evaluation.md` |
| Timeout | 5000 steps or config | 1.5x mean human-demo length per task | `evaluator.py:load_env` (`EVAL_TIMEOUT_MULTIPLIER`) |
| Time budget | none | `max_steps` x 1 s wall clock per rollout, starting after a free first query (<= 600 s); later queries <= 600 s; an over-budget rollout scores 0 | `evaluator.py:evaluate_instances_batched`, `docs/challenge/{evaluation,updates}.md` |
| Scene | task room types | exact room instances from `B100_task_misc.csv` | `evaluator.py:load_env` |
| Robot | `generate_robot_config` (joylo) | `omnigibson/eval/r1pro.yaml`; base mass 250, head aperture 40 | `evaluator.py:_apply_robot_eval_settings` |
| Cameras | 720/480 RGB | `DefaultWrapper` 224x224 RGB (CLI default) or `RGBDFullResWrapper` 720/480 RGB-D (official track) | `omnigibson/eval/wrappers/` |
| Metric | binary success | success + `q_score` (newly satisfied goal predicates, max over options; 1.0 on success) + efficiency | `omnigibson/metrics/task_metric.py`, `agent_metric.py` |
| Language | humanized task name | dataset instruction (`meta/tasks.jsonl` of `2026-challenge-demos`) | — |

The time budget is new in v3.9.4 (upstream PR 2366). Upstream charges each rollout its policy query time
plus its share of simulator stepping; with one env per rollout that is the wall clock from the first action,
which `step` measures (`time_budget: false` turns it off). A failed rollout records upstream's
`failure_reason` (`group_time_budget_exceeded` or `policy_query_timeout`) with zero `q_score`.
The configs set `server.timeout: 600` to match upstream's per-query cap. One divergence remains: a
query that never answers within 600 s ends as a harness `timeout` error (counted in `num_errors`, and
left out of the mean `q_score`) instead of upstream's zero-score record.
Upstream's multi-port mode (`--policy-endpoints`, `MultiWebsocketPolicy`) is not used: the harness
runs one env per benchmark process and shards across processes instead.

The 23-D action layout and controllers are unchanged from 2025: `base[0:3]` holonomic velocity in
[-1, 1], `torso[3:7]`, `left_arm[7:14]`, `left_gripper[14:15]`, `right_arm[15:22]`,
`right_gripper[22:23]` (absolute joint positions).

## Smoke results (2026 stack)

Run on 2026-10-31 with Charliecloud (`--runtime charliecloud --dev`, SquashFS export of the image
FROM `stanfordvl/behavior:3.9.4`, host driver 580.95.05 injected) on one H100 80GB. Isaac Sim 5.1
starts headless and renders through the image's Vulkan ICD; the first episode of a fresh process takes
~8-10 min (boot + scene load); `picking_up_trash` then stepped at ~6.5 steps/s with `DefaultWrapper`.

| Run | Task | Instance | Steps | `success` | `q_score` |
|---|---|---|---:|:---:|---:|
| Demo replay, `demo_replay.yaml` (episode 0, 1956 steps) | `turning_on_radio` | train 1 | 1360 | ✅ | 1.0 |
| Demo replay, `demo_replay.yaml` (episode 200, 7667 steps) | `picking_up_trash` | train 1 | 7902 (timeout) | ❌ | 0.0 |
| Zero action, `RGBDFullResWrapper`, `max_steps: 300` | `turning_on_radio` | public 0 (301) | 300 | ❌ | 0.0 |
| Zero action, `RGBDFullResWrapper`, `max_steps: 300` | `turning_on_radio` | public 1 (302) | 300 | ❌ | 0.0 |

Every per-episode metric (steps, `q_score`, simulator time, agent distances) is bit-identical to the
same runs on v3.9.3-post2 with the previous self-built image, so the move to v3.9.4 and the official
image changed no result. The demo run also exercised the task switch (`og.clear()`). The
`turning_on_radio` replay succeeds 4 steps earlier than the 2025 stack did (1364). The
`picking_up_trash` replay reaches no goal predicate on either version; whether that is replay drift
over a 7.7k-step trajectory or an env difference is not diagnosed, so one passing demo is the only
end-to-end correctness evidence so far. To exercise the time budget, a scratch zero-action server that
sleeps 2 s per query ran `turning_on_radio` public 0 with `max_steps: 20` (a 20 s budget): the rollout
failed after 8 simulator steps with `failure_reason: group_time_budget_exceeded`, `q_score` 0 and
upstream's zeroed normalized metrics (`normalized_time` 1/1.5). Known issue: the container process segfaults during interpreter
teardown (Isaac Sim is left running, see `cleanup`), after results are written, so `vla-eval run`
exits non-zero (245) on an otherwise complete run.

## Published reference scores

The 2026 leaderboard's `data/results.jsonl` was filled on 2026-10-30 with six verified public-test
submissions (top: q_score 0.683, success 0.482). None has been reproduced here, and whether their
checkpoints are public was not checked. The 2025 leaderboard (B50) is not comparable to B100.

## How to run

```bash
# 1. Build the image (NVIDIA Omniverse EULA: https://docs.omniverse.nvidia.com/eula/).
docker/build.sh behavior1k --accept-license behavior1k
#    It pulls stanfordvl/behavior:3.9.4 (14.6 GB compressed, ~28 GB unpacked) and adds the harness.
#    Charliecloud: ch-image build -t <tag> --build-arg ACCEPT_NVIDIA_EULA=YES -f docker/Dockerfile.behavior1k .
#    Keep CH_IMAGE_STORAGE on local disk; the image tree is ~200k files.  On a slow link ch-image 0.45
#    failed after the 13 GB layer with "unhandled authentication failure" (its Docker Hub token had
#    expired); pre-seeding $CH_IMAGE_STORAGE/dlcache/<sha256 hex>.tar.gz with the layers avoids it.

# 2. Start an R1Pro model server, e.g. the zero-action baseline.
uv run --script src/vla_eval/model_servers/behavior1k_baseline.py --port 8000

# 3. Run.  The first run downloads ~32 GiB (behavior-1k-assets 3.9.0, robot assets 3.8.2, 2026 task
#    instances; ~180k files) into the behavior1k asset cache (see vla_eval.dirs).
uv run vla-eval run -c configs/benchmarks/behavior1k/eval.yaml --accept-license behavior-dataset-tos --yes
```

`eval.yaml` is the challenge self-evaluation (100 tasks x 10 public-test instances). Upstream reports
scene loads of 150-300 s per task and ~20 FPS stepping on an RTX 4090.

## Archive: 2025 stack results (do not apply to the current default)

Measured in 2026-04 with BEHAVIOR-1K v3.7.2 / Isaac Sim 4.5.0 on A100:

- Demo replay of `turning_on_radio` instance 1 (`2025-challenge-demos` episode 00000010, 1956 steps):
  success after 1364 steps
  ([JSON](data/behavior1k_demo_replay_turning_on_radio_inst1.json)).
- Zero-action baseline on `turning_on_radio`, 100 steps: 0% success
  ([JSON](data/behavior1k_baseline_zero_action_turning_on_radio.json)).
