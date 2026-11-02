# RTC on Kinetix — Reproduction Report

Real-Time Chunking (RTC) for action-chunking flow policies. [Paper](https://arxiv.org/abs/2506.07339) |
[GitHub](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | 12 Kinetix levels, flow-matching MLP-Mixer, H = 8.

Beyond Fig. 5, this checks that vla-eval's **live** mode (wall-clock ticks, injected latency) matches the paper's
**sync** protocol, where the delay is counted in steps.

## Results Summary

| Arm | What | Result |
|-----|------|--------|
| R | Official `eval_flow.py`, run by us (2048 rollouts × 12 levels, `bc/31`, BID weak `bc/7`) | ✅ matches Fig. 5 (≤ 0.7 pp, BID ≤ 1.9 pp) |
| R′ | R with success read as Kinetix v3 reads it (see *Success metric*) | the reference vla-eval is held to |
| C | vla-eval **sync**, delay counted in steps (2048 × 12) | ✅ vs R′: mean diff −0.2 pp, 11/15 conditions pass both criteria |
| L | vla-eval **live**, 10 Hz, latency (d − ½)/hz (512 × 12) | ✅ vs C: 10/10 pass, \|diff\| ≤ 0.84 pp; vs R′: 9/10 |

Every failure is a d = 0 average near the ceiling (CI ≈ ±0.3 pp) or trampoline at naive d = 1 (see *Open issues*).

| | |
|---|---|
| **Checkpoints** | `gs://rtc-assets/bc/31/policies` (strong), `bc/7` (BID weak); see [`configs/model_servers/rtc/README.md`](../../configs/model_servers/rtc/README.md) |
| **Server** | [`src/vla_eval/model_servers/rtc.py`](../../src/vla_eval/model_servers/rtc.py) (`KinetixRTCServer`, imports the official `FlowPolicy`) |
| **Server configs** | [`configs/model_servers/rtc/`](../../configs/model_servers/rtc/) (`naive`, `rtc`, `rtc_hard`, `bid`) |
| **Benchmark configs** | [`eval.yaml`](../../configs/benchmarks/kinetix/eval.yaml) (sync), [`realtime.yaml`](../../configs/benchmarks/kinetix/realtime.yaml) (live) |
| **Results** | [`data/rtc-kinetix/`](data/rtc-kinetix/): `reference/` (R, R′), `sync/` (C), `live/` (L), `report.py`, `ref_eval.py` |

## Protocol

Official `eval_flow.py` @ `9296f31`, Kinetix `cf7453e`: chunk 0 comes from o₀ with no delay; every s steps a chunk is
sampled from that step's observation, the previous chunk covers the next d steps, the new one the rest. Action noise
N(0, 0.1²), 256-step cap, first episode of each rollout only.

`KinetixRTCServer` runs this schedule per episode and answers each observation with its step's action (`reply` in sync,
`PUBLISH` in live); the step is the benchmark's `env_step`. A boundary's chunk lands, by `latency`:
- `steps` (C): when step t + d's observation arrives.
- `wall` (L): (d − ½)/hz after its observation was admitted, so jitter below Δt/2 cannot change the effective delay.
- `wall_exact`: d/hz after admission.

Latency is slept in the episode task, never on the model thread. Methods call the official `action` /
`realtime_action` / `bid_action` unchanged: naive, RTC soft (`exp`, β = 5), RTC hard (`zeros`), BID (N = 32, K = 3).

- **Live start.** The live runner's first tick fires before any action exists, and Kinetix objects move from sim t = 0.
  `KinetixBenchmark(start_on_first_action=true)` does not step on a hold before the first policy action, so the first
  action starts env time (exactly one pre-start tick per episode), matching the paper's undelayed chunk 0.
  An earlier sync check, which replayed one zero-action step before step 0 instead of gating, was run and dropped.
- **Hold action.** Kinetix v3 actions are rates (motor a = 0 applies no impulse; thrusters scale a force), so a stale
  tick holds with zero, Kinetix's `noop_action`, not the last action.

## Reference: R vs Fig. 5 (paper metric)

Cells are R / Fig. 5, the 12-level average. Fig. 5 is read from the vector PDF (±0.003); R's 95% CI is ≈ ±0.4 pp.
d sweep, s = max(d, 1):

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| rtc | 0.922 / 0.922 | 0.896 / 0.893 | 0.847 / 0.843 | 0.743 / 0.747 | 0.612 / 0.618 |
| rtc_hard | 0.917 / 0.916 | 0.844 / 0.849 | 0.817 / 0.817 | 0.732 / 0.732 | 0.612 / 0.619 |
| bid | 0.916 / 0.921 | 0.883 / 0.877 | 0.817 / 0.801 | 0.692 / 0.673 | 0.558 / 0.552 |
| naive | 0.916 / 0.915 | 0.754 / 0.753 | 0.709 / 0.708 | 0.585 / 0.584 | 0.490 / 0.493 |

d = 1, s sweep:

| method | s=1 | s=2 | s=3 | s=4 | s=5 | s=6 | s=7 |
|---|---|---|---|---|---|---|---|
| rtc | 0.896 / 0.893 | 0.878 / 0.878 | 0.862 / 0.863 | 0.847 / 0.845 | 0.834 / 0.834 | 0.817 / 0.821 | 0.796 / 0.796 |
| rtc_hard | 0.844 / 0.848 | 0.867 / 0.866 | 0.855 / 0.856 | 0.849 / 0.850 | 0.833 / 0.830 | 0.819 / 0.821 | 0.796 / 0.796 |
| naive | 0.754 / 0.753 | 0.813 / 0.812 | 0.786 / 0.787 | 0.785 / 0.784 | 0.746 / 0.753 | 0.749 / 0.754 | 0.719 / 0.716 |

R is within 0.7 pp of Fig. 5 for RTC, hard masking and naive; BID is 1.6–1.9 pp above it at d = 2 and 3.

## Success metric: Kinetix v3 vs `cf7453e`

vla-eval pins Kinetix v3.0.0; the paper used `cf7453e`. Both run 2 frame-skip substeps per step but read success differently:
- `cf7453e`: `GoalR` from the **last** substep, so a goal touched on substep 1 and lost on substep 2 fails.
- v3: reward and `GoalR` from the **first** substep that ends the episode.

The physics is the same (a fixed 256-action replay agrees to ≈ 1e-6 on non-chaotic levels). Under the official metric
the solve rate sits below the return (e.g. cheetah 0.884 vs 0.994). **R′** keeps the official script and seeds and
logs success as "the terminal step's reward is +1" (`RTC_SOLVED=terminal_reward`); C and L are compared against it.

R′ / R:

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| rtc | 0.967 / 0.922 | 0.940 / 0.896 | 0.887 / 0.847 | 0.781 / 0.743 | 0.642 / 0.612 |
| rtc_hard | 0.961 / 0.917 | 0.890 / 0.844 | 0.857 / 0.817 | 0.770 / 0.732 | 0.643 / 0.612 |
| bid | 0.965 / 0.916 | 0.927 / 0.883 | 0.858 / 0.817 | 0.729 / 0.692 | 0.585 / 0.558 |
| naive | 0.961 / 0.916 | 0.795 / 0.754 | 0.746 / 0.709 | 0.613 / 0.585 | 0.515 / 0.490 |

## Results

Criteria, fixed in advance: (1) the level-average difference lies within its 95% CI; (2) at most 2 of 12 levels lie
outside their 95% CI (Newcombe) and, for 2048-rollout arms, none is off by more than 5 pp.

### C (sync, 2048 × 12) vs R′

| condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 2046 | 0.961 | 0.955 | -0.55 ± 0.33 | trampoline -2.5 | 2.5 | FAIL | pass |
| naive d=1 | 2037 | 0.795 | 0.804 | +0.90 ± 0.60 | trampoline +14.9 | 14.9 | FAIL | FAIL |
| naive d=2 | 2048 | 0.746 | 0.748 | +0.17 ± 0.67 | — | 1.5 | pass | pass |
| naive d=3 | 2048 | 0.613 | 0.612 | -0.05 ± 0.73 | — | 2.2 | pass | pass |
| naive d=4 | 2047 | 0.515 | 0.513 | -0.15 ± 0.74 | — | 1.6 | pass | pass |
| rtc d=0 | 2046 | 0.967 | 0.963 | -0.43 ± 0.30 | hard_lunar_lander -0.6, trampoline -3.1 | 3.1 | FAIL | pass |
| rtc d=1 | 2048 | 0.940 | 0.937 | -0.30 ± 0.38 | catcher_v3 -0.9 | 1.9 | pass | pass |
| rtc d=2 | 2048 | 0.887 | 0.882 | -0.50 ± 0.51 | hard_lunar_lander -0.9, trampoline -2.4 | 2.4 | pass | pass |
| rtc d=3 | 2048 | 0.781 | 0.779 | -0.22 ± 0.67 | — | 2.1 | pass | pass |
| rtc d=4 | 2048 | 0.642 | 0.640 | -0.12 ± 0.73 | hard_lunar_lander -3.3 | 3.3 | pass | pass |
| rtc_hard d=0 | 2048 | 0.961 | 0.956 | -0.43 ± 0.33 | mjc_walker -1.6, trampoline -3.8 | 3.8 | FAIL | pass |
| rtc_hard d=1 | 2048 | 0.890 | 0.890 | -0.04 ± 0.48 | trampoline +3.3 | 3.3 | pass | pass |
| rtc_hard d=2 | 2048 | 0.857 | 0.853 | -0.41 ± 0.56 | — | 1.8 | pass | pass |
| rtc_hard d=3 | 2048 | 0.770 | 0.765 | -0.45 ± 0.68 | chain_lander +0.5 | 1.9 | pass | pass |
| rtc_hard d=4 | 2048 | 0.643 | 0.637 | -0.53 ± 0.73 | hard_lunar_lander -3.0 | 3.0 | pass | pass |

### L (live, 512 × 12, 10 Hz, `latency=wall`) vs C — same env and metric

| condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 512 | 0.955 | 0.957 | +0.16 ± 0.53 | — | 2.9 | pass | pass |
| naive d=1 | 512 | 0.804 | 0.807 | +0.30 ± 0.95 | — | 2.5 | pass | pass |
| naive d=2 | 512 | 0.748 | 0.751 | +0.29 ± 1.05 | — | 3.3 | pass | pass |
| naive d=3 | 512 | 0.612 | 0.605 | -0.76 ± 1.15 | — | 2.8 | pass | pass |
| naive d=4 | 512 | 0.513 | 0.516 | +0.30 ± 1.16 | catapult -5.7, chain_lander -1.4 | 5.7 | pass | pass |
| rtc d=0 | 512 | 0.963 | 0.965 | +0.19 ± 0.48 | — | 2.9 | pass | pass |
| rtc d=1 | 512 | 0.937 | 0.937 | +0.04 ± 0.61 | grasp_easy -1.7 | 3.2 | pass | pass |
| rtc d=2 | 512 | 0.882 | 0.881 | -0.20 ± 0.83 | — | 2.0 | pass | pass |
| rtc d=3 | 512 | 0.779 | 0.787 | +0.84 ± 1.05 | mjc_swimmer +6.1 | 6.1 | pass | pass |
| rtc d=4 | 512 | 0.640 | 0.633 | -0.72 ± 1.16 | — | 4.5 | pass | pass |

### L vs R′

| condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 512 | 0.961 | 0.957 | -0.40 ± 0.52 | trampoline -3.6 | 3.6 | pass | pass |
| naive d=1 | 512 | 0.795 | 0.807 | +1.20 ± 0.95 | trampoline +17.4 | 17.4 | FAIL | pass |
| naive d=2 | 512 | 0.746 | 0.751 | +0.46 ± 1.05 | — | 2.8 | pass | pass |
| naive d=3 | 512 | 0.613 | 0.605 | -0.81 ± 1.16 | — | 4.5 | pass | pass |
| naive d=4 | 512 | 0.515 | 0.516 | +0.14 ± 1.16 | catapult -4.8, chain_lander -1.5 | 4.8 | pass | pass |
| rtc d=0 | 512 | 0.967 | 0.965 | -0.24 ± 0.47 | hard_lunar_lander -0.8 | 1.7 | pass | pass |
| rtc d=1 | 512 | 0.940 | 0.937 | -0.26 ± 0.61 | — | 2.2 | pass | pass |
| rtc d=2 | 512 | 0.887 | 0.881 | -0.70 ± 0.82 | hard_lunar_lander -1.9 | 2.1 | pass | pass |
| rtc d=3 | 512 | 0.781 | 0.787 | +0.62 ± 1.05 | mjc_swimmer +6.9 | 6.9 | pass | pass |
| rtc d=4 | 512 | 0.642 | 0.633 | -0.84 ± 1.16 | hard_lunar_lander -6.5 | 6.5 | pass | pass |

### Live instruments

Pass marks: effective delay = d for ≥ 99.5% of boundaries, stale ticks ≤ 0.5%, inference every s steps, every env
step < Δt/2.

| condition | delay = d (off by: count) | stale ticks | pre-start/ep | cadence misses | worst step ms | eps > Δt/2 |
|---|---|---|---|---|---|---|
| naive d=0 | 100.000% (—) | 12 (0.002%) | 1.00 | 0 | 171 | 13 |
| naive d=1 | 99.999% (1: 5) | 17 (0.003%) | 1.00 | 0 | 271 | 36 |
| naive d=2 | 99.996% (-1: 11, 1: 3) | 1 (0.000%) | 1.00 | 0 | 272 | 16 |
| naive d=3 | 99.991% (-2: 1, -1: 7, 1: 11, 2: 1) | 15 (0.002%) | 1.00 | 0 | 269 | 35 |
| naive d=4 | 99.982% (-3: 1, -2: 1, -1: 20, 1: 10) | 8 (0.001%) | 1.00 | 0 | 267 | 45 |
| rtc d=0 | 100.000% (—) | 15 (0.003%) | 1.00 | 0 | 295 | 26 |
| rtc d=1 | 100.000% (1: 1) | 9 (0.002%) | 1.00 | 0 | 169 | 21 |
| rtc d=2 | 99.992% (-1: 11, 1: 12) | 7 (0.001%) | 1.00 | 0 | 322 | 20 |
| rtc d=3 | 99.995% (-1: 5, 1: 6) | 2 (0.000%) | 1.00 | 0 | 136 | 18 |
| rtc d=4 | 99.960% (-3: 1, -2: 2, -1: 39, 1: 25) | 17 (0.002%) | 1.00 | 0 | 257 | 99 |

Only the last mark misses: step-time spikes (136–322 ms, mid-episode, shared node; JIT is pre-compiled) shift a few
boundaries by ±1–3 steps. Publish to apply ≈ Δt − ε (p50 ≈ 95 ms); inference p99 ≈ 10–25 ms on one CPU thread.

### Trampoline (R′ / C / L)

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| naive | 0.856 / 0.831 / 0.820 | 0.488 / 0.637 / 0.662 | 0.787 / 0.780 / 0.801 | 0.720 / 0.709 / 0.719 | 0.724 / 0.708 / 0.699 |
| rtc | 0.931 / 0.900 / 0.914 | 0.895 / 0.883 / 0.873 | 0.849 / 0.825 / 0.828 | 0.829 / 0.808 / 0.812 | 0.809 / 0.800 / 0.779 |
| rtc_hard | 0.856 / 0.818 | 0.694 / 0.727 | 0.847 / 0.834 | 0.815 / 0.803 | 0.809 / 0.799 |

## Deviations from the paper

- Kinetix v3.0.0 instead of `cf7453e`: different success metric (above); trampoline residual open.
- RNG streams differ (server: `sha256(episode_id)`, benchmark: `seed + episode_idx`): trajectories match only statistically.
- L uses 512 episodes per level (C uses the paper's 2048).
- Live runs at 10 Hz: a 30 Hz pilot overloaded the CPU servers (two rows at s = 1 ≈ 60 inferences/s × 20 ms).
- Live env time starts with the first action (`start_on_first_action`).
- No temporal ensembling (not in the released code). BID is R/R′ only: it needs a GPU server (≈ 185 ms per call on CPU).

## Reproducing

```bash
# server, CPU (JAX_PLATFORMS=cpu, single-threaded XLA): one per 2 shards
vla-eval serve -c configs/model_servers/rtc/rtc.yaml --arg delay=2 --arg horizon=2 --arg latency=steps
# C (sync); L (live) uses the server with --arg latency=wall
vla-eval run -c configs/benchmarks/kinetix/eval.yaml --runtime charliecloud --render cpu
vla-eval run -c configs/benchmarks/kinetix/realtime.yaml --runtime charliecloud --render cpu
# R / R′ (RTC_SOLVED=terminal_reward), in the official repo's venv
python data/rtc-kinetix/ref_eval.py <rtc>/src <run_dir with 7/ and 31/> out.jsonl core
# one run -> sync/ or live/<method>_d<d>.json; then all tables
python data/rtc-kinetix/report.py analyze recording-<eval_id>.sqlite <d> <s> [--live] -o run.json
python data/rtc-kinetix/report.py tables
```

Use a tmpfs `output_dir` (e.g. `/dev/shm`) for many shards: 60 SQLite writers on local disk starved past the 60 s
busy timeout and lost 7–31 of 24 576 episodes in four C runs. For live runs, pin one core per shard and per server.

## Compute

| Arm | Where | Time |
|-----|-------|------|
| R, R′ (38 configs each) | 6 × H200 (CORE helium) | ≈ 65–150 s per config; BID ≈ 6 min |
| C (15 conditions × 24 576 episodes) | hydrogen CPU, 40 shards + 20 servers | ≈ 25–45 min per condition |
| L (10 conditions × 6144 episodes) | hydrogen CPU, 48 shards + 24 servers, pinned | ≈ 20–30 min per condition |

## Open issues

- Trampoline differs between Kinetix versions: C is 1–4 pp below R′ in most conditions, and C and L are 15–17 pp
  above it at naive d = 1. L agrees with C there, so it is env-side; pinning `cf7453e` in the image would settle it.
- Not run yet: L at 2048 rollouts, L for hard masking and BID (GPU server), the horizon sweep, and the realism variants
  (`wall_exact`, `first_chunk_delayed`).
