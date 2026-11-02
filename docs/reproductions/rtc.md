# RTC on Kinetix — Reproduction Report

Real-Time Chunking (RTC) for action-chunking flow policies. [Paper](https://arxiv.org/abs/2506.07339) |
[GitHub](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | 12 Kinetix levels, flow-matching MLP-Mixer, H = 8.

Beyond Fig. 5, this checks that vla-eval's **live** mode (wall-clock ticks, injected latency) matches the paper's
**sync** protocol, where the delay is counted in steps.

## Results Summary

| Arm | What | Result |
|-----|------|--------|
| R | Official `eval_flow.py`, run by us (2048 rollouts × 12 levels, `bc/31`, BID weak `bc/7`) | ✅ matches Fig. 5 (≤ 0.7 pp, BID ≤ 1.9 pp) |
| R′ | R with success read as Kinetix v3 reads it and f32 physics (see *Success metric and numerics*) | the reference vla-eval is held to |
| C | vla-eval **sync**, delay counted in steps (2048 × 12) | ✅ vs R′: 17/20 conditions pass both criteria, mean diff −0.14 pp, \|diff\| ≤ 0.89 pp |
| L | vla-eval **live**, 10 Hz, latency (d − ½)/hz (2048 × 12) | ✅ vs C: 18/20 pass, \|diff\| ≤ 0.63 pp; vs R′: 19/20, mean diff +0.03 pp |

Conditions: naive, RTC (soft), RTC hard masking and BID at d = 0–4, s = max(d, 1). Every failure is criterion (1),
at most 0.16 pp outside its CI; criterion (2) holds everywhere.

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

## Success metric and numerics: Kinetix v3 vs `cf7453e`

vla-eval pins Kinetix v3.0.0; the paper used `cf7453e`. Both run 2 frame-skip substeps per step but read success differently:
- `cf7453e`: `GoalR` from the **last** substep, so a goal touched on substep 1 and lost on substep 2 fails.
- v3: reward and `GoalR` from the **first** substep that ends the episode.

The physics is the same (a fixed 256-action replay agrees to ≈ 1e-6 on non-chaotic levels). Under the official metric
the solve rate sits below the return (e.g. cheetah 0.884 vs 0.994).

**Trampoline at naive d = 1 is bistable in floating point.** Same script, seeds and checkpoints:

| trampoline, naive d = 1 (2048 rollouts, v3 metric) | solved |
|---|---|
| official script, 6 × H200 (R's run) | 0.488 |
| official script, 1 × H100 | 0.652 |
| official script, 1 × H100, f32 physics dots (R′) | 0.652 |
| C / L (CPU, f32) | 0.637 / 0.642 |

The other 11 levels move ≤ 1 pp. It is not the Kinetix version: `eval_flow`'s loop on v3.0.0 also gives 0.488 on GPU
at 256 rollouts and 0.65 with f32 physics dots or on CPU; f32 dots in the policy alone stay at 0.488. The GPU's
default f32 matmul precision in the physics, whose kernels depend on batch size and device count, selects the mode.

**R′** keeps the official script and seeds, reads success as v3 does (`RTC_SOLVED=first_terminal`) and runs the physics
dots in f32 (`RTC_PHYSICS_PRECISION=highest`), see [`ref_eval.py`](data/rtc-kinetix/ref_eval.py). C and L are
compared against it. R′ / R (1.4 pp of the naive d = 1 gap is trampoline's mode):

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| rtc | 0.965 / 0.922 | 0.939 / 0.896 | 0.887 / 0.847 | 0.779 / 0.743 | 0.643 / 0.612 |
| rtc_hard | 0.960 / 0.917 | 0.893 / 0.844 | 0.853 / 0.817 | 0.766 / 0.732 | 0.643 / 0.612 |
| bid | 0.963 / 0.916 | 0.925 / 0.883 | 0.857 / 0.817 | 0.726 / 0.692 | 0.581 / 0.558 |
| naive | 0.959 / 0.916 | 0.807 / 0.754 | 0.746 / 0.709 | 0.609 / 0.585 | 0.515 / 0.490 |

## Results

Criteria, fixed in advance: (1) the level-average difference lies within its 95% CI; (2) at most 2 of 12 levels lie
outside their 95% CI (Newcombe) and none is off by more than 5 pp.

### C (sync, 2048 × 12) vs R′

| condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 2046 | 0.959 | 0.955 | -0.36 ± 0.33 | — | 2.2 | FAIL | pass |
| naive d=1 | 2037 | 0.807 | 0.804 | -0.25 ± 0.60 | — | 1.5 | pass | pass |
| naive d=2 | 2048 | 0.746 | 0.748 | +0.18 ± 0.67 | — | 1.6 | pass | pass |
| naive d=3 | 2048 | 0.609 | 0.612 | +0.30 ± 0.73 | — | 2.6 | pass | pass |
| naive d=4 | 2047 | 0.515 | 0.513 | -0.13 ± 0.74 | — | 1.4 | pass | pass |
| rtc d=0 | 2046 | 0.965 | 0.963 | -0.20 ± 0.30 | — | 1.7 | pass | pass |
| rtc d=1 | 2048 | 0.939 | 0.937 | -0.20 ± 0.38 | catcher_v3 -0.9 | 1.7 | pass | pass |
| rtc d=2 | 2048 | 0.887 | 0.882 | -0.50 ± 0.51 | mjc_swimmer -1.4 | 1.4 | pass | pass |
| rtc d=3 | 2048 | 0.779 | 0.779 | -0.01 ± 0.67 | — | 1.8 | pass | pass |
| rtc d=4 | 2048 | 0.643 | 0.640 | -0.28 ± 0.73 | — | 2.2 | pass | pass |
| rtc_hard d=0 | 2048 | 0.960 | 0.956 | -0.38 ± 0.33 | mjc_walker -1.6 | 2.2 | FAIL | pass |
| rtc_hard d=1 | 2048 | 0.893 | 0.890 | -0.36 ± 0.47 | mjc_walker -1.9 | 1.9 | pass | pass |
| rtc_hard d=2 | 2048 | 0.853 | 0.853 | +0.00 ± 0.56 | — | 2.5 | pass | pass |
| rtc_hard d=3 | 2048 | 0.766 | 0.765 | -0.06 ± 0.68 | chain_lander +0.5 | 1.7 | pass | pass |
| rtc_hard d=4 | 2048 | 0.643 | 0.637 | -0.59 ± 0.73 | — | 2.4 | pass | pass |
| bid d=0 | 2048 | 0.963 | 0.961 | -0.24 ± 0.31 | — | 1.9 | pass | pass |
| bid d=1 | 2048 | 0.925 | 0.922 | -0.28 ± 0.41 | h17_unicycle -1.1 | 1.3 | pass | pass |
| bid d=2 | 2048 | 0.857 | 0.854 | -0.32 ± 0.56 | trampoline -2.3 | 2.3 | pass | pass |
| bid d=3 | 2048 | 0.726 | 0.726 | -0.01 ± 0.70 | — | 2.0 | pass | pass |
| bid d=4 | 2048 | 0.581 | 0.590 | +0.89 ± 0.73 | mjc_walker +2.7, chain_lander -0.8 | 2.7 | FAIL | pass |

### L (live, 2048 × 12, 10 Hz, `latency=wall`) vs C — same env and metric

| condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 2048 | 0.955 | 0.958 | +0.30 ± 0.33 | — | 2.3 | pass | pass |
| naive d=1 | 2048 | 0.804 | 0.805 | +0.11 ± 0.60 | — | 2.2 | pass | pass |
| naive d=2 | 2048 | 0.748 | 0.750 | +0.21 ± 0.67 | — | 2.7 | pass | pass |
| naive d=3 | 2048 | 0.612 | 0.610 | -0.28 ± 0.72 | — | 2.4 | pass | pass |
| naive d=4 | 2048 | 0.513 | 0.510 | -0.33 ± 0.74 | — | 2.2 | pass | pass |
| rtc d=0 | 2048 | 0.963 | 0.965 | +0.22 ± 0.30 | — | 2.1 | pass | pass |
| rtc d=1 | 2048 | 0.937 | 0.938 | +0.13 ± 0.38 | — | 1.2 | pass | pass |
| rtc d=2 | 2048 | 0.882 | 0.886 | +0.32 ± 0.52 | — | 2.1 | pass | pass |
| rtc d=3 | 2048 | 0.779 | 0.781 | +0.22 ± 0.67 | chain_lander -0.5 | 2.5 | pass | pass |
| rtc d=4 | 2048 | 0.640 | 0.641 | +0.09 ± 0.74 | — | 1.6 | pass | pass |
| rtc_hard d=0 | 2048 | 0.956 | 0.960 | +0.39 ± 0.33 | trampoline +2.5 | 2.5 | FAIL | pass |
| rtc_hard d=1 | 2048 | 0.890 | 0.893 | +0.36 ± 0.47 | — | 1.7 | pass | pass |
| rtc_hard d=2 | 2048 | 0.853 | 0.859 | +0.63 ± 0.56 | catcher_v3 +2.9 | 2.9 | FAIL | pass |
| rtc_hard d=3 | 2048 | 0.765 | 0.769 | +0.38 ± 0.68 | chain_lander -0.5 | 2.9 | pass | pass |
| rtc_hard d=4 | 2048 | 0.637 | 0.639 | +0.23 ± 0.73 | — | 1.8 | pass | pass |
| bid d=0 | 2048 | 0.961 | 0.961 | +0.02 ± 0.32 | — | 0.8 | pass | pass |
| bid d=1 | 2048 | 0.922 | 0.924 | +0.22 ± 0.41 | h17_unicycle +1.1 | 1.6 | pass | pass |
| bid d=2 | 2048 | 0.854 | 0.859 | +0.48 ± 0.56 | trampoline +2.3 | 2.3 | pass | pass |
| bid d=3 | 2048 | 0.726 | 0.725 | -0.12 ± 0.70 | — | 2.9 | pass | pass |
| bid d=4 | 2048 | 0.590 | 0.587 | -0.27 ± 0.73 | — | 2.0 | pass | pass |

### L vs R′

| condition | n/level | ref | arm | diff pp [95% CI] | levels outside (pp) | worst pp | (1) | (2) |
|---|---|---|---|---|---|---|---|---|
| naive d=0 | 2048 | 0.959 | 0.958 | -0.06 ± 0.33 | — | 0.7 | pass | pass |
| naive d=1 | 2048 | 0.807 | 0.805 | -0.13 ± 0.59 | hard_lunar_lander +1.2, h17_unicycle -3.0 | 3.0 | pass | pass |
| naive d=2 | 2048 | 0.746 | 0.750 | +0.39 ± 0.67 | — | 2.0 | pass | pass |
| naive d=3 | 2048 | 0.609 | 0.610 | +0.02 ± 0.73 | grasp_easy -4.0 | 4.0 | pass | pass |
| naive d=4 | 2048 | 0.515 | 0.510 | -0.46 ± 0.74 | — | 3.0 | pass | pass |
| rtc d=0 | 2048 | 0.965 | 0.965 | +0.02 ± 0.30 | — | 1.1 | pass | pass |
| rtc d=1 | 2048 | 0.939 | 0.938 | -0.06 ± 0.38 | grasp_easy +1.1, catcher_v3 -1.1 | 1.3 | pass | pass |
| rtc d=2 | 2048 | 0.887 | 0.886 | -0.18 ± 0.51 | — | 1.1 | pass | pass |
| rtc d=3 | 2048 | 0.779 | 0.781 | +0.21 ± 0.67 | — | 1.3 | pass | pass |
| rtc d=4 | 2048 | 0.643 | 0.641 | -0.20 ± 0.73 | hard_lunar_lander -3.6 | 3.6 | pass | pass |
| rtc_hard d=0 | 2048 | 0.960 | 0.960 | +0.00 ± 0.32 | — | 0.7 | pass | pass |
| rtc_hard d=1 | 2048 | 0.893 | 0.893 | +0.00 ± 0.47 | — | 1.5 | pass | pass |
| rtc_hard d=2 | 2048 | 0.853 | 0.859 | +0.63 ± 0.56 | mjc_walker +3.7, h17_unicycle -2.2 | 3.7 | FAIL | pass |
| rtc_hard d=3 | 2048 | 0.766 | 0.769 | +0.32 ± 0.68 | — | 2.2 | pass | pass |
| rtc_hard d=4 | 2048 | 0.643 | 0.639 | -0.36 ± 0.73 | — | 2.1 | pass | pass |
| bid d=0 | 2048 | 0.963 | 0.961 | -0.22 ± 0.31 | — | 1.4 | pass | pass |
| bid d=1 | 2048 | 0.925 | 0.924 | -0.07 ± 0.41 | grasp_easy -1.1 | 1.1 | pass | pass |
| bid d=2 | 2048 | 0.857 | 0.859 | +0.16 ± 0.55 | mjc_swimmer -1.6 | 1.9 | pass | pass |
| bid d=3 | 2048 | 0.726 | 0.725 | -0.13 ± 0.70 | — | 1.8 | pass | pass |
| bid d=4 | 2048 | 0.581 | 0.587 | +0.62 ± 0.73 | mjc_half_cheetah +1.8 | 2.7 | pass | pass |

### Live instruments

Pass marks: effective delay = d for ≥ 99.5% of boundaries, stale ticks ≤ 0.5%, inference every s steps, every env
step < Δt/2.

| condition | delay = d (off by: count) | stale ticks | pre-start/ep | cadence misses | worst step ms | eps > Δt/2 |
|---|---|---|---|---|---|---|
| naive d=0 | 100.000% (—) | 70 (0.003%) | 1.00 | 0 | 177 | 97 |
| naive d=1 | 100.000% (—) | 172 (0.007%) | 1.00 | 0 | 571 | 719 |
| naive d=2 | 99.972% (-1: 206, 1: 167) | 58 (0.002%) | 1.00 | 0 | 444 | 407 |
| naive d=3 | 99.988% (-2: 1, -1: 60, 1: 43, 2: 2) | 26 (0.001%) | 1.00 | 0 | 226 | 143 |
| naive d=4 | 99.963% (-2: 4, -1: 134, 1: 123) | 125 (0.004%) | 1.00 | 0 | 217 | 516 |
| rtc d=0 | 100.000% (—) | 88 (0.004%) | 1.00 | 0 | 294 | 138 |
| rtc d=1 | 100.000% (—) | 225 (0.010%) | 1.00 | 0 | 175 | 1285 |
| rtc d=2 | 99.954% (-1: 298, 1: 248) | 145 (0.006%) | 1.00 | 0 | 222 | 502 |
| rtc d=3 | 99.989% (-1: 50, 1: 48) | 6 (0.000%) | 1.00 | 0 | 144 | 144 |
| rtc d=4 | 99.996% (-1: 16, 1: 12) | 4 (0.000%) | 1.00 | 0 | 139 | 57 |
| rtc_hard d=0 | 100.000% (—) | 192 (0.009%) | 1.00 | 0 | 245 | 262 |
| rtc_hard d=1 | 100.000% (—) | 26 (0.001%) | 1.00 | 0 | 158 | 186 |
| rtc_hard d=2 | 99.994% (-1: 47, 1: 32) | 9 (0.000%) | 1.00 | 0 | 155 | 93 |
| rtc_hard d=3 | 99.988% (-1: 53, 1: 48) | 22 (0.001%) | 1.00 | 0 | 182 | 155 |
| rtc_hard d=4 | 99.974% (-1: 90, 1: 86) | 26 (0.001%) | 1.00 | 0 | 167 | 365 |
| bid d=0 | 100.000% (—) | 9 (0.000%) | 1.00 | 0 | 143 | 51 |
| bid d=1 | 100.000% (1: 8) | 109 (0.005%) | 1.00 | 0 | 205 | 275 |
| bid d=2 | 99.994% (-1: 43, 1: 30) | 8 (0.000%) | 1.00 | 0 | 148 | 51 |
| bid d=3 | 99.965% (-2: 3, -1: 168, 1: 135) | 44 (0.002%) | 1.00 | 0 | 279 | 530 |
| bid d=4 | 99.972% (-2: 1, -1: 100, 1: 92, 2: 1) | 115 (0.004%) | 1.00 | 0 | 223 | 299 |

Only the last mark misses: step-time spikes (139–571 ms, mid-episode, shared node; JIT is pre-compiled) shift a few
boundaries by ±1–2 steps, and at d = 1 a late chunk is superseded before use at ≤ 0.06% of boundaries. Publish to apply
≈ Δt − ε (p50 ≈ 96 ms); inference p99 ≈ 10–28 ms on one CPU thread, 5–8 ms for BID on GPU.

### Trampoline (R′ / C / L)

| method | d=0 | d=1 | d=2 | d=3 | d=4 |
|---|---|---|---|---|---|
| naive | 0.842 / 0.831 / 0.837 | 0.652 / 0.637 / 0.642 | 0.781 / 0.780 / 0.781 | 0.708 / 0.709 / 0.702 | 0.712 / 0.708 / 0.698 |
| rtc | 0.917 / 0.900 / 0.909 | 0.881 / 0.883 / 0.883 | 0.837 / 0.825 / 0.845 | 0.808 / 0.808 / 0.821 | 0.797 / 0.800 / 0.800 |
| rtc_hard | 0.840 / 0.818 / 0.843 | 0.723 / 0.727 / 0.731 | 0.829 / 0.834 / 0.833 | 0.798 / 0.803 / 0.812 | 0.797 / 0.799 / 0.810 |
| bid | 0.862 / 0.843 / 0.848 | 0.835 / 0.836 / 0.831 | 0.845 / 0.822 / 0.846 | 0.810 / 0.806 / 0.825 | 0.791 / 0.789 / 0.792 |

## Deviations from the paper

- Kinetix v3.0.0 instead of `cf7453e`: different success metric (above); R′ also runs the physics dots in f32.
- RNG streams differ (server: `sha256(episode_id)`, benchmark: `seed + episode_idx`): trajectories match only statistically.
- Live runs at 10 Hz: a 30 Hz pilot overloaded the CPU servers (two rows at s = 1 ≈ 60 inferences/s × 20 ms).
- Live env time starts with the first action (`start_on_first_action`).
- BID's servers run on GPU (default precision; ≈ 185 ms per call on CPU), the other methods' on CPU.
- No temporal ensembling (not in the released code).

## Reproducing

```bash
# server, CPU (JAX_PLATFORMS=cpu, single-threaded XLA): one per 2 shards; BID on GPU, 4–5 shards per server
vla-eval serve -c configs/model_servers/rtc/rtc.yaml --arg delay=2 --arg horizon=2 --arg latency=steps
# C (sync); L (live) uses the server with --arg latency=wall
vla-eval run -c configs/benchmarks/kinetix/eval.yaml --runtime charliecloud --render cpu
vla-eval run -c configs/benchmarks/kinetix/realtime.yaml --runtime charliecloud --render cpu
# R (paper metric) and R′, in the official repo's venv
python data/rtc-kinetix/ref_eval.py <rtc>/src <run_dir with 7/ and 31/> out.jsonl core
RTC_SOLVED=first_terminal RTC_PHYSICS_PRECISION=highest python data/rtc-kinetix/ref_eval.py ... core
# one run -> sync/ or live/<method>_d<d>.json; then all tables
python data/rtc-kinetix/report.py analyze recording-<eval_id>.sqlite <d> <s> [--live] -o run.json
python data/rtc-kinetix/report.py tables
```

Use a tmpfs `output_dir` (e.g. `/dev/shm`) for many shards: 60 SQLite writers on local disk starved past the 60 s
busy timeout and lost 7–31 of 24 576 episodes in four C runs. For live runs, pin one core per shard and per server.

## Compute

| Arm | Where | Time |
|-----|-------|------|
| R (38 configs) | 6 × H200 (CORE helium) | ≈ 65–150 s per config; BID ≈ 6 min |
| R′ (20 configs) | 1 × H100 (core), 4 × H100 (BID) | ≈ 100–155 s per config; BID ≈ 4–10 min |
| C (15 conditions × 24 576 episodes) | hydrogen CPU, 40 shards + 20 servers | ≈ 25–45 min per condition |
| C, BID (5 conditions) | hydrogen, 40 shards + 8 servers on 2 H100 | ≈ 15–40 min per condition |
| L (15 conditions × 24 576 episodes) | hydrogen CPU, 48 shards + 24 servers, pinned | ≈ 75–105 min per condition |
| L, BID (5 conditions) | hydrogen, 48 shards + 12 servers on 2 H100 | ≈ 75–100 min per condition |

## Open issues

- Not run yet: the horizon sweep (d = 1, s = 2–7) in C and L, and the realism variants (`wall_exact`, `first_chunk_delayed`).

