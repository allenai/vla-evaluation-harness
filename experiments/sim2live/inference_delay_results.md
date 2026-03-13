# Inference Delay Sweep Results — RTC Kinetix Sim2Live Evaluation

**Date:** 2026-03-10
**Model:** RTC flow-matching diffusion policy (BC checkpoint, denoising_steps=5)
**Config:** `pace: 1.0` (real-time, 10 Hz), `wait_first_action=true`, `hold_policy=repeat_last`, `action_noise_std=0.1`, `chunk_size=8`
**Episodes:** 50 per task (N=600 per delay, 12 tasks)

## Overall Results

| Delay | Actual Inference | Overall | Δ from 0ms |
|-------|-----------------|---------|------------|
| 0ms | ~2ms | 56.0% | baseline |
| 50ms | ~52ms | 56.8% | +0.8pp |
| 100ms | ~102ms | 39.8% | −16.2pp |
| 200ms | ~202ms | 28.5% | −27.5pp |
| 500ms | ~502ms | 10.8% | −45.2pp |

## Per-Task Breakdown

| Task | 0ms | 50ms | 100ms | 200ms | 500ms |
|------|:---:|:----:|:-----:|:-----:|:-----:|
| Grasp Easy | 58 | 64 | 54 | 44 | 32 |
| Catapult | 34 | 20 | 18 | 20 | 6 |
| Cartpole Thrust | 50 | 40 | 14 | 12 | 2 |
| Hard Lunar Lander | 34 | 38 | 26 | 14 | 10 |
| Half Cheetah | 94 | 96 | 90 | 78 | 6 |
| Swimmer | 24 | 34 | 2 | 0 | 0 |
| Walker | 36 | 24 | 8 | 2 | 0 |
| Unicycle | 84 | 94 | 62 | 18 | 4 |
| Chain Lander | 100 | 100 | 98 | 92 | 50 |
| Catcher | 58 | 70 | 34 | 18 | 6 |
| Trampoline | 44 | 58 | 44 | 34 | 6 |
| Car Launch | 56 | 44 | 28 | 10 | 8 |
| **Overall** | **56.0** | **56.8** | **39.8** | **28.5** | **10.8** |

## CI/LAAS Mitigation Results (pace=1.0, 0ms delay)

| Condition | Grasp | Catapult | Cartpole | Lunar | Cheetah | Swimmer | Walker | Unicycle | Chain | Catcher | Tramp. | Car | **Avg** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Sync | 62 | 26 | 50 | 36 | 96 | 32 | 32 | 90 | 100 | 68 | 58 | 64 | 59.5 |
| Sim2Live | 58 | 34 | 50 | 34 | 94 | 24 | 36 | 84 | 100 | 58 | 44 | 56 | 56.0 |
| + CI | 80 | 40 | 98 | 80 | 96 | 100 | 42 | 98 | 100 | 80 | 64 | 86 | 80.3 |
| + LAAS | 92 | 20 | 98 | 74 | 96 | 100 | 42 | 94 | 100 | 92 | 64 | 84 | 79.7 |

## Key Findings

1. **Sharp cliff at ~100ms**: Performance is stable up to 50ms inference delay
   (well within the 100ms step period) but drops sharply once inference exceeds
   the step period. The transition from 50ms (56.8%) to 100ms (39.8%) is a
   17pp cliff.

2. **Task sensitivity to latency**:
   - **Latency-robust**: Chain Lander (100→50%), Half Cheetah (94→6% but only at 500ms),
     Grasp Easy (58→32%) — gradual degradation.
   - **Latency-sensitive**: Swimmer (24→0% at 200ms), Walker (36→0% at 500ms),
     Cartpole Thrust (50→2%) — collapse once delay exceeds step period.
   - **Unicycle** is notable: 84% at 0ms, 94% at 50ms, but crashes to 4% at 500ms.

3. **CI/LAAS effectiveness**: At real-time pace with fast inference (0ms delay),
   CI and LAAS boost performance by +24pp (56.0→80.3%). This demonstrates these
   techniques are most valuable in the real-time regime where the step period
   provides sufficient headroom for overlapped inference.

4. **Statistical confidence**: N=600 per condition (12 tasks × 50 episodes),
   SE ≈ 2.0pp, 95% CI ≈ ±4pp. The 0ms→100ms drop (16.2pp ≈ 8 SE) is
   highly significant.
