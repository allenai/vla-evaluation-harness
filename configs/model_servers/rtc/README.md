---
smoke_config: rtc.yaml
---

# RTC (Real-Time Chunking)

Real-time action chunking for latency-aware control. [Paper](https://arxiv.org/abs/2506.07339) | [GitHub](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)

`src/vla_eval/model_servers/rtc.py` runs the official `eval_flow.py` chunk protocol (inference every `horizon` steps, `delay` steps covered by the previous chunk) with the official sampling code, in sync or live mode. Results: [docs/reproductions/rtc.md](../../../docs/reproductions/rtc.md).

## Configs

| File | Method | Notes |
|------|--------|-------|
| `naive.yaml` | Naive async | `policy.action` |
| `rtc.yaml` | RTC, soft masking | `realtime_action`, `exp` schedule, β = 5 |
| `rtc_hard.yaml` | RTC, hard masking | `zeros` schedule |
| `bid.yaml` | BID | N = 32, K = 3, weak policy `bc/7` |

## Setup

```bash
git clone https://github.com/Physical-Intelligence/real-time-chunking-kinetix third_party/real-time-chunking-kinetix
git -C third_party/real-time-chunking-kinetix checkout 9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b
for e in 31 7; do mkdir -p checkpoints/rtc-kinetix/bc/$e/policies; for l in car_launch cartpole_thrust catapult \
  catcher_v3 chain_lander grasp_easy h17_unicycle hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker trampoline; do
  curl -fL -o checkpoints/rtc-kinetix/bc/$e/policies/worlds_l_$l.pkl \
    "https://storage.googleapis.com/download/storage/v1/b/rtc-assets/o/bc%2F$e%2Fpolicies%2Fworlds_l_$l.pkl?alt=media"
done; done
vla-eval serve -c configs/model_servers/rtc/rtc.yaml --arg delay=2 --arg horizon=2
```

The paper evaluates the last epoch (`bc/31`); `bc/24` is the README's "pre-trained" checkpoint for training-time RTC and is not used here.
