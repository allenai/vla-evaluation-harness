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

Nothing to set up: the official repo is cloned at `9296f31` and the public `gs://rtc-assets` checkpoints are fetched into the vla-eval assets cache on first use. Pass `--arg rtc_src_path=<clone>/src` or a local `--arg checkpoint_dir=<dir>` to use your own.

```bash
vla-eval serve -c configs/model_servers/rtc/rtc.yaml --arg delay=2 --arg horizon=2
```

The paper evaluates the last epoch (`bc/31`); `bc/24` is the README's "pre-trained" checkpoint for training-time RTC and is not used here.
