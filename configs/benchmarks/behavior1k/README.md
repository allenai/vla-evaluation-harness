---
smoke_config: null  # requires Isaac Sim + R1Pro
---

# BEHAVIOR-1K (2026 challenge protocol)

Large-scale household activity benchmark (OmniGibson/Isaac Sim).
[Paper](https://arxiv.org/abs/2403.09227) | [GitHub](https://github.com/StanfordVL/BEHAVIOR-1K) |
[Challenge evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html)

**Docker image:** `ghcr.io/allenai/vla-evaluation-harness/behavior1k:latest` (local build only, NVIDIA EULA),
built FROM upstream's official `stanfordvl/behavior:3.9.4` (pinned by digest)

**Protocol version:** BEHAVIOR Challenge 2026 stack: BEHAVIOR-1K `v3.9.4`, Isaac Sim 5.1.0,
Python 3.11, 100 tasks (B100). Episodes run through upstream's `omnigibson.eval.evaluator.BatchedEvaluator`,
so instance loading, timeouts and metrics follow `python -m omnigibson.eval.eval`, and each rollout gets the
challenge's wall-clock budget (`max_steps` seconds after a free first query); only the policy transport is the
harness WebSocket protocol. The 2025 stack (v3.7.2, B50) is no longer supported.

Requires an R1Pro-compatible model server. See [behavior1k.md](../../../docs/reproductions/behavior1k.md).

## Configs

| File | Description | Tasks | Episodes/task |
|------|-------------|:-----:|:-------------:|
| `eval.yaml` | Challenge self-evaluation: public-test indices 0-9, one rollout each | 100 | 10 |
| `demo_replay.yaml` | Replays training demos on their own instances (pair with `model_servers/behavior1k/demo_replay.yaml`) | 2 | 1 |

`env_wrapper` picks the observation profile: `DefaultWrapper` (upstream CLI default, 224x224 RGB) or
`RGBDFullResWrapper` (official RGB-D track, 720x720 head / 480x480 wrists, depth sent as `obs["depth"]`).
