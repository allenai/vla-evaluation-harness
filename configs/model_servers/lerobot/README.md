# LeRobot model server

Bridges any 🤗 [LeRobot](https://github.com/huggingface/lerobot) `PreTrainedPolicy`
checkpoint (pi0, pi05, ACT, diffusion, SmolVLA, …) into vla-eval.

## Usage

```bash
vla-eval serve -c configs/model_servers/lerobot/pi0.yaml
```

Install the lerobot extra for your policy, e.g. `lerobot[pi0]` or `lerobot[smolvla]`;
plain `lerobot` does not pull the heavy policy backends.

## Config args

| arg | meaning |
|-----|---------|
| `policy_type` | LeRobot registry name (`pi0`, `pi05`, `act`, `diffusion`, `smolvla`, …) |
| `checkpoint` | HuggingFace Hub id or local path |
| `image_keys` | benchmark camera name → policy image feature (auto-mapped positionally if omitted) |
| `state_key` | proprioceptive state feature, or `null` for none |
| `device` | torch device (default `cuda`) |
| `chunk_size` | actions buffered per inference; `null` = policy's `n_action_steps` |

## How it works

The checkpoint is loaded with its saved pre/post processors, and each step runs
`predict_action_chunk` → the full `(n_action_steps, action_dim)` chunk is returned
and buffered per-session by the harness (same pattern as the π₀ server). This is
concurrency-safe under sharding for single-observation-step policies (pi0, pi05,
SmolVLA, ACT), the only kind supported. Multi-obs-step policies (diffusion,
VQ-BeT) need an observation history that chunk buffering skips, so they are
rejected at load (`n_obs_steps > 1`).

## Status

Implemented against lerobot @ [`e275ea3`](https://github.com/huggingface/lerobot/tree/e275ea3960332543e2a9f441356775a53720543f)
(pinned) — an ancestor of [v0.6.0](https://github.com/huggingface/lerobot/releases/tag/v0.6.0),
9 commits before the tag; the intervening commits only add new policies (Gr00t N1.7, EVO1)
and infra, so the pin carries the v0.6.0 processor-pipeline inference API.

Validated on an H100: checkpoint load + inference smoke, and a LIBERO Object
score reproduction with `lerobot/pi05_libero_finetuned`: **100/100 success**
(LeRobot reports 99.0, OpenPI 98.2). See `pi05_libero.yaml`.

`image_keys` / `state_key` / the action convention are checkpoint-specific;
verify them per checkpoint and benchmark. First inference triggers
torch.compile / CUDA-graph capture (tens of seconds); raise the benchmark's
`server.timeout` for the initial episodes under multi-shard runs.
