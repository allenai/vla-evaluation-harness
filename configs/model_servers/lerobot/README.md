# LeRobot model server

Bridges any 🤗 [LeRobot](https://github.com/huggingface/lerobot) `PreTrainedPolicy`
checkpoint into vla-eval: the VLA families (π₀ / π₀.₅, GR00T N1.7, X-VLA,
MolmoAct2, SmolVLA, …) and LeRobot's world-model policies (FastWAM, VLA-JEPA,
LingBot-VA). One script env serves all of them; the PEP 723 header bundles every
policy family's extras.

## Usage

```bash
vla-eval serve -c configs/model_servers/lerobot/pi05_libero.yaml
```

## Supported policies

The bridge loads any single-observation-step policy by registry name.
Multi-obs-step policies (`diffusion`, `vqbet`, `multi_task_dit`, `tdmpc`) need an
observation history that chunk buffering skips, so they are rejected at load.

Verification levels, all on an A100 at the pinned lerobot v0.6.0: "smoke" =
checkpoint load + stub-episode inference; "10-ep" = 10 LIBERO episodes
(1 per task) on the suite the checkpoint targets, compared against LeRobot's
published number where one exists.

| policy_type | category | config | example checkpoint | verified |
|---|---|---|---|---|
| `pi05` | VLA | `pi05_libero.yaml` | `lerobot/pi05_libero_finetuned` | LIBERO Object reproduced 100/100 (50 eps); 10-ep Object 10/10 |
| `molmoact2` | VLA | `molmoact2_libero.yaml` | `allenai/MolmoAct2-LIBERO` | 10-ep Goal 10/10 (published 98.0) |
| `vla_jepa` | world model | `vla_jepa_libero.yaml` | `lerobot/VLA-JEPA-LIBERO` | 10-ep LIBERO-10 10/10 (published 93.0) |
| `pi0` | VLA | `pi0.yaml` | `lerobot/pi0` | untested (`lerobot/pi0_libero_finetuned` scored 0/10 here and has no published score) |
| `groot` (N1.7) | VLA | `groot_n17.yaml` | `nvidia/GR00T-N1.7-LIBERO` | smoke; 10-ep Object 0/10, obs/action convention gap under investigation. Backbone repo `nvidia/Cosmos-Reason2-2B` is gated |
| `xvla` | VLA | `xvla.yaml` | `lerobot/xvla-base` | smoke (base checkpoint; no benchmark-ready finetune published) |
| `smolvla` | VLA | `smolvla.yaml` | `lerobot/smolvla_base` | smoke (base checkpoint; no benchmark-ready finetune published) |
| `fastwam` | world model | `fastwam_libero.yaml` | `ZibinDong/fastwam_libero_uncond_2cam224` | smoke; 10-ep LIBERO-10 0/10 (published 94.0), not reproduced through the bridge yet |
| `lingbot_va` | world model | `lingbot_va_libero.yaml` | `lerobot/lingbot_va_libero_long` | smoke; 10-ep LIBERO-10 0/10, suspected chunk-cadence mismatch with its autoregressive video state |

Other single-obs-step registry names (`act`, `pi0_fast`, `eo1`, `evo1`, `wall_x`)
load through the same path but ship no config here and are untested.

## Config args

| arg | meaning |
|-----|---------|
| `policy_type` | LeRobot registry name (see table above) |
| `checkpoint` | HuggingFace Hub id or local path |
| `image_keys` | benchmark camera name → policy image feature (auto-mapped positionally if omitted) |
| `state_key` | proprioceptive state feature, or `null` for none |
| `device` | torch device (default `cuda`) |
| `chunk_size` | actions buffered per inference; `null` = policy's `n_action_steps` |
| `compile_model` | override the checkpoint's torch.compile setting; `null` keeps it |
| `policy_kwargs` | extra policy-config fields, e.g. GR00T's `embodiment_tag`, MolmoAct2's `norm_tag` / `inference_action_mode` |
| `features` | input/output feature spec for original-format checkpoints that carry none (see `molmoact2_libero.yaml`) |

## How it works

The checkpoint is loaded with its saved pre/post processors, and each step runs
`predict_action_chunk` → the full `(n_action_steps, action_dim)` chunk is returned
and buffered per-session by the harness (same pattern as the π₀ server). This is
concurrency-safe under sharding for single-observation-step policies, the only
kind supported.

Checkpoints stored in a policy's original (non-LeRobot) format load through a
fallback chain: the policy class's own loader (GR00T's `from_pretrained` consumes
the NVIDIA training dumps directly), or a config built from `policy_kwargs` +
`features` for policies that take the source via a config field (MolmoAct2's
`checkpoint_path` + `norm_tag`). GR00T's LIBERO checkpoints live in per-suite
subfolders of `nvidia/GR00T-N1.7-LIBERO`; download one and point `checkpoint`
at the local dir (see `groot_n17.yaml`).

## Status

Pinned to lerobot [v0.6.0](https://github.com/huggingface/lerobot/releases/tag/v0.6.0);
every "smoke" row in the table above was verified at the tag on an A100.

Score validation on an H100 at `e275ea3` (9 commits before v0.6.0; additive diff):
a LIBERO Object reproduction with `lerobot/pi05_libero_finetuned`: **100/100 success**
(LeRobot reports 99.0, OpenPI 98.2). See `pi05_libero.yaml`.

`image_keys` / `state_key` / the action convention are checkpoint-specific;
verify them per checkpoint and benchmark.

Keep the checkpoint's `compile_model` setting for real runs: pi05 chunk inference
on an A100 measures 0.11 s compiled vs 0.49 s eager (4.4x). The cost is the first
inference (tens of seconds on H100; ~1.5 min warm / up to ~8 min cold inductor
cache on A100 at `max-autotune`), so raise the benchmark's `server.timeout` for
the initial episodes; this also exceeds `vla-eval test`'s fixed smoke timeout.
Set `compile_model: false` only for quick debug servers. Inference speed varies
widely per policy; measured per-chunk on an A100: pi05 0.11 s, VLA-JEPA 0.20 s,
GR00T N1.7 0.24 s, X-VLA 0.26 s, MolmoAct2 0.29 s, SmolVLA 0.31 s, FastWAM
0.77 s, and LingBot-VA (video-generation world model) ~21 s.
