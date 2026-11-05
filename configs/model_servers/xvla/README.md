---
smoke_config: libero.yaml
---

# X-VLA

Cross-embodiment VLA with domain-specific soft prompts (0.9B, Florence-2). [Paper](https://arxiv.org/abs/2510.10274) | [GitHub](https://github.com/2toinf/X-VLA)

## Configs

| File | Benchmark | Checkpoint |
|------|-----------|------------|
| `libero.yaml` | LIBERO | `2toINF/X-VLA-Libero` |
| `calvin.yaml` | CALVIN | `2toINF/X-VLA-Calvin-ABC_D` |
| `simpler_widowx.yaml` | SimplerEnv WidowX | `2toINF/X-VLA-WidowX` |
| `simpler_google_robot.yaml` | SimplerEnv GR | `2toINF/X-VLA-Google-Robot` |
| `robotwin.yaml` | RoboTwin 2.0 | `2toINF/X-VLA-RoboTwin2` |
| `robotwin_xpolicylab.yaml` | RoboTwin 2.0 | RoboTwin cotrain `X_VLA/RoboTwin-cotrain-aloha_agilex-ee-0` |

Uses `_base.yaml` for shared settings. `domain_id` selects the embodiment prompt.

RoboTwin follows X-VLA's `evaluation/robotwin-2.0/client.py`: the server decodes each
30-action chunk into 16D `ee` actions, builds proprio from the commanded end pose, and
prompts with the task name. Run it with `configs/benchmarks/robotwin/xvla.yaml`.

`robotwin_xpolicylab.yaml` serves RoboTwin's cotrain checkpoint (HF `TianxingChen/RoboTwin2.0`,
`cotrain_ckpt/X_VLA`) the way XPolicyLab `policy/X_VLA/model.py` does: head camera only, the
episode instruction as prompt, `domain_id` 0, wxyz quaternions, raw gripper values, and
proprio from the measured end pose on every query. The tarball lacks the `trust_remote_code`
files; copy them from `2toINF/X-VLA-RoboTwin2`. Set `XVLA_ROBOTWIN_CKPT` to the `ckpt-30000`
directory and use the same benchmark config.
