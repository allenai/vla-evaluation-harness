# Reproductions

Systematic verification that vla-eval reproduces published VLA model scores across codebases and benchmarks.

## Summary

| Codebase | Model | LIBERO | CALVIN | SE WidowX VM | SE GR VM | SE GR VA | RoboTwin | Other |
|----------|-------|:------:|:------:|:------------:|:--------:|:--------:|:--------:|:-----:|
| [openvla/openvla](https://github.com/openvla/openvla) | [OpenVLA](https://arxiv.org/abs/2406.09246) | ⬜<br>[76.5%](https://huggingface.co/openvla/openvla-7b) | · | · | · | · | · | |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | [π₀.₅](https://arxiv.org/abs/2410.24164) | ✅<br>**97.7%** / [96.9%](https://huggingface.co/collections/physical-intelligence/openpi-6752e26cfc04d5f5013709ef) | · | · | · | · | · | |
| [microsoft/CogACT](https://github.com/microsoft/CogACT) | [CogACT](https://arxiv.org/abs/2411.19650) | · | · | ⬜<br>[51.3%](https://huggingface.co/CogACT/CogACT-Base) | ⬜<br>[74.8%](https://huggingface.co/CogACT/CogACT-Base) | ⬜<br>[61.3%](https://huggingface.co/CogACT/CogACT-Base) | · | |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | ⬜<br>[97.1%](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) (sp: 94.0%) | · | · | · | · | · | |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | [GR00T N1](https://arxiv.org/abs/2503.14734) | 🟡<br>**94.9%** / [97.0%¹](https://huggingface.co/0xAnkitSingh/GR00T-N1.6-LIBERO) | · | 🔧<br>**29.6%** / [57.1%](https://huggingface.co/nvidia/GR00T-N1.6-bridge) | 🟡<br>**59.7%** / [67.7%](https://huggingface.co/nvidia/GR00T-N1.6-fractal) | · | · | |
| [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | [RTC](https://arxiv.org/abs/2506.07339) | · | · | · | · | · | · | Kinetix: ckpt |
| [baaivision/UniVLA](https://github.com/baaivision/UniVLA) | [UniVLA](https://arxiv.org/abs/2506.19850) | ⬜<br>[95.5%](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_LIBERO_VIDEO_BS192_8K) | ⬜<br>[4.63](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_CALVIN_ABCD_VIDEO_BS192_8K) | ⬜<br>[69.8%](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K) | · | · | · | |
| [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | [X-VLA](https://arxiv.org/abs/2510.10274) | ✅<br>**97.4%** / [98.1%](https://huggingface.co/2toINF/X-VLA-Libero) | ✅<br>**4.30** / [4.43](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D) | ✅<br>**94.8%** / [95.8%](https://huggingface.co/2toINF/X-VLA-WidowX) | ⬜<br>[80.4%](https://huggingface.co/2toINF/X-VLA-WidowX) | ⬜<br>[75.7%](https://huggingface.co/2toINF/X-VLA-WidowX) | ⬜<br>[70/39%](https://huggingface.co/2toINF/X-VLA-WidowX) | VLABench: 51.1% |
| [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic) | [DB-CogACT](https://arxiv.org/abs/2510.23511) | ✅<br>**94.7%** / [94.9%](https://huggingface.co/Dexmal/libero-db-cogact) | ✅<br>**4.02** / [4.06](https://huggingface.co/Dexmal/calvin-db-cogact) | 🟡<br>**63.5%** / [69.5%](https://huggingface.co/Dexmal/simpler-db-cogact) | · | · | ⬜<br>[58.5%](https://huggingface.co/Dexmal/robotwin-db-cogact) | [MS2: 58%](https://huggingface.co/Dexmal/maniskill2-db-cogact) |
| | [DB-π₀](https://arxiv.org/abs/2510.23511) | ⬜<br>[93.9%](https://huggingface.co/Dexmal/libero-db-pi0) | · | · | · | · | · | [MS2: 65%](https://huggingface.co/Dexmal/maniskill2-db-pi0) |
| | [DB-OFT](https://arxiv.org/abs/2510.23511) | · | ⬜<br>[3.54](https://huggingface.co/Dexmal/calvin-db-oft) | ⬜<br>[76.4%](https://huggingface.co/Dexmal/simpler-db-oft) | · | · | · | [MS2: 63%](https://huggingface.co/Dexmal/maniskill2-db-oft) |
| | [DB-MemVLA](https://arxiv.org/abs/2510.23511) | ⬜<br>[97.0%](https://huggingface.co/Dexmal/libero-db-memvla) | · | ⬜<br>[84.4%](https://huggingface.co/Dexmal/simpler-db-memvla) | · | · | · | |
| | [DB-GR00TN1](https://arxiv.org/abs/2510.23511) | ⬜<br>94.8%† | · | · | · | · | · | |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | Qwen3-FAST | ⬜<br>95.4%† | · | ⬜<br>31.6%† | · | · | · | |
| | Qwen3-OFT | ⬜<br>[96.6%](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1) | · | ⬜<br>[42.7%](https://huggingface.co/StarVLA/Qwen3VL-OFT-Bridge-RT-1) | · | · | ⬜<br>[50.4%](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All) | |
| | Qwen3-PI | ⬜<br>[95.7%](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1) | · | ⬜<br>60.9%† | · | · | · | |
| | Qwen3-GR00T | ⬜<br>96.5%† | ⬜<br>3.76† | ✅<br>**66.7%** / [65.3%](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) | · | · | · | |

<!-- StarVLA Qwen2.5 variants (also available):
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | Q2.5-FAST | ⬜<br>[95.2%](https://huggingface.co/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1) | · | ⬜<br>[58.6%](https://huggingface.co/StarVLA/Qwen-FAST-Bridge-RT-1) | · | · | · | |
| | Q2.5-OFT | ⬜<br>[96.1%](https://huggingface.co/StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1) | · | ⬜<br>[41.8%](https://huggingface.co/StarVLA/Qwen-OFT-Bridge-RT-1) | · | · | · | |
| | Q2.5-PI | ⬜<br>95.4% | · | ⬜<br>[62.5%](https://huggingface.co/StarVLA/Qwen-PI-Bridge-RT-1) | · | · | · | |
| | Q2.5-GR00T | ⬜<br>[95.4%](https://huggingface.co/StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1) | ⬜<br>[3.79](https://huggingface.co/Simplicissimus-S/StarVLA-QwenGR00T_Qwen2.5-VL-3B-Instruct-Action_calvin_D_D) | ⬜<br>[63.6%](https://huggingface.co/StarVLA/Qwen-GR00T-Bridge-RT-1) | · | · | · | |
-->

SE = SimplerEnv. SE GR = Google Robot VM. MS2 = ManiSkill2.

**Status:** ✅ Reproduced · 🟡 Approximate (≤5pp) · 🔧 WIP · ⬜ Not attempted · `·` No score / no checkpoint

**Cell format:** status / [reported%](HF checkpoint link). Bold = our reproduced score.

¹ Community checkpoint (not official NVIDIA). † Checkpoint not publicly available on HuggingFace. Per-model details in linked docs below.

Per-codebase details: [openvla](openvla.md) · [openpi](openpi.md) · [cogact](cogact.md) · [oft](oft.md) · [groot](groot.md) · [rtc](rtc.md) · [xvla](xvla.md) · [dexbotic](dexbotic.md) · [starvla](starvla.md)

## Benchmarks with no model coverage yet

Integrated in vla-eval: RLBench, RoboCasa, Mikasa, RoboMMe, RoboCerebra, LIBERO-90, LIBERO-Pro.

## Verdict Criteria

Binomial 95% CI for 500 episodes/suite (±1.9pp at p=0.95):
✅ within CI · 🟡 outside CI, ≤5pp · 🔧 in progress or >5pp with known cause · ⬜ not yet attempted.

## Files

| File | Contents |
|------|----------|
| [common-pitfalls.md](common-pitfalls.md) | Reproduction pitfalls taxonomy |
| [running-guide.md](running-guide.md) | How to run evaluations + supply/demand data |
| [`data/`](data/) | Raw result JSONs per codebase×benchmark |
