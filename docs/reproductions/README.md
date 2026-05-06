# Reproductions

Systematic verification that vla-eval reproduces published VLA model scores across codebases and benchmarks.

## Core Benchmarks

| Codebase | Model | LIBERO | CALVIN | SE WidowX VM | SE GR VM | SE GR VA |
|----------|-------|:------:|:------:|:------------:|:--------:|:--------:|
| [openvla/openvla](https://github.com/openvla/openvla) | [OpenVLA](https://arxiv.org/abs/2406.09246) | ✅<br>**76.2%** / [76.5%](https://huggingface.co/openvla/openvla-7b) | · | · | · | · |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | [π₀.₅](https://arxiv.org/abs/2410.24164) | ✅<br>**97.7%** / [96.9%](https://huggingface.co/collections/physical-intelligence/openpi-6752e26cfc04d5f5013709ef) | · | · | · | · |
| [microsoft/CogACT](https://github.com/microsoft/CogACT) | [CogACT](https://arxiv.org/abs/2411.19650) | · | · | ⬜<br>[51.3%](https://huggingface.co/CogACT/CogACT-Base) | ⬜<br>[74.8%](https://huggingface.co/CogACT/CogACT-Base) | ⬜<br>[61.3%](https://huggingface.co/CogACT/CogACT-Base) |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | ✅<br>**96.7%** / [97.1%](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) | · | · | · | · |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | [GR00T N1](https://arxiv.org/abs/2503.14734) | 🟡<br>**94.9%** / [97.0%¹](https://huggingface.co/0xAnkitSingh/GR00T-N1.6-LIBERO) | · | 🔧<br>**29.6%** / [57.1%](https://huggingface.co/nvidia/GR00T-N1.6-bridge) | 🟡<br>**59.7%** / [67.7%](https://huggingface.co/nvidia/GR00T-N1.6-fractal) | · |
| [baaivision/UniVLA](https://github.com/baaivision/UniVLA) | [UniVLA](https://arxiv.org/abs/2506.19850) | ⬜<br>[95.5%](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_LIBERO_VIDEO_BS192_8K) | ⬜<br>[4.63](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_CALVIN_ABCD_VIDEO_BS192_8K) | ⬜<br>[69.8%](https://huggingface.co/Yuqi1997/UniVLA/tree/main/UNIVLA_SIMPLER_BRIDGE_VIDEO_BS128_20K) | · | · |
| [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | [X-VLA](https://arxiv.org/abs/2510.10274) | ✅<br>**97.4%** / [98.1%](https://huggingface.co/2toINF/X-VLA-Libero) | ✅<br>**4.30** / [4.43](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D) | ✅<br>**94.8%** / [95.8%](https://huggingface.co/2toINF/X-VLA-WidowX) | ✅<br>**100%** / [98.3%](https://huggingface.co/2toINF/X-VLA-Google-Robot) | 🟡<br>**80.8%** / [84.0%⁴](https://huggingface.co/2toINF/X-VLA-Google-Robot) |
| [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic) | [DB-CogACT](https://arxiv.org/abs/2510.23511) | ✅<br>**94.7%** / [94.9%](https://huggingface.co/Dexmal/libero-db-cogact) | ✅<br>**4.02** / [4.06](https://huggingface.co/Dexmal/calvin-db-cogact) | 🟡<br>**63.5%** / [69.5%](https://huggingface.co/Dexmal/simpler-db-cogact) | · | · |
| | [DB-π₀](https://arxiv.org/abs/2510.23511) | ⬜<br>[93.9%](https://huggingface.co/Dexmal/libero-db-pi0) | · | · | · | · |
| | [DB-OFT](https://arxiv.org/abs/2510.23511) | · | ⬜<br>[3.54](https://huggingface.co/Dexmal/calvin-db-oft) | ⬜<br>[76.4%](https://huggingface.co/Dexmal/simpler-db-oft) | · | · |
| | [DB-MemVLA](https://arxiv.org/abs/2510.23511) | ⬜<br>[97.0%](https://huggingface.co/Dexmal/libero-db-memvla) | · | ⬜<br>[84.4%](https://huggingface.co/Dexmal/simpler-db-memvla) | · | · |
| | [DB-GR00TN1](https://arxiv.org/abs/2510.23511) | ⬜<br>94.8%† | · | · | · | · |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | Q2.5-FAST | ⬜<br>[95.2%](https://huggingface.co/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1) | · | ✅<br>**64.6%** / [58.6%](https://huggingface.co/StarVLA/Qwen-FAST-Bridge-RT-1) | · | · |
| | Qwen3-FAST | ⬜<br>95.4%† | · | ⬜<br>31.6%† | · | · |
| | Qwen3-OFT | ✅<br>**96.8%** / [97.8%²](https://huggingface.co/StarVLA/Qwen3-VL-OFT-LIBERO-4in1) | · | ⬜<br>[42.7%](https://huggingface.co/StarVLA/Qwen3VL-OFT-Bridge-RT-1) | · | · |
| | Qwen3-PI | ⬜<br>[95.7%](https://huggingface.co/StarVLA/Qwen3-VL-PI-LIBERO-4in1) | · | ⬜<br>60.9%† | · | · |
| | Qwen3-GR00T | ⬜<br>96.5%† | ⬜<br>3.76† | ✅<br>**66.7%** / [65.3%](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) | · | · |
| [DravenALG/VLANeXt](https://github.com/DravenALG/VLANeXt) | [VLANeXt](https://arxiv.org/abs/2602.18532) | ✅<br>[**96.9%**](https://github.com/allenai/vla-evaluation-harness/pull/34) / [97.4%](https://huggingface.co/DravenALG/VLANeXt)³ | · | · | · | · |

SE = SimplerEnv. SE GR = Google Robot VM.

¹ Community checkpoint (not official NVIDIA). ² Spatial suite only (reported 97.8%); 4-suite avg is 96.6%. ³ 4-suite LIBERO average (Spatial 98.2, Object 98.8, Goal 97.0, Long 93.6). ⁴ Self-reported best rollout. Move Near 10 variants × 60 eps. † Checkpoint not publicly available on HuggingFace.

## Other Benchmarks

| Codebase | Model | RoboTwin | RoboMME | ManiSkill2 | Kinetix | VLABench | MolmoSpaces-Bench |
|----------|-------|:--------:|:-------:|:----------:|:-------:|:--------:|:-----------------:|
| [2toinf/X-VLA](https://github.com/2toinf/X-VLA) | [X-VLA](https://arxiv.org/abs/2510.10274) | ⬜<br>[70/39%](https://huggingface.co/2toINF/X-VLA-WidowX) | · | · | · | ⬜<br>51.1% | · |
| [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic) | [DB-CogACT](https://arxiv.org/abs/2510.23511) | ⬜<br>[58.5%](https://huggingface.co/Dexmal/robotwin-db-cogact) | · | ⬜<br>[58%](https://huggingface.co/Dexmal/maniskill2-db-cogact) | · | · | · |
| | [DB-π₀](https://arxiv.org/abs/2510.23511) | · | · | ⬜<br>[65%](https://huggingface.co/Dexmal/maniskill2-db-pi0) | · | · | · |
| | [DB-OFT](https://arxiv.org/abs/2510.23511) | · | · | ⬜<br>[63%](https://huggingface.co/Dexmal/maniskill2-db-oft) | · | · | · |
| [starVLA/starVLA](https://github.com/starVLA/starVLA) | Qwen3-OFT | ⬜<br>[50.4%](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All) | · | · | · | · | · |
| [Physical-Intelligence/rtc](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | [RTC](https://arxiv.org/abs/2506.07339) | · | · | · | ⬜ ckpt | · | · |
| [RoboMME/robomme_policy_learning](https://github.com/RoboMME/robomme_policy_learning) | [MME-VLA π₀.5](https://arxiv.org/abs/2603.04639) | · | ✅<br>**25.5%** / [22.7%](https://huggingface.co/Yinpei/pi05_baseline)¹ | · | · | · | · |
| | [MME-VLA FrameSamp](https://arxiv.org/abs/2603.04639) | · | ⬜<br>[44.5%](https://huggingface.co/Yinpei/mme_vla_suite) | · | · | · | · |
| [allenai/MolmoBot](https://github.com/allenai/MolmoBot) | [MolmoBot (F=2)](https://arxiv.org/abs/2603.16861) | · | · | · | · | · | ✅<br>**57.0%** / [57.7%](https://huggingface.co/allenai/MolmoBot-DROID)² |

¹ Counting suite only (4/16 tasks). Full 4-suite evaluation pending.
² Pick-and-Place only on `procthor-objaverse/FrankaPickandPlaceHardBench` (200 ep). Other task types (pick, open, close, door, navigation) and scene datasets (ithor, procthor-10k, holodeck) not yet reproduced.

**Cell format:** status / [reported%](HF checkpoint link). Bold = our reproduced score. Per-codebase reproduction details live in the per-codebase docs linked below; rows without a per-codebase doc (currently VLANeXt) hyperlink the bolded score to the landing PR instead.

**Status:** ✅ within 95% CI · 🟡 outside CI but ≤5pp · 🔧 in progress or >5pp with known cause · ⬜ not attempted · `·` no score / no checkpoint. CI is binomial at p=0.95 (±1.9pp for 500 episodes).

## Benchmarks with No Model Coverage Yet

Integrated in vla-eval: RLBench, RoboCasa, Mikasa, RoboCerebra, LIBERO-90, LIBERO-Pro, BEHAVIOR-1K ([details](behavior1k.md) — needs an R1Pro-compatible model server).

## Per-Codebase Details

- [openvla/openvla](https://github.com/openvla/openvla): [openvla.md](openvla.md)
- [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi): [openpi.md](openpi.md)
- [microsoft/CogACT](https://github.com/microsoft/CogACT): [cogact.md](cogact.md)
- [moojink/openvla-oft](https://github.com/moojink/openvla-oft): [oft.md](oft.md)
- [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T): [groot.md](groot.md)
- [Physical-Intelligence/rtc](https://github.com/Physical-Intelligence/real-time-chunking-kinetix): [rtc.md](rtc.md)
- [2toinf/X-VLA](https://github.com/2toinf/X-VLA): [xvla.md](xvla.md)
- [Dexmal/dexbotic](https://github.com/Dexmal/dexbotic): [dexbotic.md](dexbotic.md)
- [starVLA/starVLA](https://github.com/starVLA/starVLA): [starvla.md](starvla.md)
- [RoboMME/robomme\_policy\_learning](https://github.com/RoboMME/robomme_policy_learning): [robomme.md](robomme.md)
- [allenai/MolmoBot](https://github.com/allenai/MolmoBot): [molmobot.md](molmobot.md)

## Files

| File | Contents |
|------|----------|
| [common-pitfalls.md](common-pitfalls.md) | Reproduction pitfalls taxonomy |
| [running-guide.md](running-guide.md) | How to run evaluations + supply/demand data |
| [`data/`](data/) | Raw result JSONs per codebase×benchmark |
