---
smoke_config: null  # needs the ~7 GiB released checkpoint + GPU
---

# π₀.₅ (RoboDojo leaderboard checkpoint)

Serves the officially released RoboDojo π₀.₅ simulation checkpoints
([leaderboard](https://robodojo-benchmark.com/leaderboard) rank 4;
paper [arXiv:2607.04434](https://arxiv.org/abs/2607.04434) Table 1:
average score 11.41 / success rate 6.91%, mean over 3 training seeds).

## Checkpoint

All three training seeds are released in the `RoboDojo-Benchmark/RoboDojo`
Hugging Face dataset:

```bash
# sparse-pull one seed (~7 GiB)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse \
  https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo robodojo-ckpt
cd robodojo-ckpt
git sparse-checkout set ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0
git lfs pull
export ROBODOJO_PI05_CKPT=$(pwd)/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999
```

For seeds 1/2, also set `config_name: pi05_base_aloha_full_sim_arx-x5_seed_<n>`
(the fork's per-seed train configs are identical except the seed, but keeping
them matched preserves the exact upstream deploy path).

## Run

```bash
vla-eval serve -c configs/model_servers/robodojo_pi05/pi05.yaml
# One smoke task:
vla-eval run  -c configs/benchmarks/robodojo/smoke_test.yaml
# Full protocol (one process per task — Isaac's SimulationContext is process-global,
# so a single `vla-eval run` on eval.yaml would only complete the first task):
scripts/run_robodojo_protocol.sh
```

Input contract mirrors XPolicyLab `policy/Pi_05/model.py`: three CHW uint8
cameras (`cam_high` ← RoboDojo `cam_head`, `cam_left_wrist`, `cam_right_wrist`),
14-D packed joint state, instruction prompt; the full action chunk executes
open-loop before the next inference, as in upstream `eval_one_episode`.
