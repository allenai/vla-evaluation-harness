# SimplerEnv Google Robot -- Variant Aggregation (VA) Reference

VA evaluates generalization across visual domain shifts: different backgrounds,
lighting, distractors, camera angles, table textures, and cabinet models.
The variant matrix is derived from the official SimplerEnv evaluation scripts
(`scripts/*_variant_agg.sh`).

Reference: [SimplerEnv paper (CoRL 2024)](https://arxiv.org/abs/2405.05941) |
[GitHub](https://github.com/simpler-env/SimplerEnv)

## Configs

| Task group | Config | Entries | Episodes/entry | Source script |
|------------|--------|--------:|---------------:|---------------|
| Pick Coke Can | `simpler_google_robot_pick_coke_can_va.yaml` | 33 | 25 | `rt1_pick_coke_can_variant_agg.sh` |
| Open/Close Drawer | `simpler_google_robot_drawer_va.yaml` | 42 | 9 | `rt1_drawer_variant_agg.sh` |
| Move Near | `simpler_google_robot_move_near_va.yaml` | 10 | 60 | `rt1_move_near_variant_agg.sh` |
| Place in Drawer | `simpler_google_robot_put_in_drawer_va.yaml` | 7 | 27 | `rt1_put_in_drawer_variant_agg.sh` |

Each entry is one visual variant.  The VA score for a task group is the
mean success rate across all entries in that file.  The overall VA score
is the mean of pick_coke_can, drawer, and move_near (3-task average);
place-in-drawer is reported separately as an optional sub-score.

## Docker images

Most models use `simpler:latest`.  X-VLA requires `simpler-xvla:latest`
which adds an absolute EE controller (`base_pose`, `use_delta=False`)
for both WidowX and Google Robot.

## Running

```bash
# Start model server
vla-eval serve -c configs/model_servers/<model>/simpler_google_robot.yaml -v

# Run VA evaluation (example: move_near, 8 shards)
for i in $(seq 0 7); do
  vla-eval run -c configs/simpler_google_robot_move_near_va.yaml \
    --shard-id $i --num-shards 8 \
    --output-dir ./results/va_move_near -y &
done
```

## Reproduction results

Per-model results live in each model's reproduction doc:
- **X-VLA**: [xvla.md](xvla.md) -- VM 100%, VA 85.0% (move_near base)
- **OpenVLA**: 50.0% VA avg across 10 variants (600 eps, -17.7pp vs reported)

## Reference numbers (SimplerEnv paper, Table 2)

| Task | RT-1 (tf) VA | Octo-base VA |
|------|:------------:|:------------:|
| Pick Coke Can | 56.7% | 18.5% |
| Move Near | 46.9% | 30.6% |
| Open/Close Drawer | 30.8% | 38.9% |
| **3-task avg** | **44.8%** | **29.3%** |
