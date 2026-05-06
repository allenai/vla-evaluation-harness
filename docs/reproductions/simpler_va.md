# SimplerEnv Google Robot -- Variant Aggregation (VA) Reproduction

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

## Results

### X-VLA (X-VLA-Google-Robot) -- VM + VA

| | |
|---|---|
| **Checkpoint** | `2toINF/X-VLA-Google-Robot` (HuggingFace) |
| **Server config** | [`configs/model_servers/xvla/simpler_google_robot.yaml`](../../configs/model_servers/xvla/simpler_google_robot.yaml) |
| **Docker image** | `simpler-xvla` (absolute EE controller required) |

**VM (pick_coke_can, 24 episodes):**

| Task | Reproduced | Reported | Verdict |
|------|:----------:|:--------:|:-------:|
| pick_coke_can | **100%** | 98.3% | Reproduced |

**VA (move_near base variant, 60 episodes):**

| Task | Reproduced | Reported | Verdict |
|------|:----------:|:--------:|:-------:|
| move_near (base) | **85.0%** | 84.0% | Reproduced |

Key integration fixes required for X-VLA Google Robot:
- Correct checkpoint (`X-VLA-Google-Robot`, not `X-VLA-WidowX`)
- Google Robot `base_pose` controller (no `use_target`) + prepackaged
  defaults changed to `base_pose` in `Dockerfile.simpler_xvla`
- Position accumulation + action stride (::2, cap 10) in xvla.py
- Euler rotation via `euler_offset` config
- Gripper threshold 0.25, zero proprio init, `max_episode_steps=160`

### OpenVLA (openvla-7b) -- Move Near VA

| | |
|---|---|
| **Checkpoint** | `openvla/openvla-7b` (HuggingFace) |
| **Server config** | [`configs/model_servers/openvla/simpler_google_robot.yaml`](../../configs/model_servers/openvla/simpler_google_robot.yaml) |
| **Docker image** | `simpler` (standard) |

600 episodes (10 visual variants x 60 episodes each). `seed=0`, `success_mode=truncation`.

| Visual variant | Reproduced |
|----------------|:----------:|
| base | 46.7% |
| no_distractor | 56.7% |
| bg_1 | 45.0% |
| bg_2 | 43.3% |
| light_dark | 61.7% |
| light_bright | 48.3% |
| table_tex_1 | 63.3% |
| table_tex_2 | 61.7% |
| camera_1 | 33.3% |
| camera_2 | 40.0% |
| **Average** | **50.0%** |

Reported move_near VA for OpenVLA: **67.7%** (from CogACT paper comparison table).
Delta: -17.7pp. Possible causes:
- SimplerEnv version difference (pinned Docker image vs original evaluation)
- Reported number is from a third-party comparison table, not OpenVLA's own paper

### Remaining tasks

Pick Coke Can, Open/Close Drawer, and Place in Drawer VA evaluations
have not yet been run with full episodes across all variants.  The configs
and infrastructure are ready; these require additional compute time.

### Reference numbers (SimplerEnv paper, Table 2)

| Task | RT-1 (tf) VA | Octo-base VA |
|------|:------------:|:------------:|
| Pick Coke Can | 56.7% | 18.5% |
| Move Near | 46.9% | 30.6% |
| Open/Close Drawer | 30.8% | 38.9% |
| **3-task avg** | **44.8%** | **29.3%** |
