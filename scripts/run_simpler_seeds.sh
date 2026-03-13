#!/usr/bin/env bash
# Run SimplerEnv evaluation across multiple seeds and merge results.
#
# Usage:
#   scripts/run_simpler_seeds.sh                          # seeds 0,2,4, 16 shards
#   scripts/run_simpler_seeds.sh -s "0 2 4" -n 8          # custom seeds/shards
#   scripts/run_simpler_seeds.sh -c configs/custom.yaml   # custom config
set -euo pipefail

CONFIG="configs/simpler_all_tasks.yaml"
SEEDS="0 2 4"
NUM_SHARDS=16
OUTPUT_BASE="results/simpler_seeds"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)  CONFIG="$2"; shift 2 ;;
    -s|--seeds)   SEEDS="$2"; shift 2 ;;
    -n|--shards)  NUM_SHARDS="$2"; shift 2 ;;
    -o|--output)  OUTPUT_BASE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") [-c config] [-s seeds] [-n shards] [-o output_base]"
      echo "  -c  Config YAML (default: configs/simpler_all_tasks.yaml)"
      echo "  -s  Space-separated seeds (default: \"0 2 4\")"
      echo "  -n  Number of shards (default: 16)"
      echo "  -o  Output base directory (default: results/simpler_seeds)"
      exit 0 ;;
    *)  echo "Unknown flag: $1"; exit 1 ;;
  esac
done

for seed in $SEEDS; do
  echo ""
  echo "========================================="
  echo "Seed $seed"
  echo "========================================="

  seed_dir="${OUTPUT_BASE}/seed${seed}"
  tmp_config="$(mktemp "/tmp/simpler_seed${seed}_XXXX.yaml")"

  python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
cfg['output_dir'] = '${seed_dir}'
for b in cfg.get('benchmarks', []):
    b.setdefault('params', {})['seed'] = ${seed}
with open('${tmp_config}', 'w') as f:
    yaml.dump(cfg, f)
"

  ./scripts/run_sharded.sh -c "$tmp_config" -n "$NUM_SHARDS"
  rm -f "$tmp_config"
done

echo ""
echo "========================================="
echo "All seeds complete. Results in ${OUTPUT_BASE}/"
echo "========================================="
