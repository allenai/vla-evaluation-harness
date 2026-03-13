#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-o <output>]

Run a benchmark in parallel shards and merge results.

Options:
  -c <config>       Config YAML file (required)
  -n <num_shards>   Number of shards (default: 50)
  -o <output>       Output file for merged results (default: results/<config_name>.json)
  -h                Show this help
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=50
OUTPUT=""

while getopts "c:n:o:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    n) NUM_SHARDS="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: -c <config> is required." >&2
  usage 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

# Derive output name from config filename if not specified
if [[ -z "$OUTPUT" ]]; then
  config_name="$(basename "$CONFIG" .yaml)"
  config_name="$(basename "$config_name" .yml)"
  OUTPUT="results/${config_name}.json"
fi

cleanup() {
  echo "Cleaning up background processes..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT

echo "Config:     $CONFIG"
echo "Shards:     $NUM_SHARDS"
echo "Output:     $OUTPUT"
echo ""

# Check for existing shard results
existing=$(CONFIG="$CONFIG" NUM_SHARDS="$NUM_SHARDS" python3 -c "
import os, yaml, re
from pathlib import Path
with open(os.environ['CONFIG']) as f:
    cfg = yaml.safe_load(f)
num_shards = os.environ['NUM_SHARDS']
output_dir = Path(cfg.get('output_dir', './results'))
found = []
seen = set()
for b in cfg.get('benchmarks', []):
    name = b.get('name') or b['benchmark'].rsplit(':', 1)[-1]
    sub = b.get('subname')
    if sub:
        name = f'{name}_{sub}'
    safe = re.sub(r'[^\w\-.]', '_', name)
    if safe in seen:
        continue
    seen.add(safe)
    found.extend(output_dir.glob(f'{safe}_shard*of{num_shards}.json'))
if found:
    print(f'{len(found)} existing shard file(s) found, e.g.: {found[0]}')
")
if [[ -n "$existing" ]]; then
  echo "Error: $existing" >&2
  echo "Remove existing results or use a different output_dir." >&2
  exit 1
fi

echo "Launching ${NUM_SHARDS} shards..."

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  vla-eval run -c "$CONFIG" --shard-id "$i" --num-shards "$NUM_SHARDS" &
  pids+=($!)
done

echo "Waiting for all shards to finish..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
  exit 1
fi

echo "Merging results..."
vla-eval merge -c "$CONFIG" -o "$OUTPUT"

echo "Done. Results saved to $OUTPUT"
