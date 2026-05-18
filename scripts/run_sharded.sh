#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-e <eval_id>] [-o <output_dir>]

Spawn N shards of \`vla-eval run\` against the same SQLite recording, then
call \`vla-eval merge\` once after all shards exit. Shards share an eval id
(default: a fresh uuid) so they all write to one
\`<output_dir>/recording-<eval_id>.sqlite\`.

Options:
  -c <config>          Config YAML file (required)
  -n <num_shards>      Number of shards (default: 50)
  -e <eval_id>         Eval id (default: fresh uuid)
  -o <output_dir>      Override the config's output_dir (passed to each shard
                       AND to merge so the SQLite + materialised outputs land
                       in the same place)
  -h                   Show this help
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=50
EVAL_ID=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CONFIG="$2"; shift 2 ;;
    -n) NUM_SHARDS="$2"; shift 2 ;;
    -e) EVAL_ID="$2"; shift 2 ;;
    -o) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage 1 ;;
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

if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
fi

cleanup() {
  echo "Cleaning up background processes..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT

echo "Config:     $CONFIG"
echo "Shards:     $NUM_SHARDS"
echo "Eval ID:    $EVAL_ID"
if [[ -n "$OUTPUT_DIR" ]]; then
  echo "Output dir: $OUTPUT_DIR"
fi
echo ""

# Build the shared CLI args once so the run and merge invocations stay in sync.
RUN_OPTS=(-c "$CONFIG" --eval-id "$EVAL_ID")
MERGE_OPTS=(-c "$CONFIG" --eval-id "$EVAL_ID")
if [[ -n "$OUTPUT_DIR" ]]; then
  RUN_OPTS+=(--output-dir "$OUTPUT_DIR")
  MERGE_OPTS+=(--output-dir "$OUTPUT_DIR")
fi

echo "Launching ${NUM_SHARDS} shards..."
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  vla-eval run "${RUN_OPTS[@]}" --shard-id "$i" --num-shards "$NUM_SHARDS" &
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
fi

echo "Materializing per-episode jsonl + aggregate JSON via 'vla-eval merge'..."
vla-eval merge "${MERGE_OPTS[@]}" || \
  echo "WARNING: merge failed; the SQLite recording still has the raw data — rerun 'vla-eval merge' manually." >&2

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
