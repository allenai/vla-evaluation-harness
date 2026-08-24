#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-e <eval_id>] [-o <output_dir>] [--record-video|--no-record-video] [--render gpu|cpu]

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
  --record-video       Enable per-episode mp4 recording for all shard runs
  --no-record-video    Disable per-episode mp4 recording for all shard runs
  --render <gpu|cpu>   Render backend for every shard (default: the config's).
                       'cpu' software-renders and attaches no GPU to the shards.
  -h                   Show this help
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=50
EVAL_ID=""
OUTPUT_DIR=""
RECORD_VIDEO_FLAG=""
RENDER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) CONFIG="$2"; shift 2 ;;
    -n) NUM_SHARDS="$2"; shift 2 ;;
    -e) EVAL_ID="$2"; shift 2 ;;
    -o) OUTPUT_DIR="$2"; shift 2 ;;
    --render) RENDER="$2"; shift 2 ;;
    --record-video|--no-record-video) RECORD_VIDEO_FLAG="$1"; shift ;;  # last flag wins, matching vla-eval run
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

if ! command -v vla-eval >/dev/null 2>&1; then
  echo "Error: 'vla-eval' is not on PATH. Activate the venv first" >&2
  echo "       (e.g. '. .venv/bin/activate') or invoke this script via 'uv run'." >&2
  exit 1
fi

if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"
fi

# kill -- -$$ only works when this script leads its own process group, which is
# true under an interactive shell but not under CI, a container, or another script.
pids=()
cleanup() {
  echo "Cleaning up background processes..."
  local pid
  for pid in "${pids[@]}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
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
if [[ -n "$RECORD_VIDEO_FLAG" ]]; then
  RUN_OPTS+=("$RECORD_VIDEO_FLAG")
fi
if [[ -n "$RENDER" ]]; then
  RUN_OPTS+=(--render "$RENDER")
fi

echo "Launching ${NUM_SHARDS} shards..."
set -m  # each shard leads its own process group, so cleanup() can signal its whole tree
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  # </dev/null: with -m, background jobs keep terminal stdin; a shard reading it would stop on SIGTTIN
  vla-eval run "${RUN_OPTS[@]}" --shard-id "$i" --num-shards "$NUM_SHARDS" </dev/null &
  pids+=($!)
done
set +m

echo "Waiting for all shards to finish..."
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
pids=()  # all shards reaped; keep the trap from signaling reused PIDs during merge

if [[ "$failed" -gt 0 ]]; then
  echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
fi

echo "Materializing per-episode jsonl + aggregate JSON via 'vla-eval merge'..."
vla-eval merge "${MERGE_OPTS[@]}" || \
  echo "WARNING: merge failed; the SQLite recording still has the raw data — rerun 'vla-eval merge' manually." >&2

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
