#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-e <eval_id>] [-o <output_dir>] [--record-video|--no-record-video] [--render gpu|cpu]

Spawn N shards of \`vla-eval run\` against the same SQLite recording, then
call \`vla-eval export\` once after all shards exit. Shards share an eval id
(default: a fresh uuid) so they all write to one
\`<output_dir>/recording-<eval_id>.sqlite\`. A shard that exits non-zero while the
recording shows work left (crash, kill, or exit 3) is relaunched with the same
shard id, which resumes its unfinished items, up to \$VLA_EVAL_SHARD_RESTARTS
(default 10) times; one that never registered in the recording is not.

Options:
  -c <config>          Config YAML file (required)
  -n <num_shards>      Number of shards (default: 50)
  -e <eval_id>         Eval id (default: fresh uuid)
  -o <output_dir>      Override the config's output_dir (passed to each shard
                       AND to export so the SQLite + materialised outputs land
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

# Resolve the output directory once for both recording and export; count entries for shard_state.
CONFIG_INFO="$(python3 - "$CONFIG" <<'PY'
import sys
from vla_eval.cli.config_loader import load_config
config = load_config(sys.argv[1])
print(len(config.get("benchmarks") or []), config.get("output_dir") or "./results")
PY
)"
N_ENTRIES="${CONFIG_INFO%% *}"
OUTPUT_DIR="${OUTPUT_DIR:-${CONFIG_INFO#* }}"
RUN_OPTS=(-c "$CONFIG" --eval-id "$EVAL_ID")
RUN_OPTS+=(--output-dir "$OUTPUT_DIR")
if [[ -n "$RECORD_VIDEO_FLAG" ]]; then
  RUN_OPTS+=("$RECORD_VIDEO_FLAG")
fi
if [[ -n "$RENDER" ]]; then
  RUN_OPTS+=(--render "$RENDER")
fi

# Exit 0: shard $1 has nothing left in the recording; 1: work left; 2: it never registered (startup error).
shard_state() {
  python3 - "$OUTPUT_DIR/recording-$EVAL_ID.sqlite" "$EVAL_ID-" "$1" "$N_ENTRIES" <<'PY'
import os, sqlite3, sys
db, prefix, shard, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
if not os.path.exists(db):
    sys.exit(2)
conn = sqlite3.connect(db, timeout=60)
rows = conn.execute("SELECT eval_id, shard_id, complete FROM eval_shards WHERE substr(eval_id, 1, ?) = ?",
                    (len(prefix), prefix)).fetchall()
mine = {e: c for e, s, c in rows if s == shard}
entries = {e for e, _, _ in rows}

def done(entry):  # every queue item done, whichever shard did it, or this shard finished the entry
    total, finished = conn.execute("SELECT COUNT(*), COALESCE(SUM(done), 0) FROM work_queue WHERE eval_id = ?",
                                   (entry,)).fetchone()
    return 0 < total == finished or mine.get(entry) == 1

sys.exit(2 if not mine else 0 if len(entries) >= n and all(map(done, entries)) else 1)
PY
}

MAX_RESTARTS="${VLA_EVAL_SHARD_RESTARTS:-10}"
run_shard() {
  local attempt=0 rc state
  while true; do
    rc=0
    # </dev/null: with -m, background jobs keep terminal stdin; a shard reading it would stop on SIGTTIN
    vla-eval run "${RUN_OPTS[@]}" --shard-id "$1" --num-shards "$NUM_SHARDS" </dev/null || rc=$?
    if [[ "$rc" -eq 0 ]]; then
      return 0
    fi
    state=0
    shard_state "$1" || state=$?
    if [[ "$state" -eq 0 ]]; then
      echo "Shard $1 exited $rc with its work recorded; treating it as done." >&2
      return 0
    fi
    if [[ "$state" -ne 1 || "$attempt" -ge "$MAX_RESTARTS" ]]; then
      echo "Shard $1 exited $rc; not restarting (attempt $attempt, recording state $state)." >&2
      return "$rc"
    fi
    attempt=$((attempt + 1))
    echo "Shard $1 exited $rc with work left; restarting it ($attempt/$MAX_RESTARTS)..." >&2
    sleep "${VLA_EVAL_RESTART_DELAY:-5}"
  done
}

echo "Launching ${NUM_SHARDS} shards..."
set -m  # each shard leads its own process group, so cleanup() can signal its whole tree
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  run_shard "$i" &
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
pids=()  # all shards reaped; keep the trap from signaling reused PIDs during export

if [[ "$failed" -gt 0 ]]; then
  echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
fi

echo "Materializing per-episode jsonl + aggregate JSON via 'vla-eval export'..."
vla-eval export "$OUTPUT_DIR/recording-$EVAL_ID.sqlite" || {
  echo "WARNING: export failed; the SQLite recording still has the raw data — rerun 'vla-eval export' manually." >&2
  echo "         If it failed with 'Permission denied', the shard containers ran as root; set 'docker.user: host' in the config." >&2
  exit 1
}

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
