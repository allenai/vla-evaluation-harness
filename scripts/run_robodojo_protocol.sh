#!/usr/bin/env bash
# Run the RoboDojo protocol one task per `vla-eval run` invocation.
#
# Isaac Lab's SimulationContext is process-global and RoboDojo env teardown
# hangs, so a single process (container) can evaluate exactly one task —
# upstream's eval_policy.sh has the same shape. This driver reads the task
# table from a protocol config (default: configs/benchmarks/robodojo/eval.yaml),
# materialises a single-task config per task, and runs them sequentially.
#
# Usage:
#   scripts/run_robodojo_protocol.sh [--config <eval.yaml>] [--filter <regex>] \
#       [--dry-run] [-- <extra vla-eval run args, e.g. --gpus 0 --record-video>]
#
# Parallelism: run several instances with disjoint --filter values (one lane
# per GPU). Each task writes its own aggregate under <output_base>/<task>/.
#
# Isaac Sim occasionally crashes a task's process mid-run (breakpad,
# unrecoverable). That task's aggregate is then incomplete; re-run it with the
# same --filter to redo it from scratch.
set -euo pipefail

CONFIG="configs/benchmarks/robodojo/eval.yaml"
FILTER=""
DRY_RUN=false
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)  CONFIG="$2"; shift 2 ;;
    --filter)  FILTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --)        shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TMPDIR_CFG="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_CFG}"' EXIT

# Emit "<task>\t<episodes>\t<single-task-config-path>" per protocol row.
if ! python3 - "$CONFIG" "$TMPDIR_CFG" > "${TMPDIR_CFG}/rows.tsv" <<'EOF'
import copy, sys, yaml

config_path, out_dir = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
for entry in cfg.get("benchmarks", []):
    for task in entry.get("params", {}).get("tasks", []):
        single = copy.deepcopy(cfg)
        one = copy.deepcopy(entry)
        one["params"]["tasks"] = [task]
        single["benchmarks"] = [one]
        path = f"{out_dir}/{task}.yaml"
        yaml.safe_dump(single, open(path, "w", encoding="utf-8"), sort_keys=False)
        print(f"{task}\t{entry.get('episodes_per_task', 50)}\t{path}")
EOF
then
  echo "ERROR: failed to expand task table from ${CONFIG}"; exit 1
fi
mapfile -t ROWS < "${TMPDIR_CFG}/rows.tsv"
if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "ERROR: no tasks found in ${CONFIG}"; exit 1
fi

OUTPUT_BASE="${ROBODOJO_OUTPUT_BASE:-./results/robodojo}"

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r task episodes path <<< "$row"
  if [[ -n "$FILTER" && ! "$task" =~ $FILTER ]]; then
    continue
  fi
  echo ">>> $task (${episodes} episodes) -> ${OUTPUT_BASE}/${task}"
  if $DRY_RUN; then
    continue
  fi
  # Per-task output dir: the aggregate filename derives from the benchmark
  # class, so tasks sharing a directory would overwrite each other's aggregate.
  vla-eval run -c "$path" --yes --output-dir "${OUTPUT_BASE}/${task}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
done
