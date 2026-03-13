#!/bin/bash
# Pace=1.0 experiments: main table (sim2live/ci/laas) + inference delay sweep
# All conditions use pace=1.0, wait_first_action=true, 50 eps/task, 12 tasks
set -euo pipefail

HARNESS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HARNESS"

RESULTS_BASE="./results/pace1_experiments"
SLURM_ARGS="--partition=h100 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00"
EP=50
BENCH_CONFIG="experiments/sim2live/kinetix_realtime_wait.yaml"
VOLUME_MOUNT="${HARNESS}/src/vla_eval:/opt/conda/envs/kinetix/lib/python3.11/site-packages/vla_eval"
JOB_ID=""
NODE=""

cleanup() {
    if [[ -n "$JOB_ID" ]]; then
        echo "Cancelling slurm job $JOB_ID..."
        scancel "$JOB_ID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

launch_server() {
    local config="$1"
    local job_name="rtc-${2:-serve}"

    echo "=== Launching model server: $config ==="
    JOB_ID=$(sbatch --parsable $SLURM_ARGS --job-name="$job_name" \
        --wrap="cd $HARNESS && uv run vla-eval serve --config $config")
    echo "Submitted job $JOB_ID"

    echo "Waiting for job to start..."
    for i in $(seq 1 120); do
        state=$(squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "GONE")
        if [[ "$state" == "RUNNING" ]]; then break; fi
        if [[ "$state" == "GONE" || "$state" == "FAILED" ]]; then
            echo "ERROR: Job $JOB_ID failed (state=$state)"
            exit 1
        fi
        sleep 5
    done

    NODE=$(squeue -j "$JOB_ID" -h -o "%N")
    echo "Running on node: $NODE"

    echo "Waiting for server on $NODE:8000..."
    for i in $(seq 1 120); do
        if nc -z "$NODE" 8000 2>/dev/null; then
            echo "Port open, waiting for WS handshake..."
            for j in $(seq 1 60); do
                if python3 -c "
import asyncio, websockets
async def check():
    async with websockets.connect('ws://$NODE:8000') as ws:
        pass
asyncio.run(check())
" 2>/dev/null; then
                    echo "Server ready!"
                    return 0
                fi
                sleep 5
            done
            echo "WARNING: Port open but WS handshake failing, trying anyway..."
            return 0
        fi
        sleep 5
    done
    echo "ERROR: Server did not start in 10 min"
    exit 1
}

stop_server() {
    if [[ -n "$JOB_ID" ]]; then
        echo "Stopping model server (job $JOB_ID)..."
        scancel "$JOB_ID" 2>/dev/null || true
        sleep 5
        JOB_ID=""
    fi
}

run_benchmark() {
    local label="$1"
    local bench_config="$2"
    local out_dir="${RESULTS_BASE}/${label}"
    mkdir -p "$out_dir"

    local tmp_config
    tmp_config=$(mktemp /tmp/vla-eval-XXXXX.yaml)

    # Inject server URL, output dir, episode count, and volume mount
    sed -e "s|url:.*|url: \"ws://${NODE}:8000\"|" \
        -e "s|output_dir:.*|output_dir: \"${out_dir}\"|" \
        -e "s|episodes_per_task:.*|episodes_per_task: ${EP}|" \
        "$bench_config" > "$tmp_config"

    # Add volume mount after docker image line
    sed -i "/image:.*/a\\  volumes:\\n    - \"${VOLUME_MOUNT}\"" "$tmp_config"

    echo "=== Running benchmark: $label (${EP} ep/task) ==="
    cat "$tmp_config"
    echo "---"

    uv run vla-eval run --config "$tmp_config" -y
    rm -f "$tmp_config"
    echo "=== Done: $label ==="
}

echo "============================================"
echo " Pace=1.0 Experiment Suite"
echo " 12 tasks × ${EP} episodes"
echo "============================================"
echo ""

# ==========================================
# Experiment 1: Main table (sim2live/ci/laas)
# ==========================================

echo ">>> EXPERIMENT 1: Main table <<<"

# --- Sim2Live (base, no CI/LAAS) ---
launch_server configs/model_servers/rtc_kinetix.yaml "base"
run_benchmark "sim2live" "$BENCH_CONFIG"
stop_server

# --- +CI ---
launch_server configs/model_servers/rtc_kinetix_ci.yaml "ci"
run_benchmark "ci" "$BENCH_CONFIG"
stop_server

# --- +LAAS ---
launch_server configs/model_servers/rtc_kinetix_laas.yaml "laas"
run_benchmark "laas" "$BENCH_CONFIG"
stop_server

echo ""
echo ">>> EXPERIMENT 1 COMPLETE <<<"
echo ""

# ==========================================
# Experiment 2: Inference delay sweep
# ==========================================

echo ">>> EXPERIMENT 2: Delay sweep <<<"

for delay_config in \
    configs/model_servers/rtc_kinetix_delay_50ms.yaml \
    configs/model_servers/rtc_kinetix_delay_100ms.yaml \
    configs/model_servers/rtc_kinetix_delay_200ms.yaml \
    configs/model_servers/rtc_kinetix_delay_500ms.yaml \
; do
    # Extract delay label from filename: rtc_kinetix_delay_50ms.yaml -> delay_50ms
    label=$(basename "$delay_config" .yaml | sed 's/rtc_kinetix_//')

    launch_server "$delay_config" "$label"
    run_benchmark "$label" "$BENCH_CONFIG"
    stop_server
done

echo ""
echo "============================================"
echo " All pace=1.0 experiments complete!"
echo " Results in: ${RESULTS_BASE}/"
echo "============================================"
ls -la ${RESULTS_BASE}/*/
