#!/bin/bash
set -e
Xvfb :99 -screen 0 1280x1024x24 +extension GLX +render -noreset &
export DISPLAY=:99
# Wait for Xvfb to be ready before continuing.
retries=10
while [ ! -S "/tmp/.X11-unix/X99" ] && [ $retries -gt 0 ]; do
    sleep 0.5
    retries=$((retries-1))
done
if [ $retries -eq 0 ]; then
    echo "Xvfb failed to start" >&2
    exit 1
fi
# Activate conda env directly instead of `conda run` — conda run waits
# for all child processes (including CoppeliaSim) which prevents exit.
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rlbench
vla-eval "$@"
