#!/bin/bash
set -e
Xvfb :99 -screen 0 1280x1024x24 +extension GLX +render -noreset &
export DISPLAY=:99
sleep 2
# Activate conda env directly instead of `conda run` — conda run waits
# for all child processes (including CoppeliaSim) which prevents exit.
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate rlbench
vla-eval "$@"
