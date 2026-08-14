#!/bin/bash
# Single-embodiment baseline: droid alone, trained from scratch (no cross-embodiment modeling).
# Usage: bash examples/baselines/baseline_droid.sh [--override key=val (e.g., training.max_train_steps=1) ...]
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config configs/experiment/baseline_droid.yaml \
    "$@"
