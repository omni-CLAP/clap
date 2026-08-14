#!/bin/bash
# Novel-embodiment adaptation: bimanual YAM (14-dim joint-space action).
# Usage: bash examples/adapt/adapt_bimanual_yam.sh [--override key=val (e.g., training.max_train_steps=1) ...]
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config configs/experiment/adapt_bimanual_yam.yaml \
    "$@"
