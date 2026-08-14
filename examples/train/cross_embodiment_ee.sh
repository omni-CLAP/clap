#!/bin/bash
# Cross-embodiment modeling, EE-cartesian conditioning: 7 OXE datasets, 7-dim action.
# Usage: bash examples/train/cross_embodiment_ee.sh [--override key=val (e.g., training.max_train_steps=1) ...]
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config configs/experiment/cross_embodiment_oxe_ee.yaml \
    "$@"
