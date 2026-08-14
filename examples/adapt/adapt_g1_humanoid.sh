#!/bin/bash
# Novel-embodiment adaptation: G1 humanoid, BrainCo hands (26-dim joint-space action,
# 4-view 768x320 stack). Needs $CLAP_OXE_BASE_PATH to contain a g1_humanoid/ dataset
# tree (see clap-preprocess-g1) and a matching dataset_meta_info/g1_humanoid/stat.json.
# Usage: bash examples/adapt/adapt_g1_humanoid.sh [--override key=val (e.g., training.max_train_steps=1) ...]
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config configs/experiment/adapt_g1_humanoid.yaml \
    "$@"
