#!/bin/bash
# Cross-embodiment modeling, per-step language-caption conditioning.
# VARIANT selects the action-caption style (see clap.data.action_caption).
# Usage: bash examples/train/cross_embodiment_language.sh [--override key=val (e.g., training.max_train_steps=1) ...]
#        VARIANT=absolute bash examples/train/cross_embodiment_language.sh
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

VARIANT=${VARIANT:-relative}  # "absolute" | "relative"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config "configs/experiment/cross_embodiment_oxe_language_${VARIANT}.yaml" \
    "$@"