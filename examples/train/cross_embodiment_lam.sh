#!/bin/bash
# Cross-embodiment modeling, LAM latent-action conditioning.
# VARIANT selects which latent-action extractor run to train against.
# Usage: bash examples/train/cross_embodiment_lam.sh [--override key=val (e.g., training.max_train_steps=1) ...]
#        VARIANT=dreamdojo bash examples/train/cross_embodiment_lam.sh
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

VARIANT=${VARIANT:-clap}  # "clap" | "dreamdojo"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config "configs/experiment/cross_embodiment_oxe_lam_${VARIANT}.yaml" \
    "$@"