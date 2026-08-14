#!/bin/bash
# Curriculum: continue a LAM-latent checkpoint with EE-cartesian actions (action_encoder
# reset, since its input dim changes 32-dim LAM -> 7-dim EE). Depends on
# cross_embodiment_lam.sh's checkpoint already existing at the finetune_ckpt path set
# in configs/experiment/cross_embodiment_oxe_curriculum_lam_ee.yaml.
# Usage: bash examples/train/cross_embodiment_curriculum_lam_to_ee.sh [--override key=val (e.g., training.max_train_steps=1) ...]
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config configs/experiment/cross_embodiment_oxe_curriculum_lam_ee.yaml \
    "$@"
