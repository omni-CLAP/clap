#!/bin/bash
# Post-train a cross-embodiment checkpoint onto droid alone (EE-cartesian actions,
# regardless of which family BASE was originally trained with — see
# clap.eval.experiments's comment on _POST_TRAIN_DROID for why).
# BASE selects the source checkpoint: oxe_ee | oxe_lam_clap | oxe_lam_dreamdojo |
#   oxe_curriculum_lam_ee | oxe_language_absolute | oxe_language_relative
# Usage: BASE=oxe_lam_clap bash examples/posttrain/posttrain_droid.sh [--override key=val (e.g., training.max_train_steps=1) ...]
set -euo pipefail
source "$(dirname "$0")/../_common.sh"

BASE=${BASE:-oxe_ee}
TAG="post_train_${BASE}_droid"
RESET=true
[ "$BASE" = "oxe_ee" ] && RESET=false  # same 7-dim EE action shape as the base -> no reset needed

accelerate launch --num_processes="${NUM_GPUS}" --main_process_port="${MASTER_PORT}" \
    -m clap.training.train \
    --config "configs/experiment/cross_embodiment_${BASE}.yaml" \
    --override \
        data.conditioning=droid \
        model.conditioning=ee \
        model.action_dim=7 \
        "training.tag=${TAG}" \
        "training.output_dir=model_ckpt/${TAG}" \
        "training.finetune_ckpt=cross_embodiment_${BASE}/checkpoint-100000.pt" \
        "training.reset_action_encoder=${RESET}" \
    "$@"
