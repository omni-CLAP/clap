#!/bin/bash
# Offline checkpoint evaluation: rollout + PSNR/SSIM/LPIPS/FVD/FID against a cached test set.
# EXPERIMENT selects the checkpoint (see clap.eval.experiments); TEST_SET selects the episodes.
# Usage: EXPERIMENT=cross_embodiment_oxe_ee TEST_SET=oxe_mix_100 bash examples/rollout/rollout_eval.sh [-- extra clap-eval flags]
set -euo pipefail
cd "$(dirname "$0")/../.."

EXPERIMENT=${EXPERIMENT:-cross_embodiment_oxe_ee}
TEST_SET=${TEST_SET:-oxe_mix_100}
CKPT=${CKPT:-last}
CONFIG=${CONFIG:-configs/experiment/${EXPERIMENT}.yaml}

clap-eval \
    --config "${CONFIG}" \
    --experiment "${EXPERIMENT}" \
    --ckpt "${CKPT}" \
    --test-set "${TEST_SET}" \
    "$@"