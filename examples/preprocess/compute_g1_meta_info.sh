#!/bin/bash
# Download + convert the unitreerobotics/* G1-humanoid HF datasets into this repo's OXE
# mp4/annotation layout under $CLAP_OXE_BASE_PATH/g1_humanoid, and write its
# dataset_meta_info/g1_humanoid/stat.json in the same step -- unlike every other dataset
# (see examples/preprocess/compute_meta_info.sh), g1_humanoid's stat.json doesn't go through
# clap-preprocess-oxe-meta; clap-preprocess-g1 writes it directly as part of the conversion
# (see clap.preprocess.g1_humanoid's own docstring for the full usage reference this wraps).
#
# Usage:
#   bash examples/preprocess/compute_g1_meta_info.sh
#   VAL_EPISODES=20 bash examples/preprocess/compute_g1_meta_info.sh   # more episodes held out for val
#   DATASET_INDEX=0 bash examples/preprocess/compute_g1_meta_info.sh  # only this index into --datasets, for SLURM array jobs
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${CLAP_OXE_BASE_PATH:?CLAP_OXE_BASE_PATH must be set}"

EXTRA_ARGS=()
[[ -n "${VAL_EPISODES:-}" ]] && EXTRA_ARGS+=(--val-episodes "$VAL_EPISODES")
[[ -n "${DATASET_INDEX:-}" ]] && EXTRA_ARGS+=(--dataset-index "$DATASET_INDEX")
[[ -n "${HF_HOME:-}" ]] && EXTRA_ARGS+=(--hf-cache-dir "$HF_HOME")

clap-preprocess-g1 --out-base "$CLAP_OXE_BASE_PATH" "${EXTRA_ARGS[@]}"