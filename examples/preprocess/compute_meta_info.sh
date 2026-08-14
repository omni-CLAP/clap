#!/bin/bash
# Compute p01/p99 action-normalization stat.json for a dataset that doesn't already
# have one under dataset_meta_info/ (this repo ships stat.json for every OXE_CATALOG
# entry except egodex, which has no end-effector action -- see clap.preprocess.oxe_meta's
# own docstring for the full usage reference this wraps).
#
# Usage:
#   bash examples/preprocess/compute_meta_info.sh                    # every ee7 dataset (see below)
#   DATASET=my_new_dataset bash examples/preprocess/compute_meta_info.sh
#   DATASET=bimanual_yam FULL_STATE=1 bash examples/preprocess/compute_meta_info.sh   # joint-space: raw state, not state[:6]+gripper
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${CLAP_OXE_BASE_PATH:?CLAP_OXE_BASE_PATH must be set}"

FULL_STATE=${FULL_STATE:-0}
EXTRA_ARGS=()
[[ "$FULL_STATE" == "1" ]] && EXTRA_ARGS+=(--full-state)

if [[ -n "${DATASET:-}" ]]; then
    clap-preprocess-oxe-meta --oxe-base-path "$CLAP_OXE_BASE_PATH" --dataset-name "$DATASET" "${EXTRA_ARGS[@]}"
else
    # ee7 (cartesian) datasets: the 7 cross-embodiment sets + the 4 held-out
    # generalization targets (see clap.data.oxe_catalog.OXE_CATALOG). bimanual_yam
    # (joint-space, needs --full-state) and g1_humanoid (its own clap-preprocess-g1
    # pipeline writes stat.json directly) are intentionally not in this default loop.
    for ds in bridge fractal bc_z fmb taco_play furniture_bench droid \
              austin_sailor berkeley_autolab_ur5 stanford_hydra utaustin_mutex; do
        clap-preprocess-oxe-meta --oxe-base-path "$CLAP_OXE_BASE_PATH" --dataset-name "$ds"
    done
fi
