#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_outputs/eval/%x/out_%x_%j.out

# One clap-eval job: EXPERIMENT (required) x TEST_SET (optional -- omit to eval on
# the experiment's own default_datasets instead of a cached test set, e.g. for the
# adaptation experiments). Meant to be submitted via examples/slurm/run_all_eval.sh,
# which sets these through `sbatch --export`.
#
# Does NOT source bash_scripts/setup.bash itself -- the caller must have already sourced
# it (or their own env file exporting the CLAP_* variables and activating the venv) before
# invoking this script, so that job-specific overrides (e.g. run_all_eval.sh's held-out
# CLAP_OXE_BASE_PATH swap) reach clap-eval unclobbered; both `sbatch --export=ALL,...` and
# a plain `env KEY=val bash eval_job.sh` propagate the sourcing shell's full environment
# (PATH/venv included) to this script, so re-sourcing here would just overwrite it back to
# setup.bash's own defaults. To run this standalone:
#   source bash_scripts/setup.bash  # or your own env file
#   EXPERIMENT=cross_embodiment_oxe_ee TEST_SET=oxe_mix_100 sbatch examples/slurm/eval_job.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"

# set up networking (needed on compute nodes without direct internet, e.g. for HF/GCS downloads)
module load proxy/default || true  # best-effort -- harmless if this node has no such module (e.g. a login node)

nvidia-smi

EXPERIMENT=${EXPERIMENT:?EXPERIMENT must be set}
TEST_SET=${TEST_SET:-}  # empty -> --datasets/default_datasets instead of a cached test set
CKPT_ITER=${CKPT_ITER:-last}
EVAL_OUTPUTS_ROOT=${EVAL_OUTPUTS_ROOT:-$PROJECT_DIR/eval_outputs}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
MAX_CHUNKS=${MAX_CHUNKS:-0}
MAX_EPISODES_PER_DATASET=${MAX_EPISODES_PER_DATASET:-0}
NO_FVD=${NO_FVD:-0}
NO_FID=${NO_FID:-0}
DATASETS=${DATASETS:-}  # optional space-separated dataset filter (e.g. "droid" or "bridge droid")

# Post-train experiments reuse their cross-embodiment base's config (same data/model
# shape -- clap-eval overrides model.conditioning/action_dim from the experiment
# registry regardless); every other category has its own configs/experiment/<name>.yaml.
CONFIG="configs/experiment/${EXPERIMENT}.yaml"
if [[ "$EXPERIMENT" == post_train_* ]]; then
    BASE=${EXPERIMENT#post_train_}
    BASE=${BASE%_droid}
    BASE=${BASE%_bridge}
    CONFIG="configs/experiment/cross_embodiment_${BASE}.yaml"
fi

OUT_DIR="$EVAL_OUTPUTS_ROOT/$EXPERIMENT/${TEST_SET:-default}/ckpt_$CKPT_ITER"

echo "========================================"
echo "  Experiment:   ${EXPERIMENT}"
echo "  Config:       ${CONFIG}"
echo "  Test set:     ${TEST_SET:-<none, using default_datasets>}"
echo "  Checkpoint:   ${CKPT_ITER}"
echo "  Output dir:   ${OUT_DIR}"
echo "  Steps:        ${NUM_INFERENCE_STEPS}"
echo "  Max chunks:   ${MAX_CHUNKS}  Max episodes/dataset: ${MAX_EPISODES_PER_DATASET}"
echo "  NO_FVD:       ${NO_FVD}  NO_FID: ${NO_FID}"
echo "========================================"

EXTRA_ARGS=()
[[ "$NO_FVD" == "1" ]] && EXTRA_ARGS+=(--no-fvd)
[[ "$NO_FID" == "1" ]] && EXTRA_ARGS+=(--no-fid)
[[ -n "$TEST_SET" ]] && EXTRA_ARGS+=(--test-set "$TEST_SET")
# shellcheck disable=SC2206
[[ -n "$DATASETS" ]] && EXTRA_ARGS+=(--datasets $DATASETS)

clap-eval \
    --config "$CONFIG" \
    --experiment "$EXPERIMENT" \
    --ckpt "$CKPT_ITER" \
    --save-dir "$OUT_DIR" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --max-chunks "$MAX_CHUNKS" \
    --max-episodes-per-dataset "$MAX_EPISODES_PER_DATASET" \
    "${EXTRA_ARGS[@]}"
