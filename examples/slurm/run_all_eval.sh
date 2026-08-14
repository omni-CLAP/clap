#!/bin/bash
# Submit one sbatch eval job (examples/slurm/eval_job.sh) per (experiment, test-set)
# combination, using clap.eval.experiments' own category registry (not a hardcoded
# list here) so this never drifts from the actual experiment set.
#
# Model → test-set mapping:
#   Baselines                  → droid_100 (baseline_droid) / bridge_100 (baseline_bridge)
#   Cross-embodiment           → bridge_100, droid_100, oxe_mix_100
#   Cross-embodiment held-out  → austin_sailor_100, berkeley_autolab_ur5_100,
#                                stanford_hydra_100, utaustin_mutex_100 (unseen during
#                                pretraining; data lives under CLAP_OXE_HELD_OUT_PATH,
#                                not CLAP_OXE_BASE_PATH -- see the held-out section below)
#   Post-trained DROID models  → droid_100
#   Post-trained Bridge models → bridge_100
#   Adaptation                 → bimanual_yam_val (bimanual YAM) / g1_humanoid_val (G1 humanoid)
#
# Usage:
#   bash examples/slurm/run_all_eval.sh             # submit all
#   DRY_RUN=1 bash examples/slurm/run_all_eval.sh    # print sbatch commands only
#   CKPT_ITER=100000 bash examples/slurm/run_all_eval.sh
#   RUN_LOCAL=1 bash examples/slurm/run_all_eval.sh  # run each eval_job.sh directly via bash,
#                                                     # sequentially, instead of submitting to
#                                                     # slurm -- for a machine without slurm, or
#                                                     # debugging on an already-allocated GPU
#
# Required environment: CLAP_OXE_BASE_PATH (see env.example.sh)
# Required for the held-out section: CLAP_OXE_HELD_OUT_PATH (root of the held-out OXE tree,
#   separate from CLAP_OXE_BASE_PATH -- the held-out section is skipped with a warning if unset)
#
# Optional environment (forwarded to eval_job.sh via --export):
#   CKPT_ITER            checkpoint iteration (default: last)
#   EVAL_OUTPUTS_ROOT     output root (default: PROJECT_DIR/eval_outputs)
#   NUM_INFERENCE_STEPS   (default: 50)
#   MAX_CHUNKS            (default: 0 = unlimited)
#   MAX_EPISODES_PER_DATASET  (default: 0 = unlimited)
#   NO_FVD / NO_FID       set to 1 to skip that metric
#   DATASETS              space-separated dataset filter, e.g. "droid" (default: all)
#   SKIP_DONE             set to 0 to re-run finished evals (default: 1)
#   TIME_LIMIT            slurm time limit (default: 3:00:00)
set -euo pipefail

# PROJECT_DIR defaults to this script's repo root (examples/slurm/../..), so it works
# regardless of the caller's cwd; override it if you're calling from somewhere unusual.
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
source bash_scripts/setup.bash  # or your own env file exporting the CLAP_* variables

# set up networking (needed on compute nodes without direct internet, e.g. for HF/GCS downloads)
module load proxy/default || true  # best-effort -- harmless if this node has no such module (e.g. a login node)

: "${CLAP_OXE_BASE_PATH:?CLAP_OXE_BASE_PATH must be set}"

DRY_RUN="${DRY_RUN:-0}"
RUN_LOCAL="${RUN_LOCAL:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
CKPT_ITER="${CKPT_ITER:-last}"
EVAL_OUTPUTS_ROOT="${EVAL_OUTPUTS_ROOT:-$PROJECT_DIR/eval_outputs}"
TIME_LIMIT="${TIME_LIMIT:-3:00:00}"
JOB_SCRIPT="$PROJECT_DIR/examples/slurm/eval_job.sh"

mkdir -p slurm_outputs/eval

n_submitted=0
n_skipped=0
n_failed=0
failed_jobs=()

# Submits one eval_job.sh sbatch job for (exp, ts). ts="" means "no cached test
# set" -- eval_job.sh then falls back to the experiment's own default_datasets.
# held_out=1 points CLAP_OXE_BASE_PATH/CLAP_OXE_LAM_ROOT at CLAP_OXE_HELD_OUT_PATH for
# just this job (an explicit KEY=VALUE in --export overrides the ALL-inherited value for
# that key), since held-out datasets live under a separate root, not the main OXE tree.
submit() {
    local exp="$1" ts="$2" held_out="${3:-0}"
    local out_dir="$EVAL_OUTPUTS_ROOT/$exp/${ts:-default}/ckpt_$CKPT_ITER"
    local job_name="eval_${exp}_${ts:-default}"

    # per_view_summary.json is clap-eval's last write -- its presence means the run finished.
    if [[ "$SKIP_DONE" == "1" && -f "$out_dir/per_view_summary.json" ]]; then
        echo "[skip] $exp / ${ts:-default} — already done"
        (( n_skipped++ )) || true
        return
    fi

    # Env vars this job needs, shared between the sbatch --export path and the RUN_LOCAL
    # (plain bash) path below -- ALL (sbatch-only) keeps this shell's existing environment
    # too, e.g. the CLAP_* vars sourced from bash_scripts/setup.bash; RUN_LOCAL already runs
    # in this same shell, so it inherits that environment without needing an ALL equivalent.
    local job_env=(
        "EXPERIMENT=$exp" "TEST_SET=$ts" "CKPT_ITER=$CKPT_ITER" "EVAL_OUTPUTS_ROOT=$EVAL_OUTPUTS_ROOT"
        "NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}" "MAX_CHUNKS=${MAX_CHUNKS:-0}"
        "MAX_EPISODES_PER_DATASET=${MAX_EPISODES_PER_DATASET:-0}"
        "NO_FVD=${NO_FVD:-0}" "NO_FID=${NO_FID:-0}" "DATASETS=${DATASETS:-}"
    )
    if [[ "$held_out" == "1" ]]; then
        job_env+=("CLAP_OXE_BASE_PATH=$CLAP_OXE_HELD_OUT_PATH" "CLAP_OXE_LAM_ROOT=$CLAP_OXE_HELD_OUT_PATH")
    fi

    if [[ "$RUN_LOCAL" == "1" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run] env ${job_env[*]} bash $JOB_SCRIPT"  # print the local invocation only, don't run
        else
            mkdir -p "slurm_outputs/eval/${job_name}"
            # `if ! ( pipeline )` -- not a bare pipeline -- is what actually shields this from
            # set -e: one job's eval_job.sh failing (bad checkpoint, CUDA OOM, ...) must not
            # abort every job still queued behind it in this sequential loop.
            if ! ( env "${job_env[@]}" bash "$JOB_SCRIPT" 2>&1 | tee "slurm_outputs/eval/${job_name}/out_${job_name}_local.out" ); then
                echo "[FAILED] $exp / ${ts:-default} -- see slurm_outputs/eval/${job_name}/out_${job_name}_local.out"
                failed_jobs+=("$exp/${ts:-default}")
                (( n_failed++ )) || true
                return
            fi
            echo "[ran] $exp / ${ts:-default}"
        fi
    else
        local export_str="ALL,$(IFS=,; echo "${job_env[*]}")"
        local cmd=(
            sbatch
            --job-name="$job_name"
            --time="$TIME_LIMIT"
            --output="slurm_outputs/eval/${job_name}/out_${job_name}_%j.out"
            --export="$export_str"
            "$JOB_SCRIPT"
        )
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry-run] ${cmd[*]}"  # print the sbatch invocation only, don't submit
        else
            mkdir -p "slurm_outputs/eval/${job_name}"  # sbatch --output needs the dir to exist first
            # sbatch itself failing (bad script, slurm down, quota) shouldn't stop the rest of
            # the batch from being submitted -- only this one job's *submission* fails, not its
            # (not-yet-run) execution, so this is a much rarer case than the RUN_LOCAL one above.
            if ! "${cmd[@]}"; then
                echo "[FAILED] $exp / ${ts:-default} -- sbatch submission failed"
                failed_jobs+=("$exp/${ts:-default}")
                (( n_failed++ )) || true
                return
            fi
            echo "[submitted] $exp / ${ts:-default}"
        fi
    fi
    (( n_submitted++ )) || true  # `|| true`: bash treats a 0 result from ((...)) as failure under set -e
}

# Pull each category's experiment names from the registry itself (clap.eval.experiments'
# EXPERIMENTS/_CATEGORIES dicts), not a hardcoded list here, so this can't drift out of
# sync with the actual experiment set as entries are added/removed.
category() { python3 -c "from clap.eval.experiments import list_experiments; print(' '.join(list_experiments('$1')))"; }

echo "================================================================"
echo "  clap evaluation — $([[ "$RUN_LOCAL" == "1" ]] && echo "running locally via bash" || echo "submitting slurm jobs")"
echo "  CKPT_ITER=$CKPT_ITER   DRY_RUN=$DRY_RUN   RUN_LOCAL=$RUN_LOCAL"
echo "================================================================"

echo ""
echo "--- Baselines ---"
submit "baseline_droid" "droid_100"
submit "baseline_bridge" "bridge_100"

echo ""
echo "--- Cross-embodiment models (x3 test sets) ---"
for exp in $(category cross_embodiment); do
    submit "$exp" "bridge_100"
    submit "$exp" "droid_100"
    submit "$exp" "oxe_mix_100"
done

echo ""
echo "--- Cross-embodiment held-out (x4 held-out test sets) ---"
if [[ -z "${CLAP_OXE_HELD_OUT_PATH:-}" ]]; then
    echo "  CLAP_OXE_HELD_OUT_PATH not set — skipping held-out section"
else
    for exp in $(category cross_embodiment); do
        for ts in austin_sailor_100 berkeley_autolab_ur5_100 stanford_hydra_100 utaustin_mutex_100; do
            submit "$exp" "$ts" 1  # held_out=1 -- swap in CLAP_OXE_HELD_OUT_PATH for this job
        done
    done
fi

echo ""
echo "--- Post-trained DROID models ---"
for exp in $(category droid); do
    submit "$exp" "droid_100"
done

echo ""
echo "--- Post-trained Bridge models ---"
for exp in $(category bridge); do
    submit "$exp" "bridge_100"
done

echo ""
echo "--- Adaptation ---"
submit "adapt_bimanual_yam" "bimanual_yam_val"
submit "adapt_g1_humanoid" "g1_humanoid_val"

echo ""
echo "================================================================"
echo "  $([[ "$RUN_LOCAL" == "1" ]] && echo "ran" || echo "submitted")=$n_submitted  skipped=$n_skipped  failed=$n_failed"
if [[ "$n_failed" -gt 0 ]]; then
    echo "  Failed:"
    for j in "${failed_jobs[@]}"; do
        echo "    - $j"
    done
fi
[[ "$RUN_LOCAL" != "1" ]] && echo "  Monitor:  squeue -u \$USER"
echo "  Results:  clap-eval-aggregate --results-root $EVAL_OUTPUTS_ROOT/<experiment>/<test_set>"
echo "================================================================"

# Exit non-zero only after every job has been attempted, so a scripted/CI caller can still
# detect a bad run -- without this, a run with failures would otherwise report success (exit 0).
[[ "$n_failed" -gt 0 ]] && exit 1
exit 0
