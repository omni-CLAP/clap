# Sourced by every examples/getting_started/*.sh script: points CLAP_OXE_BASE_PATH at
# this package's shipped sample data and downloads a checkpoint from HF if it isn't
# already cached locally. Not meant to be run directly.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root

export CLAP_OXE_BASE_PATH="$(pwd)/sample_data/oxe"  # droid/bridge/taco_play val episodes shipped with this package
# LAM family: droid/bridge/taco_play also ship precomputed latent_actions.npy for their 3 sample
# episodes each, under the package's own oxe_lam_subdir default -- point explicitly at the sample data
export CLAP_OXE_LAM_ROOT="$(pwd)/sample_data/oxe"
export CLAP_OXE_LAM_SUBDIR=latent_actions

CKPT_NAME=${CKPT_NAME:-cross_embodiment_oxe_curriculum_lam_ee}  # short alias (e.g. clap-curr, calp-ee) or full experiment name -- see CKPT_ALIASES in clap.eval.experiments
# Resolves the alias AND looks up this experiment's registered family (ee/lam/language) in the
# same call, so replay.sh can default FAMILY to the value the checkpoint actually needs instead
# of a hardcoded "ee" -- a mismatched FAMILY (e.g. a language checkpoint replayed with FAMILY=ee)
# fails deep inside the model's conditioning branch with a confusing error, not an upfront one.
read -r CKPT_NAME CKPT_FAMILY CKPT_DISPLAY_NAME <<< "$(python3 -c "
from clap.eval.experiments import display_ckpt_name, get_experiment, resolve_ckpt_name
name = resolve_ckpt_name('${CKPT_NAME}')
print(name, get_experiment(name).family, display_ckpt_name(name))
")"
CKPT_REPO=${CKPT_REPO:-omni-CLAP/CLAP}           # see the README's checkpoint table for every published <name>
CKPT_STEP=${CKPT_STEP:-100000}                   # every checkpoint we publish is step 100000

CKPT_DIR="model_ckpt/${CKPT_NAME}"
CKPT_PATH="${CKPT_DIR}/checkpoint-${CKPT_STEP}.pt"
CONFIG="configs/experiment/${CKPT_NAME}.yaml"

# unset annotation directories, if already set -- sample_data/oxe/* only ever ships a plain
# "annotation" subdir, so an inherited override would break this demo
unset CLAP_DROID_ANNOTATION_SUBDIR CLAP_BRIDGE_ANNOTATION_SUBDIR CLAP_TACO_PLAY_ANNOTATION_SUBDIR \
      CLAP_BIMANUAL_YAM_ANNOTATION_SUBDIR CLAP_G1_HUMANOID_ANNOTATION_SUBDIR

# if on slurm, set up networking (compute nodes otherwise have no outbound
# internet access, breaking the HF download below) -- no-op off-cluster,
# where the `module` command doesn't exist
if command -v module &>/dev/null; then
    module load proxy/default || true  # best-effort -- e.g. fails outright on login nodes even though the module command exists
fi

RUN_TAG="${CKPT_NAME}_$(date +%Y%m%d_%H%M%S)"  # disambiguates eval_outputs/ across checkpoints/runs

if [[ ! -f "$CKPT_PATH" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        # Compute-node proxies (module proxy/default) often only allowlist huggingface.co
        # itself, not the CDN host it redirects large-file downloads to -- that shows up as
        # an httpx.ProxyError: 403 Forbidden here even with the module loaded. Pre-fetching
        # from a login node (usually unrestricted egress) sidesteps it entirely.
        echo "Warning: downloading inside a SLURM job (SLURM_JOB_ID=${SLURM_JOB_ID}) -- if this" >&2
        echo "  403s past the proxy, pre-download the checkpoint from a login node instead:" >&2
        echo "    CKPT_NAME=${CKPT_NAME} bash examples/getting_started/_common.sh" >&2
        echo "  then re-run this job; it'll find checkpoint-${CKPT_STEP}.pt already cached." >&2
    fi
    echo "Downloading ${CKPT_NAME} (checkpoint-${CKPT_STEP}.pt) from ${CKPT_REPO}..."
    export HF_HUB_ENABLE_HF_TRANSFER=1  # faster Rust-backed download, if hf_transfer is installed
    export HF_HUB_DISABLE_XET=1  # Xet's separate cas-server.xethub.hf.co host often isn't proxy-allowlisted even when huggingface.co is; fall back to classic HTTP/LFS (still hf_transfer-accelerated)
    hf download "$CKPT_REPO" --include "${CKPT_NAME}/checkpoint-${CKPT_STEP}.pt" --local-dir model_ckpt
fi
