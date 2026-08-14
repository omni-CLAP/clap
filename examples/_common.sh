# Sourced by every examples/*.sh launcher: GPU count + a free rendezvous port for `accelerate launch`.
# Not meant to be run directly.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# if on slurm, set up networking (compute nodes otherwise have no outbound
# internet access, breaking the HF download below) -- no-op off-cluster,
# where the `module` command doesn't exist
if command -v module &>/dev/null; then
    module load proxy/default || true  # best-effort -- e.g. fails outright on login nodes even though the module command exists
fi

NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}}
MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
