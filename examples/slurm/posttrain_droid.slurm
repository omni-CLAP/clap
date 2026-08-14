#!/bin/bash
#SBATCH --job-name=posttrain_droid
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out

# BASE selects the source cross-embodiment checkpoint (see examples/posttrain/posttrain_droid.sh).
# Usage: BASE=oxe_lam_clap sbatch examples/slurm/posttrain_droid.slurm

cd "$(dirname "$0")/../.."
source bash_scripts/setup.bash

# set up networking (needed on compute nodes without direct internet, e.g. for HF/GCS downloads)
module load proxy/default || true  # best-effort -- harmless if this node has no such module (e.g. a login node)

bash examples/posttrain/posttrain_droid.sh
