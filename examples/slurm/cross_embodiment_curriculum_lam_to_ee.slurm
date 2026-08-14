#!/bin/bash
#SBATCH --job-name=cross_embodiment_curriculum_lam_to_ee
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=450G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out

# Depends on cross_embodiment_lam.slurm's checkpoint already existing (see
# examples/train/cross_embodiment_curriculum_lam_to_ee.sh).
# Usage: sbatch examples/slurm/cross_embodiment_curriculum_lam_to_ee.slurm

cd "$(dirname "$0")/../.."
source bash_scripts/setup.bash

# set up networking (needed on compute nodes without direct internet, e.g. for HF/GCS downloads)
module load proxy/default || true  # best-effort -- harmless if this node has no such module (e.g. a login node)

bash examples/train/cross_embodiment_curriculum_lam_to_ee.sh
