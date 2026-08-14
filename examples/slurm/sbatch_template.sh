#!/bin/bash
#SBATCH --job-name=REPLACE_ME
#SBATCH --nodes=1
#SBATCH --partition=REPLACE_ME
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=450G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out

# Template for wrapping any examples/*.sh launcher in a SLURM job.
# Copy this file, fill in the #SBATCH headers above, and set LAUNCHER below.
# See examples/slurm/cross_embodiment_ee.slurm and examples/slurm/adapt_g1_humanoid.slurm
# for filled-in examples.

cd "$(dirname "$0")/../.."
source bash_scripts/setup.bash  # or your own env file exporting the CLAP_* variables

# set up networking (needed on compute nodes without direct internet, e.g. for HF/GCS downloads)
module load proxy/default || true  # best-effort -- harmless if this node has no such module (e.g. a login node)

LAUNCHER=examples/train/REPLACE_ME.sh
bash "${LAUNCHER}"