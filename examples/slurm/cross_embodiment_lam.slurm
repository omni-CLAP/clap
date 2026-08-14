#!/bin/bash
#SBATCH --job-name=cross_embodiment_lam
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=450G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out

# VARIANT selects which latent-action extractor run to train against ("clap" | "dreamdojo").
# Usage: sbatch examples/slurm/cross_embodiment_lam.slurm
#        VARIANT=dreamdojo sbatch examples/slurm/cross_embodiment_lam.slurm

cd "$(dirname "$0")/../.."
source bash_scripts/setup.bash

# set up networking (needed on compute nodes without direct internet, e.g. for HF/GCS downloads)
module load proxy/default || true  # best-effort -- harmless if this node has no such module (e.g. a login node)

bash examples/train/cross_embodiment_lam.sh
