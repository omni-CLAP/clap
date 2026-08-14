#!/usr/bin/env bash
# Site-specific environment for training/eval/rollout with clap.
#
# Copy this file (e.g. `cp env.example.sh env.sh`), fill in real values for
# your cluster/data layout, and `source env.sh` before running any
# clap-* command. Never hardcode these values in code or configs/*.yaml —
# see clap.config.paths.PathConfig and clap.data.oxe_catalog for how each
# var is read.

# --- Required ---------------------------------------------------------------

# Root of the OXE mp4/annotation dataset tree (<root>/<dataset_name>/videos/...).
export CLAP_OXE_BASE_PATH=/path/to/oxe

# Getting started with no data of your own: this package ships a handful of real,
# full-length droid/bridge/taco_play val episodes at sample_data/oxe/ (~10MB total) --
# enough to run clap-rollout-replay/clap-teleop/clap-rollout-deploy end to end against
# a real checkpoint (see examples/getting_started/). No annotation-subdir override
# needed here -- unlike most real droid deployments, the sample droid episodes' single
# annotation/ dir already carries everything ("state" + joint position) in one file.
# export CLAP_OXE_BASE_PATH=$(pwd)/sample_data/oxe

# --- Optional paths (sane defaults if unset) --------------------------------

# Root for precomputed LAM latent-action extractions, if it differs from
# CLAP_OXE_BASE_PATH. Only needed when training/evaluating with conditioning="lam".
export CLAP_OXE_LAM_ROOT=/path/to/oxe_lam

# Root of the held-out OXE tree (austin_sailor/berkeley_autolab_ur5/stanford_hydra/
# utaustin_mutex) -- generalization targets unseen during cross-embodiment modeling,
# kept under a separate root from CLAP_OXE_BASE_PATH. Only needed for
# examples/slurm/run_all_eval.sh's held-out section.
# export CLAP_OXE_HELD_OUT_PATH=/path/to/oxe_held_out

# Where per-dataset stat.json normalization bounds are read from/written to.
# Defaults to "dataset_meta_info" (relative to cwd), which this package ships
# pre-populated with stat.json for every OXE_CATALOG dataset (egodex has no
# end-effector action, so it needs none). Only set this to point at a different
# directory, or run clap-preprocess-oxe-meta / clap-preprocess-g1 to rebuild an entry.
# export CLAP_META_INFO_ROOT=dataset_meta_info

# Base dir relative checkpoint paths (finetune_ckpt/output_dir) resolve against.
# export CLAP_CHECKPOINT_ROOT=model_ckpt

# HF repo ids or local paths for the base SVD/CLIP models. Unset falls back
# to downloading the public HF models.
# export CLAP_SVD_MODEL_PATH=stabilityai/stable-video-diffusion-img2vid
# export CLAP_CLIP_MODEL_PATH=openai/clip-vit-base-patch32

# --- Dataset-layout overrides (per-deployment; DataConfig defaults are just
# generic placeholders, not real subdir names) -------------------------------

# Real LAM latent-action extraction run to read (DataConfig.oxe_lam_subdir's
# default "latent_actions" is a placeholder). Also settable per-experiment
# via configs/experiment/*.yaml.
# export CLAP_OXE_LAM_SUBDIR=my_latent_action_run

# Per-dataset annotation-subdir override, for a deployment whose copy of a
# given dataset's annotations live under a different directory name than the
# package default ("annotation"). <NAME> is the dataset name, upper-cased
# (see clap.data.oxe_catalog.OXE_CATALOG for the full list of names).
# export CLAP_DROID_ANNOTATION_SUBDIR=annotation2

# Pre-encoded SVD video-latent subdir name, if it differs from the package
# default "latent_videos_svd" (e.g. a copy staged under a different name on
# faster local/scratch storage -- see clap.data.base.get_latent_video_subdir).
# export CLAP_LATENT_VIDEO_SUBDIR=latent_videos