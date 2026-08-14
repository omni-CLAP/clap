#!/bin/bash
# Getting-started demo: replay a real checkpoint against this package's shipped
# sample data (sample_data/oxe/, real full-length val episodes for droid/bridge/
# taco_play/bimanual_yam/g1_humanoid) -- no dataset of your own needed. Downloads
# the checkpoint from HF on first run (see the README's checkpoint table for
# every published CKPT_NAME).
#
# "ee", "language", and "lam" family checkpoints all work against this sample data --
# language captions are built at runtime from the same state array ee uses
# (clap.data.action_caption), no precomputed caption files needed; droid/bridge/taco_play
# also ship precomputed latent_actions.npy (clap.data.lam) for lam-family checkpoints.
# bimanual_yam/g1_humanoid have neither language captions nor latent actions, so only
# their own ee-family adapt_* checkpoints work against them.
#
# bimanual_yam/g1_humanoid are novel-embodiment adaptation targets (14-/26-dim
# joint-space actions, not the 7-dim EE cartesian the cross_embodiment_* /
# baseline_* checkpoints use) -- pair them with their matching adapt_* checkpoint,
# not an arbitrary CKPT_NAME. Their sample episodes also run much longer than
# droid/bridge/taco_play's (1000+ frames vs. tens) -- set MAX_CHUNKS for a quick
# demo instead of autoregressively replaying the full episode.
#
# Usage: bash examples/getting_started/replay.sh
#        DATASET=bridge bash examples/getting_started/replay.sh
#        CKPT_NAME=baseline_droid DATASET=droid bash examples/getting_started/replay.sh
#        CKPT_NAME=clap-lang bash examples/getting_started/replay.sh   # FAMILY auto-derived (language)
#        CKPT_NAME=clap-lam DATASET=bridge bash examples/getting_started/replay.sh   # FAMILY auto-derived (lam)
#        CKPT_NAME=adapt_bimanual_yam DATASET=bimanual_yam MAX_CHUNKS=5 bash examples/getting_started/replay.sh
#        CKPT_NAME=adapt_g1_humanoid DATASET=g1_humanoid MAX_CHUNKS=5 bash examples/getting_started/replay.sh
#        TRIM_STATIC=1 DATASET=bridge bash examples/getting_started/replay.sh   # skip the near-static lead-in before real motion starts
set -euo pipefail
source "$(dirname "$0")/_common.sh"

DATASET=${DATASET:-droid}                       # droid | bridge | taco_play | bimanual_yam | g1_humanoid
FAMILY=${FAMILY:-$CKPT_FAMILY}                  # defaults to CKPT_NAME's own registered family (ee/lam/language, from _common.sh) -- override only to deliberately mismatch
MAX_CHUNKS=${MAX_CHUNKS:-0}                     # 0 = replay the full episode; cap it for a quick demo on longer episodes
MAX_EPISODES=${MAX_EPISODES:-1}                 # episodes to replay per dataset
TRIM_STATIC=${TRIM_STATIC:-0}       # 1 = force-drop the leading near-static frames (default: per-dataset clap.eval.dataset_specs, off for all 5 sample datasets)
SKIP_FRAMES=${SKIP_FRAMES:-}        # force-skip this many leading frames unconditionally (default: per-dataset spec); empty = don't override

EXTRA_ARGS=()
[[ "$TRIM_STATIC" == "1" ]] && EXTRA_ARGS+=(--trim-static-prefix)
[[ -n "$SKIP_FRAMES" ]] && EXTRA_ARGS+=(--skip-first-n-frames "$SKIP_FRAMES")

clap-rollout-replay \
    --config "$CONFIG" \
    --ckpt "$CKPT_PATH" \
    --family "$FAMILY" \
    --datasets "$DATASET" \
    --max-episodes-per-dataset "$MAX_EPISODES" \
    --num-inference-steps 25 \
    --max-chunks "$MAX_CHUNKS" \
    --save-dir "eval_outputs/getting_started_replay/${RUN_TAG}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done -- see eval_outputs/getting_started_replay/${RUN_TAG}/{video,info}/"
