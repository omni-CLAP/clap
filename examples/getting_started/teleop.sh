#!/bin/bash
# Getting-started demo: keyboard-driven world-model teleop, seeded from this package's
# shipped sample episodes (sample_data/oxe/<dataset>). Each key nudges the tracked pose
# and asks the model to imagine the next frame -- useful for qualitatively probing a
# checkpoint's action-following behavior without a robot. Downloads the checkpoint from
# HF on first run (see the README's checkpoint table for every published CKPT_NAME).
#
# Works against droid/bridge/taco_play (7-dim EE cartesian) as well as bimanual_yam
# (14-dim joint-space, 2 arms) and g1_humanoid (26-dim joint-space, 2 arms + 2 hands) --
# see clap.rollout.teleop_controls's module docstring for exactly what each dataset's
# keys drive. Note keyboard_control's workspace safety bounds (_X_RANGE/_Y_RANGE/
# _Z_RANGE in clap.rollout.teleop_controls) only apply to the 7-dim cartesian pose
# (droid/bridge/taco_play) -- bimanual_yam/g1_humanoid have no established safety box here.
#
# bimanual_yam/g1_humanoid are novel-embodiment adaptation targets and only work with
# their own matching adapt_* checkpoint (not the generic cross-embodiment default) --
# DATASET=bimanual_yam/g1_humanoid below auto-selects it, unless CKPT_NAME is set
# explicitly (that always wins).
#
# Interactive by default (reads keys live from your terminal, Ctrl-C to quit) -- set KEYS to
# instead replay a fixed scripted sequence non-interactively.
#
# Key vocabulary (clap.rollout.teleop_controls.KEY_HELP): droid/bridge/taco_play get
# w/a/s/d/z/x (x,y,z) + e/r,o/p,t/y (roll,pitch,yaw) + c/v (gripper). bimanual_yam/
# g1_humanoid get q/a,w/s,e/d,r/f,t/g,y/h(,u/j) per-joint keys targeting whichever
# arm/hand is active -- Tab cycles the active target, Space toggles dual mode (mirrors
# onto that target's left/right counterpart too).
#
# Also starts a live-preview server (clap.rollout.live_viewer) by default -- it prints a
# http://localhost:<port>/teleop_viewer.html URL to open in a browser, showing each predicted
# frame as it's generated instead of only the final .mp4 after the run ends, including the
# current active-target/dual-mode state for bimanual_yam/g1_humanoid. Works the same
# locally or over SSH/SLURM (port-forward both printed ports first in the remote case).
#
# Runs with --num-inference-steps 25 (vs. the real default of 50 -- see
# `clap-teleop --help`) to keep each step's prediction quick for interactive use.
#
# Usage: bash examples/getting_started/teleop.sh
#        DATASET=bridge bash examples/getting_started/teleop.sh
#        EPISODE=7099 bash examples/getting_started/teleop.sh
#        DATASET=bimanual_yam bash examples/getting_started/teleop.sh   # CKPT_NAME auto-set to adapt_bimanual_yam
#        DATASET=g1_humanoid bash examples/getting_started/teleop.sh    # CKPT_NAME auto-set to adapt_g1_humanoid
#        CKPT_NAME=my_own_ckpt DATASET=bimanual_yam bash examples/getting_started/teleop.sh   # explicit CKPT_NAME always wins over the auto-default above
#        KEYS=wwaaz bash examples/getting_started/teleop.sh          # scripted, non-interactive
#        NO_LIVE_VIEW=1 bash examples/getting_started/teleop.sh      # skip the live-preview server entirely
#        HISTORY_IDX="0 0 -12 -9 -6 -3" bash examples/getting_started/teleop.sh   # sparse history offsets instead of the last 6 contiguous frames
set -euo pipefail

DATASET=${DATASET:-droid}         # droid | bridge | taco_play | bimanual_yam | g1_humanoid
case "$DATASET" in                # each dataset's shipped sample episodes differ, so the default EPISODE does too
    bridge)       EPISODE=${EPISODE:-10} ;;    # 10 | 1002 | 1003
    taco_play)    EPISODE=${EPISODE:-1002} ;;  # 1002 | 1003 | 1010
    bimanual_yam) EPISODE=${EPISODE:-989} ;;   # 989 | 21000 | 26579
    g1_humanoid)  EPISODE=${EPISODE:-196} ;;   # 196 | 10196 | 20199
    *)            EPISODE=${EPISODE:-7099} ;;  # droid: 2799 | 7099 | 9199
esac

# bimanual_yam/g1_humanoid only work with their own matching adapt_* checkpoint (14-/
# 26-dim joint-space action_encoder, reset from the 7-dim EE cross-embodiment default) --
# default CKPT_NAME to it here, before _common.sh resolves CKPT_NAME/CKPT_PATH/CONFIG
# below, but only if the caller hasn't already set CKPT_NAME explicitly (that always wins).
if [[ -z "${CKPT_NAME:-}" ]]; then
    case "$DATASET" in
        bimanual_yam) export CKPT_NAME=adapt_bimanual_yam ;;
        g1_humanoid)  export CKPT_NAME=adapt_g1_humanoid ;;
    esac
fi

source "$(dirname "$0")/_common.sh"

KEYS=${KEYS:-}                    # empty (default) = live interactive session; set to e.g. wwaaxxcv for a scripted non-interactive run
LIVE_VIEW_WS_PORT=${LIVE_VIEW_WS_PORT:-8865}
LIVE_VIEW_HTTP_PORT=${LIVE_VIEW_HTTP_PORT:-8866}
LIVE_VIEW_FPS=${LIVE_VIEW_FPS:-12}
NO_LIVE_VIEW=${NO_LIVE_VIEW:-0}   # 1 = don't start the live-preview server
HISTORY_IDX=${HISTORY_IDX:-"0 0 -32 -24 -8 -2"}      # space-separated sparse history offsets (e.g. "0 0 -8 -6 -4 -2"); empty = last num_history contiguous frames

EXTRA_ARGS=()
[[ "$NO_LIVE_VIEW" == "1" ]] && EXTRA_ARGS+=(--no-live-view)
[[ -n "$KEYS" ]] && EXTRA_ARGS+=(--keys "$KEYS")
[[ -n "$HISTORY_IDX" ]] && EXTRA_ARGS+=(--history-idx $HISTORY_IDX)

clap-teleop \
    --config "$CONFIG" \
    --ckpt "$CKPT_PATH" \
    --family ee \
    --dataset "$DATASET" \
    --episode "$EPISODE" \
    --live-view-ws-port "$LIVE_VIEW_WS_PORT" \
    --live-view-http-port "$LIVE_VIEW_HTTP_PORT" \
    --live-view-fps "$LIVE_VIEW_FPS" \
    --ckpt-name "$CKPT_DISPLAY_NAME" \
    --num-inference-steps 25 \
    --save-dir "eval_outputs/getting_started_teleop/${RUN_TAG}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done -- see eval_outputs/getting_started_teleop/${RUN_TAG}/"