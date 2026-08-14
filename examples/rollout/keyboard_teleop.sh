#!/bin/bash
# Interactive keyboard teleop against a recorded seed episode's first frame.
# Usage: DATASET=droid EPISODE=62099 KEYS=wwaaz \
#        CONFIG=configs/experiment/cross_embodiment_oxe_ee.yaml CKPT=model_ckpt/cross_embodiment_oxe_ee/last.pt \
#        bash examples/rollout/keyboard_teleop.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${CONFIG:?Set CONFIG to the world model TrainingRunConfig YAML}"
: "${CKPT:?Set CKPT to the CLAPModel checkpoint path}"
DATASET=${DATASET:-droid}
EPISODE=${EPISODE:?Set EPISODE to a recorded episode id to seed from}
KEYS=${KEYS:?Set KEYS to a sequence of teleop keys, e.g. wwaaz -- see clap.rollout.teleop_controls.KEY_HELP for the full map (droid/bridge/taco_play: w/s/a/d/z/x=forward/backward/left/right/up/down, e/r/o/p/t/y=roll/pitch/yaw, c/v=close/open; bimanual_yam/g1_humanoid: q/a..y/h(,u/j) per-joint keys, Tab=switch target, Space=toggle dual mode)}

clap-teleop \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --dataset "${DATASET}" \
    --episode "${EPISODE}" \
    --keys "${KEYS}" \
    "$@"