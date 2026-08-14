#!/bin/bash
# Real-robot (or simulated) policy-in-the-loop deployment with a world-model preview.
# Needs $CONFIG (the world model TrainingRunConfig YAML), $DEPLOY_CONFIG (a
# RolloutDeployConfig-shaped YAML - see examples/getting_started/deploy_config.yaml
# with task_name/val_dataset_dir/dataset_name/episode_ids/
# start_idx/instructions/policy_type/policy_ckpt/adapter_ckpt), and $CKPT.
# Usage: CONFIG=configs/experiment/cross_embodiment_oxe_ee.yaml \
#        DEPLOY_CONFIG=configs/experiment/deploy_pickplace.yaml \
#        CKPT=model_ckpt/cross_embodiment_oxe_ee/last.pt \
#        bash examples/rollout/deploy_policy_in_the_loop.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${CONFIG:?Set CONFIG to the world model TrainingRunConfig YAML}"
: "${DEPLOY_CONFIG:?Set DEPLOY_CONFIG to a RolloutDeployConfig-shaped YAML}"
: "${CKPT:?Set CKPT to the CLAPModel checkpoint path}"

clap-rollout-deploy \
    --config "${CONFIG}" \
    --deploy-config "${DEPLOY_CONFIG}" \
    --ckpt "${CKPT}" \
    "$@"