#!/bin/bash
# Getting-started demo: policy-in-the-loop deployment (simulation mode, no live
# robot) against this package's shipped sample bimanual_yam episodes, using
# examples/getting_started/deploy_config_yam.yaml (episode_ids/instructions left
# unset there -- auto-discovered from sample_data/oxe/bimanual_yam directly).
#
# bimanual_yam is joint-space (action_mode="joint14") -- no forward kinematics, no
# separate cartesian representation; only MolmoAct2's bimanual_yam checkpoint (server
# mode only) is wired up, no openpi (see this repo's README's bimanual_yam deploy section).
#
# Requires a running MolmoAct2 YAM server (see the README) and CKPT_NAME's matching
# adapt_bimanual_yam checkpoint, fetched automatically below if not already cached.
#
# Usage: bash examples/getting_started/deploy_yam.sh
#        NO_LIVE_VIEW=1 bash examples/getting_started/deploy_yam.sh      # skip the live-preview server entirely
#        LIVE_VIEW_WS_PORT=9765 LIVE_VIEW_HTTP_PORT=9766 bash examples/getting_started/deploy_yam.sh
#        POLICY_SERVER_OVERRIDE=host:port bash examples/getting_started/deploy_yam.sh  # force server mode against a different host
set -euo pipefail

# bimanual_yam only works with its own matching adapt_bimanual_yam checkpoint (14-dim
# joint-space action) -- auto-select it unless the caller already set CKPT_NAME, same
# pattern as teleop.sh's DATASET=bimanual_yam auto-default.
export CKPT_NAME=${CKPT_NAME:-adapt_bimanual_yam}

source "$(dirname "$0")/_common.sh"

# If policy_server ends up set and it's an internal cluster host, module proxy/default's
# HTTP(S) proxy (meant for reaching the public internet, e.g. HF/GCS) will otherwise
# intercept that connection and get a 403 from the proxy itself -- the proxy doesn't route
# intra-cluster traffic. Exclude that host from the proxy. Tracks whichever policy_server
# actually ends up in effect: POLICY_SERVER_OVERRIDE (if set) takes priority over
# deploy_config_yam.yaml's own value, matching the EXTRA_ARGS override passed below.
POLICY_SERVER=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('examples/getting_started/deploy_config_yam.yaml'))
print(cfg.get('policy_server') or '')
")
if [[ -n "${POLICY_SERVER_OVERRIDE:-}" ]]; then
    POLICY_SERVER="$POLICY_SERVER_OVERRIDE"
fi
if [[ -n "$POLICY_SERVER" ]]; then
    POLICY_HOST="${POLICY_SERVER%%:*}"
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${POLICY_HOST}"
    export no_proxy="${no_proxy:+$no_proxy,}${POLICY_HOST}"
fi

LIVE_VIEW_WS_PORT=${LIVE_VIEW_WS_PORT:-8765}
LIVE_VIEW_HTTP_PORT=${LIVE_VIEW_HTTP_PORT:-8766}
LIVE_VIEW_FPS=${LIVE_VIEW_FPS:-4}
NO_LIVE_VIEW=${NO_LIVE_VIEW:-0}   # 1 = don't start the live-preview server

EXTRA_ARGS=()
[[ "$NO_LIVE_VIEW" == "1" ]] && EXTRA_ARGS+=(--no-live-view)
[[ -n "${POLICY_SERVER_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--policy-server "$POLICY_SERVER_OVERRIDE")

clap-rollout-deploy \
    --config "$CONFIG" \
    --deploy-config examples/getting_started/deploy_config_yam.yaml \
    --ckpt "$CKPT_PATH" \
    --family ee \
    --num-inference-steps 25 \
    --live-view-ws-port "$LIVE_VIEW_WS_PORT" \
    --live-view-http-port "$LIVE_VIEW_HTTP_PORT" \
    --live-view-fps "$LIVE_VIEW_FPS" \
    --ckpt-name "$CKPT_DISPLAY_NAME" \
    --save-dir "eval_outputs/getting_started_deploy_yam/${RUN_TAG}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done -- see eval_outputs/getting_started_deploy_yam/${RUN_TAG}/"