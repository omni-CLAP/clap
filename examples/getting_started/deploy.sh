#!/bin/bash
# Getting-started demo: policy-in-the-loop deployment (simulation mode, no live
# robot) against this package's shipped sample droid episodes, using
# examples/getting_started/deploy_config.yaml (episode_ids/instructions left
# unset there -- auto-discovered from sample_data/oxe/droid directly).
#
# Requires a real policy checkpoint: install openpi or MolmoAct2 (see the
# README) and fill in policy_ckpt (+ adapter_ckpt for openpi) in
# deploy_config.yaml first -- this will fail on import until you do.
#
# Usage: bash examples/getting_started/deploy.sh
#        NO_LIVE_VIEW=1 bash examples/getting_started/deploy.sh      # skip the live-preview server entirely
#        LIVE_VIEW_WS_PORT=9765 LIVE_VIEW_HTTP_PORT=9766 bash examples/getting_started/deploy.sh
#        IN_PROCESS=1 bash examples/getting_started/deploy.sh        # force in-process, overriding deploy_config.yaml's policy_server
#        POLICY_SERVER_OVERRIDE=host:port bash examples/getting_started/deploy.sh  # force server mode against a different host
#        POLICY_TYPE=molmoact2 bash examples/getting_started/deploy.sh  # override deploy_config.yaml's policy_type ("pi05"|"pi0"|"pi0fast"|"molmoact2")
set -euo pipefail
source "$(dirname "$0")/_common.sh"

IN_PROCESS=${IN_PROCESS:-0}   # 1 = force in-process mode, overriding deploy_config.yaml's policy_server

# If policy_server ends up set (server mode) and it's an internal cluster host, module
# proxy/default's HTTP(S) proxy (meant for reaching the public internet, e.g. HF/GCS)
# will otherwise intercept that connection and get a 403 from the proxy itself -- the
# proxy doesn't route intra-cluster traffic. Exclude that host from the proxy. Tracks
# whichever policy_server actually ends up in effect: IN_PROCESS/POLICY_SERVER_OVERRIDE
# (if set) take priority over deploy_config.yaml's own value, matching the EXTRA_ARGS
# overrides passed to clap-rollout-deploy below.
POLICY_SERVER=$(python3 -c "
import yaml
cfg = yaml.safe_load(open('examples/getting_started/deploy_config.yaml'))
print(cfg.get('policy_server') or '')
")
if [[ "$IN_PROCESS" == "1" ]]; then
    POLICY_SERVER=""
elif [[ -n "${POLICY_SERVER_OVERRIDE:-}" ]]; then
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
[[ "$IN_PROCESS" == "1" ]] && EXTRA_ARGS+=(--in-process)
[[ -n "${POLICY_SERVER_OVERRIDE:-}" ]] && EXTRA_ARGS+=(--policy-server "$POLICY_SERVER_OVERRIDE")
[[ -n "${POLICY_TYPE:-}" ]] && EXTRA_ARGS+=(--policy-type "$POLICY_TYPE")

clap-rollout-deploy \
    --config "$CONFIG" \
    --deploy-config examples/getting_started/deploy_config.yaml \
    --ckpt "$CKPT_PATH" \
    --family ee \
    --num-inference-steps 25 \
    --live-view-ws-port "$LIVE_VIEW_WS_PORT" \
    --live-view-http-port "$LIVE_VIEW_HTTP_PORT" \
    --live-view-fps "$LIVE_VIEW_FPS" \
    --ckpt-name "$CKPT_DISPLAY_NAME" \
    --save-dir "eval_outputs/getting_started_deploy/${RUN_TAG}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "Done -- see eval_outputs/getting_started_deploy/${RUN_TAG}/"
