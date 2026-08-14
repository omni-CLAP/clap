#!/bin/bash
# Pre-fetch an openpi checkpoint using openpi's own environment (optional_dependencies/openpi/) --
# doesn't construct a policy or need a GPU, just resolves + downloads the gs://
# checkpoint via openpi.shared.download, same as OpenPIPolicy does lazily on first use.
#
# Usage:
#   bash examples/getting_started/download_openpi_checkpoint.sh              # pi05_droid
#   bash examples/getting_started/download_openpi_checkpoint.sh pi0_droid
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OPENPI_DIR=optional_dependencies/openpi
if [[ ! -d "$OPENPI_DIR" ]]; then
    echo "Error: $OPENPI_DIR not found -- clone openpi there first (see the README's openpi install section)."
    exit 1
fi

CKPT_NAME=${1:-pi05_droid}
SCRIPT_PATH="$(pwd)/examples/getting_started/download_openpi_checkpoint.py"

cd "$OPENPI_DIR"
uv run python "$SCRIPT_PATH" "$CKPT_NAME"