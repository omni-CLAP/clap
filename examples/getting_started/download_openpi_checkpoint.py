"""Download an openpi checkpoint without constructing a policy (no torch/model load).

Run with openpi's own environment (not clap's) -- e.g. from optional_dependencies/openpi/:
    uv run python /path/to/this/download_openpi_checkpoint.py pi05_droid

See download_openpi_checkpoint.sh for a wrapper that handles the cd + uv run for you.
"""

import sys

from openpi.shared import download


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pi05_droid"  # matches deploy_config.yaml's policy_ckpt convention
    gs_path = f"gs://openpi-assets/checkpoints/{name}"
    checkpoint_dir = download.maybe_download(gs_path)  # no-op if already cached under OPENPI_DATA_HOME/~/.cache/openpi
    print(f"{name} -> {checkpoint_dir}")


if __name__ == "__main__":
    main()