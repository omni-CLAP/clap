"""Compute p01/p99 action-normalization stats for one OXE dataset: `clap-preprocess-oxe-meta`.

Reads annotation JSONs (`state[:, :6]` + `continuous_gripper_state` for "ee7"
datasets, or the raw `state` array as-is for joint-space ones) and writes
`<meta_info_path>/<dataset_name>/stat.json` in the format `EEDataset`'s
`BoundNormalizer` reads at training/eval time. Run once per dataset before
training on it.

Usage:
  clap-preprocess-oxe-meta --oxe-base-path $CLAP_OXE_BASE_PATH --dataset-name bridge

  # ee7 (cartesian) datasets: the 7 cross-embodiment sets + the 4
  # held-out generalization targets (see clap.data.oxe_catalog.OXE_CATALOG).
  for ds in bridge fractal bc_z fmb taco_play furniture_bench droid \\
            austin_sailor berkeley_autolab_ur5 stanford_hydra utaustin_mutex; do
      clap-preprocess-oxe-meta --oxe-base-path $CLAP_OXE_BASE_PATH --dataset-name $ds
  done

  # bimanual_yam: same OXE annotation-JSON layout, but joint-space (--full-state)
  # since its 14-dim action is the raw `state` array, not state[:6]+gripper.
  clap-preprocess-oxe-meta --oxe-base-path $CLAP_OXE_BASE_PATH --dataset-name bimanual_yam --full-state

  # g1_humanoid has its own annotation-generating pipeline (clap-preprocess-g1),
  # which writes its stat.json directly — it doesn't go through this script.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from clap.data.oxe_catalog import get_embodiment_config
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def compute_percentile_stats(all_arrays: List[np.ndarray]) -> dict:
    """p01/p99 of the concatenated (N_total, D) array, in the stat.json shape every embodiment shares."""
    stacked = np.concatenate(all_arrays, axis=0)
    return {"state_01": np.percentile(stacked, 1, axis=0).tolist(), "state_99": np.percentile(stacked, 99, axis=0).tolist()}


def _load_states_from_ann(ann_path: str, full_state: bool) -> Optional[np.ndarray]:
    """One episode's (T, D) per-frame action array, or None if the annotation is missing/malformed.

    full_state=False (the "ee7" convention): D=7, [x,y,z,roll,pitch,yaw,gripper],
    built from state[:, :6] + the separate continuous_gripper_state scalar.
    full_state=True: the raw `state` array as-is (e.g. 26-dim joint angles,
    where any gripper/hand dims are already included and there's no cartesian
    pose to slice out).
    """
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except Exception:
        return None  # missing/corrupt annotation file
    state = np.array(ann.get("state", []))
    if state.ndim != 2 or len(state) == 0:
        return None  # malformed or empty state array
    if full_state:
        return state.astype(np.float32)  # joint-space: use the raw state array as-is
    gripper = np.array(ann.get("continuous_gripper_state", []))
    if gripper.ndim != 1:
        return None  # gripper trace missing/malformed
    arm = state[:, :6]  # (T, 6) cartesian pose, drop any extra state dims
    g = gripper[:, None] if gripper.shape[0] == arm.shape[0] else np.zeros((len(arm), 1))  # (T, 1), fallback to zeros on length mismatch
    return np.concatenate([arm, g], axis=-1).astype(np.float32)  # (T, 7): arm pose + gripper


def compute_stats(oxe_base_path: str, dataset_name: str, n_workers: int = 32, full_state: bool = False) -> dict:
    # train split only -- val must stay unseen by anything derived from training data,
    # including the normalization bounds themselves.
    ann_subdir = get_embodiment_config(dataset_name).annotation_subdir  # per-embodiment annotation folder layout
    ann_dir = os.path.join(oxe_base_path, dataset_name, ann_subdir, "train")
    if not os.path.isdir(ann_dir):
        raise RuntimeError(f"[{dataset_name}] no train-split annotation dir at {ann_dir}")
    ann_files = [os.path.join(ann_dir, fn) for fn in os.listdir(ann_dir) if fn.endswith(".json")]
    logger.info(f"📊 [{dataset_name}/train] {len(ann_files)} annotation files")

    all_states: List[np.ndarray] = []
    # Parse annotation JSONs concurrently (I/O bound) across n_workers threads.
    with ThreadPoolExecutor(n_workers) as pool:
        futs = {pool.submit(_load_states_from_ann, ap, full_state): ap for ap in ann_files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{dataset_name}/train", leave=False):
            s = fut.result()
            if s is not None:  # skip episodes whose annotation failed to load
                all_states.append(s)

    if not all_states:
        raise RuntimeError(f"No valid train-split annotation data found for {dataset_name}")
    logger.info(f"📊 [{dataset_name}] total frames: {sum(len(s) for s in all_states)}")
    return compute_percentile_stats(all_states)


def cli():
    """Parse command-line arguments for `clap-preprocess-oxe-meta`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oxe-base-path", required=True, help="Root of OXE mp4 datasets.")
    p.add_argument("--dataset-name", required=True, help="Sub-dataset name (e.g. bridge, fractal, droid).")
    p.add_argument("--meta-info-path", default="dataset_meta_info", help="Where to write stat.json.")
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--full-state", action="store_true",
                   help="Use the raw 'state' array as-is (joint-space embodiments) instead of the ee7 convention.")
    return p.parse_args()


def main():
    args = cli()
    logger.info(f"🧮 Computing stats for {args.dataset_name} ...")
    stat = compute_stats(args.oxe_base_path, args.dataset_name, args.n_workers, args.full_state)

    out_dir = os.path.join(args.meta_info_path, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stat.json")
    with open(out_path, "w") as f:
        json.dump(stat, f, indent=2)  # {"state_01": [...], "state_99": [...]}
    logger.info(f"💾 Saved -> {out_path}")


if __name__ == "__main__":
    main()
