"""Preprocess Unitree G1 BrainCo LeRobot-v3 HF datasets into the `g1_humanoid` embodiment layout.

For each `--datasets` entry (a `unitreerobotics/<name>` HF repo):
  - Downloads to the HF cache if not already present.
  - Extracts every `src_fps // out_fps`-th frame per camera.
  - Resizes each frame to `--out-h`x`--out-w`.
  - Re-encodes per-episode per-camera mp4s (libx264, crf=18, `--out-fps`).
  - Writes annotation JSONs (`episode_id`, `texts`, `state`) matching what
    `clap.data.ee.EEDataset` reads for `dataset_name="g1_humanoid"`.
  - After all datasets: writes `<meta-out>/g1_humanoid/stat.json` (p01/p99 over
    the raw 26-dim joint action, via `clap.preprocess.oxe_meta.compute_percentile_stats`).

Output layout matches every other embodiment (`clap.data.base.EmbodimentDataset`):
  <out-base>/g1_humanoid/
    annotation/{train,val}/<global_ep_id>.json
    videos/{train,val}/<global_ep_id>/{0,1,2,3}.mp4   # right_high, left_high, right_wrist, left_wrist

Global episode ID: dataset_index * 10000 + local_episode_id (avoids collisions across sub-datasets).

Usage:
  clap-preprocess-g1 --out-base $CLAP_OXE_BASE_PATH --meta-out dataset_meta_info \\
      [--dataset-index 0]  # process only this one HF dataset (for SLURM array jobs)
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

from clap.preprocess.oxe_meta import compute_percentile_stats
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

HF_ORG = "unitreerobotics"

DEFAULT_DATASETS = [
    "G1_Brainco_GraspOreo_Dataset", "G1_Brainco_GraspRubiksCube_Dataset", "G1_Brainco_PickApple_Dataset",
    "G1_Brainco_PickCharger_Dataset", "G1_Brainco_PickDoll_Dataset", "G1_Brainco_PickDrink_Dataset",
    "G1_Brainco_PickTissues_Dataset", "G1_Brainco_PickToothpaste_Dataset",
]

# Camera ordering in the output (0-3 -> cam_id in videos/<split>/<ep>/<cam_id>.mp4).
CAM_KEYS = [
    "observation.images.cam_right_high", "observation.images.cam_left_high",
    "observation.images.cam_right_wrist", "observation.images.cam_left_wrist",
]

_EMBODIMENT_NAME = "g1_humanoid"
_video_shape_cache: Dict[str, Tuple[int, int]] = {}


def load_parquet_dir(directory: str) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("chunk-*/file-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def extract_episode_frames(video_path: str, from_ts: float, ep_length: int, src_fps: int, stride: int) -> np.ndarray:
    """Decode every `stride`-th frame of one episode from an mp4 via ffmpeg. Returns (T, H, W, 3) uint8."""
    if video_path not in _video_shape_cache:
        # Only need to probe each source video's frame size once; cache across episodes.
        probe = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path,
        ]).decode().strip()
        w, h = map(int, probe.split(","))
        _video_shape_cache[video_path] = (h, w)
    H, W = _video_shape_cache[video_path]

    duration = ep_length / src_fps
    cmd = [
        # Seek to from_ts, decode only `duration` seconds, keep every `stride`-th
        # frame, and stream raw rgb24 bytes out over stdout instead of writing a file.
        "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{from_ts:.6f}", "-i", video_path, "-t", f"{duration:.6f}",
        "-vf", f"select='not(mod(n,{stride}))'", "-vsync", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {video_path}:\n{proc.stderr.decode()}")
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)  # flat uint8 buffer of all decoded frames
    n_frames = len(raw) // (H * W * 3)
    if n_frames == 0:
        raise RuntimeError(f"No frames decoded from {video_path} at ts={from_ts:.3f}s")
    return raw[: n_frames * H * W * 3].reshape(n_frames, H, W, 3)  # (T, H, W, 3), drop any trailing partial frame


def resize_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.stack([cv2.resize(frames[t], (w, h), interpolation=cv2.INTER_LINEAR) for t in range(frames.shape[0])])


def write_mp4_ffmpeg(frames: np.ndarray, out_path: str, fps: int, crf: int = 18):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    T, H, W, C = frames.shape  # (T, H, W, 3) rgb24 frames
    cmd = [
        # Read raw rgb24 frames from stdin, re-encode as libx264/yuv420p mp4.
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "pipe:",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "fast", "-pix_fmt", "yuv420p", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())  # stream the whole clip in as one write
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def process_dataset(
    ds_idx: int, ds_name: str, out_root: str, hf_cache_dir: Optional[str],
    val_episodes: int, out_h: int, out_w: int, out_fps: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Process one HF dataset end to end. Returns (train_actions, val_actions) for stat computation."""
    logger.info(f"🎬 [{ds_idx + 1}] {ds_name}")
    local_dir = snapshot_download(f"{HF_ORG}/{ds_name}", repo_type="dataset", cache_dir=hf_cache_dir)  # downloads to HF cache if missing
    logger.info(f"  local path: {local_dir}")

    with open(os.path.join(local_dir, "meta", "info.json")) as f:
        info = json.load(f)
    src_fps = int(info["fps"])
    stride = src_fps // out_fps  # keep every `stride`-th source frame to hit out_fps
    assert stride >= 1, f"src_fps={src_fps} must be >= out_fps={out_fps}"

    # Episode metadata (task text, per-camera video chunk/file locations) and
    # per-frame robot state, loaded from the HF dataset's parquet shards.
    episodes_df = load_parquet_dir(os.path.join(local_dir, "meta", "episodes")).sort_values("episode_index").reset_index(drop=True)
    data_df = load_parquet_dir(os.path.join(local_dir, "data")).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    ep_actions = {int(ep_id): np.stack(grp["action"].tolist()).astype(np.float32) for ep_id, grp in data_df.groupby("episode_index")}

    val_ep_ids = set(episodes_df["episode_index"].tolist()[-val_episodes:])  # last N episodes -> val split
    train_actions_list: List[np.ndarray] = []
    val_actions_list: List[np.ndarray] = []

    for _, ep_row in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc=f"  {ds_name}"):
        ep_id, ep_len = int(ep_row["episode_index"]), int(ep_row["length"])
        tasks = ep_row["tasks"]
        if len(tasks) == 0:
            raise ValueError(f"Episode {ep_id} in {ds_name} has no task text")
        text = str(tasks[0])  # use the first task string as the episode's caption

        split = "val" if ep_id in val_ep_ids else "train"
        global_id = ds_idx * 10000 + ep_id  # avoid episode-id collisions across sub-datasets
        ann_dir = os.path.join(out_root, _EMBODIMENT_NAME, "annotation", split)
        video_dir = os.path.join(out_root, _EMBODIMENT_NAME, "videos", split, str(global_id))
        ann_path = os.path.join(ann_dir, f"{global_id}.json")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        if os.path.isfile(ann_path) and all(os.path.isfile(os.path.join(video_dir, f"{c}.mp4")) for c in range(len(CAM_KEYS))):
            # Episode already fully written by a previous run; skip re-encoding
            # but still need its (subsampled) actions for the global stat computation.
            actions_10 = ep_actions[ep_id][::stride]
            (train_actions_list if split == "train" else val_actions_list).append(actions_10)
            continue

        cam_frame_counts = []
        for cam_out_id, cam_key in enumerate(CAM_KEYS):
            prefix = f"videos/{cam_key}"
            chunk_idx, file_idx = int(ep_row[f"{prefix}/chunk_index"]), int(ep_row[f"{prefix}/file_index"])
            from_ts = float(ep_row[f"{prefix}/from_timestamp"])
            video_path = os.path.join(local_dir, "videos", cam_key, f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.mp4")

            # Decode, resize, and re-encode this camera's clip for the episode.
            frames = extract_episode_frames(video_path, from_ts, ep_len, src_fps, stride)
            frames = resize_frames(frames, out_h, out_w)
            write_mp4_ffmpeg(frames, os.path.join(video_dir, f"{cam_out_id}.mp4"), fps=out_fps)
            cam_frame_counts.append(len(frames))

        T = min(cam_frame_counts)  # use the shortest camera's frame count for safety
        actions_10 = ep_actions[ep_id][::stride][:T]  # subsample + truncate actions to match video length

        ann = {"episode_id": global_id, "texts": [text], "state": actions_10.tolist()}
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        (train_actions_list if split == "train" else val_actions_list).append(actions_10)

    return train_actions_list, val_actions_list


def cli():
    """Parse command-line arguments for `clap-preprocess-g1`."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-base", required=True, help="Root of the OXE-format dataset tree (CLAP_OXE_BASE_PATH).")
    p.add_argument("--meta-out", default="dataset_meta_info", help="Where to write stat.json.")
    p.add_argument("--hf-cache-dir", default=None, help="HuggingFace cache dir (default: $HF_HOME).")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="unitreerobotics/<name> HF dataset repo names.")
    p.add_argument("--dataset-index", type=int, default=None, help="Process only this index (0-based) into --datasets; for SLURM array jobs.")
    p.add_argument("--val-episodes", type=int, default=13, help="Last N episodes per sub-dataset held out for val.")
    p.add_argument("--out-h", type=int, default=192)
    p.add_argument("--out-w", type=int, default=320)
    p.add_argument("--out-fps", type=int, default=10)
    return p.parse_args()


def main():
    args = cli()
    hf_cache = args.hf_cache_dir or os.environ.get("HF_HOME")
    # --dataset-index restricts this run to a single sub-dataset (for SLURM array jobs);
    # otherwise process every dataset in --datasets.
    indices = [args.dataset_index] if args.dataset_index is not None else list(range(len(args.datasets)))

    all_train_actions: List[np.ndarray] = []
    all_val_actions: List[np.ndarray] = []
    for ds_idx in indices:
        train_acts, val_acts = process_dataset(
            ds_idx, args.datasets[ds_idx], args.out_base, hf_cache,
            args.val_episodes, args.out_h, args.out_w, args.out_fps,
        )
        all_train_actions.extend(train_acts)
        all_val_actions.extend(val_acts)

    if args.dataset_index is None and all_train_actions:
        # Only compute the global stat.json when this run covered every dataset
        # (a single-index SLURM-array run doesn't see the full action distribution).
        stat = compute_percentile_stats(all_train_actions)
        stat_dir = os.path.join(args.meta_out, _EMBODIMENT_NAME)
        os.makedirs(stat_dir, exist_ok=True)
        stat_path = os.path.join(stat_dir, "stat.json")
        with open(stat_path, "w") as f:
            json.dump(stat, f, indent=2)
        logger.info(f"💾 Wrote stat.json -> {stat_path}")
    elif args.dataset_index is not None:
        logger.info(f"✅ Dataset index {args.dataset_index} done. Run without --dataset-index to compute the global stat.json.")


if __name__ == "__main__":
    main()
