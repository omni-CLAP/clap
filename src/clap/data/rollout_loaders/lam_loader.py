"""Full-episode loader for LAM rollout replay.

Genuinely different from `EEEpisodeLoader`: episodes are discovered by
walking the LAM latent-action tree (not annotation JSON), and video paths
have to be resolved against a separate, sometimes differently-padded,
episode-naming convention in the video tree.
"""

import logging
import os

import numpy as np
from decord import VideoReader, cpu

from clap.data.base import maybe_flip_bgr
from clap.data.camera_stacking import STACKERS, frames_to_video_tensor
from clap.data.lam import _normalize_ep_key, build_lam_text_map
from clap.data.oxe_catalog import OXE_LAM_LAYOUT
from clap.eval.episode_eligibility import chunk_eligible, compute_data_lengths
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _candidate_video_paths(rel, video_root, layout):
    """Camera file path(s) for one episode, given its video_layout ('flat_mp4' | 'folder_single' | 'folder_stacked')."""
    if layout["video_layout"] == "flat_mp4":
        return [os.path.join(video_root, rel + ".mp4")]  # one mp4 per episode, no per-camera subfolder
    if layout["video_layout"] == "folder_single":
        return [os.path.join(video_root, rel, f"{layout.get('cam_name', '0')}.mp4")]
    # folder_stacked: 2 or 3 camera files under the episode's own subfolder.
    paths = [os.path.join(video_root, rel, f"{layout['left_cam']}.mp4"),
             os.path.join(video_root, rel, f"{layout['right_cam']}.mp4")]
    if layout.get("stacking_mode") == "three_view":
        paths.append(os.path.join(video_root, rel, f"{layout['wrist_cam']}.mp4"))
    return paths


def _video_paths_for(rel, video_root, layout):
    """Try `rel` as-is first, then its normalized form — the LAM tree and video tree
    sometimes disagree on zero-padding for the same episode."""
    for candidate in dict.fromkeys([rel, _normalize_ep_key(rel)]):  # de-dupe while preserving order
        paths = _candidate_video_paths(candidate, video_root, layout)
        if all(os.path.isfile(p) for p in paths):
            return paths
    return paths  # neither candidate fully resolved; return the last attempt so the caller's check fails informatively


class LAMEpisodeLoader:
    """Walks one LAM dataset's latent-action tree and returns full-episode tensors on demand.

    Args:
        num_history / num_frames: If both given (and nonzero), enables the
            cross-family eligibility filter; LAM-only datasets (egodex,
            language_table have no EE/language annotation tree, so this is
            left disabled for them by the caller — the per-chunk length guard
            inside `clap.rollout.agent` still protects correctness either way.
    """

    def __init__(
        self, dataset_name, oxe_base_path, oxe_lam_root, oxe_lam_subdir,
        video_size=(576, 320), split="val", num_history=0, num_frames=0,
        egodex_lam_subdir=None, lam_subdir_override=None, strict_eligibility=False,
    ):
        self.dataset_name = dataset_name
        self.layout = OXE_LAM_LAYOUT.get(dataset_name, {})  # per-dataset video/LAM layout config
        self.video_size = list(video_size)
        self.split = split
        self.oxe_base_path = oxe_base_path
        self.oxe_lam_root = oxe_lam_root

        # Same subdir-resolution precedence as the training-time LAMDataset.
        if lam_subdir_override is not None and self.layout.get("lam_subdir") is not None:
            effective_subdir = lam_subdir_override  # explicit override wins
        elif dataset_name == "egodex" and egodex_lam_subdir is not None:
            effective_subdir = egodex_lam_subdir  # egodex-specific subdir (different extractor run)
        else:
            effective_subdir = self.layout.get("lam_subdir", oxe_lam_subdir)  # per-dataset default, else the global default
        self.oxe_lam_subdir = effective_subdir

        self.video_root = os.path.join(oxe_base_path, dataset_name, "videos", split)
        self.lam_dir = os.path.join(oxe_lam_root, dataset_name, effective_subdir, split)
        self.fps_downsample_ratio = self.layout.get("fps_downsample_ratio", 1)
        self._T_required = (num_history + num_frames) if (strict_eligibility and num_history and num_frames) else 0

        self.text_map = build_lam_text_map(dataset_name, oxe_base_path, os.path.join(oxe_base_path, dataset_name), split)  # {episode_key: caption}
        self.episodes = self._enumerate_episodes()
        n_with_text = sum(1 for ep in self.episodes if self.text_map.get(_normalize_ep_key(ep["rel"]), ""))  # for logging coverage
        logger.info(f"📊 [{dataset_name}/{split}] {len(self.episodes)} episodes ({n_with_text} with text) "
                    f"(strict_eligibility={'on' if self._T_required else 'off'})")

    def _enumerate_episodes(self):
        if not os.path.isdir(self.lam_dir):
            logger.warning(f"⚠️ LAM dir missing: {self.lam_dir}")
            return []  # no LAM latents for this dataset/split
        episodes = []
        n_rejected = 0
        for root, _dirs, files in os.walk(self.lam_dir):
            if "latent_actions.npy" not in files:
                continue  # this directory isn't an episode's LAM folder
            rel = os.path.relpath(root, self.lam_dir)
            npy_path = os.path.join(root, "latent_actions.npy")
            try:
                num_latents = int(np.load(npy_path, mmap_mode="r").shape[0])  # mmap: just read the header
            except Exception:
                continue  # unreadable/corrupt latent file -> skip this episode
            if num_latents <= 0:
                continue  # empty latent array -> no usable frames

            video_paths = _video_paths_for(rel, self.video_root, self.layout)
            if not all(os.path.isfile(p) for p in video_paths):
                continue  # no matching video files under either naming convention

            if self._T_required:
                ep_id = _normalize_ep_key(rel)  # canonical form, matches EE/language enumerators and test-set JSONs
                lengths = compute_data_lengths(
                    self.dataset_name, ep_id, self.oxe_base_path,
                    self.oxe_lam_root, self.oxe_lam_subdir, self.split,
                )
                if not chunk_eligible(lengths, self.fps_downsample_ratio, self._T_required):
                    n_rejected += 1
                    continue

            episodes.append({"rel": rel, "npy_path": npy_path, "video_paths": video_paths, "num_latents": num_latents})

        episodes.sort(key=lambda e: e["rel"])  # deterministic ordering for reproducible episode_indices
        max_episodes = self.layout.get("max_episodes")
        if max_episodes is not None and len(episodes) > max_episodes:
            episodes = episodes[:max_episodes]  # e.g. egodex's train/val split is a tail-cut, not a separate tree
        if self._T_required:
            logger.info(f"🔀 {n_rejected} eps rejected by cross-family eligibility "
                        f"(T_required={self._T_required}, d={self.fps_downsample_ratio})")
        return episodes

    def __len__(self):
        return len(self.episodes)

    def load(self, idx):
        """Load one full episode: stacked video, per-frame LAM action, text."""
        ep = self.episodes[idx]
        d = self.fps_downsample_ratio
        readers = [VideoReader(p, ctx=cpu(0)) for p in ep["video_paths"]]
        n_video = min(len(vr) for vr in readers)
        frame_ids = np.arange(0, n_video, d)  # downsample to this dataset's native replay rate
        n_use = min(len(frame_ids), ep["num_latents"])  # LAM is per-frame and may be shorter by 1
        frame_ids = frame_ids[:n_use]

        # Load and stack each camera's frames at the same downsampled indices.
        if self.layout.get("video_layout") in ("flat_mp4", "folder_single"):
            cam_id = self.layout.get("cam_name", "0")
            frames = maybe_flip_bgr(readers[0].get_batch(frame_ids).asnumpy(), self.dataset_name, cam_id)
            frames = STACKERS[None](frames).astype(np.uint8)  # single camera, tiled into 3 slots
        else:
            left = maybe_flip_bgr(readers[0].get_batch(frame_ids).asnumpy(), self.dataset_name, self.layout["left_cam"])
            right = maybe_flip_bgr(readers[1].get_batch(frame_ids).asnumpy(), self.dataset_name, self.layout["right_cam"])
            if self.layout.get("stacking_mode") == "three_view":
                wrist = maybe_flip_bgr(readers[2].get_batch(frame_ids).asnumpy(), self.dataset_name, self.layout["wrist_cam"])
                frames = STACKERS["three_view"](right, left, wrist).astype(np.uint8)
            else:
                frames = STACKERS["two_view"](right, left).astype(np.uint8)
        video = frames_to_video_tensor(frames, self.video_size)  # (C, T, H, W) uint8

        arr = np.load(ep["npy_path"])  # (T_full, action_dim) — full load this time, not mmap, since we read every row
        actions = np.zeros((n_use, arr.shape[1]), dtype=np.float32)
        last = arr.shape[0] - 1
        # Most datasets extract one latent per raw frame (latent_skip=1: latent for
        # raw frame `fid` is arr[fid]). A few extract latents at a coarser stride
        # (e.g. rgb_skip=3), recorded as "latent_skip" so frame_ids (already stepping
        # in raw-frame units of that stride) map to the right row via integer division.
        latent_skip = self.layout.get("latent_skip", d)
        for k, fid in enumerate(frame_ids):
            actions[k] = arr[min(fid // latent_skip, last)]

        text = self.text_map.get(_normalize_ep_key(ep["rel"]), "")
        # Only folder_stacked layouts have a real multi-camera stacking_mode;
        # flat_mp4/folder_single are single-camera, encoded as None (-> tiled).
        stacking_mode_used = self.layout.get("stacking_mode") if self.layout.get("video_layout") == "folder_stacked" else None

        return {
            "video": video,
            "actions": actions,
            "frame_ids": frame_ids,
            "rel": ep["rel"],
            "dataset_name": self.dataset_name,
            "text": text,
            "stacking_mode_used": stacking_mode_used,
        }
