"""Full-episode loader for EE-cartesian / joint-space / language rollout replay.

Covers ee/bridge/bimanual_yam/g1_humanoid/language families — they all read
the same annotation-JSON + per-camera-mp4 layout and differ only in which
`EmbodimentConfig` is selected (per-frame caption building for "language"
happens downstream in `clap.rollout.agent`, using the same states/grippers/text
this loader returns).
"""

import json
import logging
import os

import numpy as np
from decord import VideoReader, cpu

from clap.data.base import maybe_flip_bgr
from clap.data.camera_stacking import STACKERS, frames_to_video_tensor
from clap.data.oxe_catalog import get_embodiment_config
from clap.eval.episode_eligibility import chunk_eligible, compute_data_lengths
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _extract_text(ann: dict) -> str:
    """An annotation JSON's recorded instruction: "texts" (list, use first) is the
    newer field; "text" (str) is the older one, kept as a fallback."""
    texts = ann.get("texts")
    return (texts[0] if isinstance(texts, list) and texts else ann.get("text", "")) or ""


def motion_onset(xyz, speed_frac, speed_floor):
    """First native frame where translation speed becomes sustained (for static-prefix trimming).

    Threshold = max(speed_floor, speed_frac * p90(speed)); onset is the first
    frame crossing it that stays active over a 5-frame window. Falls back to
    the single fastest frame if nothing sustains (a near-static episode).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) < 6:
        return 0  # too short to estimate a reliable onset
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)  # per-step translation speed
    threshold = max(speed_floor, speed_frac * float(np.percentile(speed, 90)))
    window = 5
    for t in range(len(speed) - window):
        # Require the threshold to hold over a short window, not just one noisy frame.
        if speed[t] >= threshold and speed[t:t + window].mean() >= threshold:
            return int(t)
    return int(np.argmax(speed))  # degenerate case: nothing sustains, use the single fastest frame


class EEEpisodeLoader:
    """Walks one dataset and returns full-episode tensors for autoregressive replay.

    Args:
        strict_eligibility: If True (with num_history/num_frames both set),
            drops episodes that don't have enough LAM/language/EE samples for
            one chunk — so the same episode set is eligible across every family.
    """

    def __init__(
        self, dataset_name, oxe_base_path, video_size=(576, 320), split="val",
        num_history=0, num_frames=0, oxe_lam_root="", oxe_lam_subdir="latent_actions",
        strict_eligibility=False, dataset_meta_info_path="dataset_meta_info",
    ):
        self.dataset_name = dataset_name
        self.config = get_embodiment_config(dataset_name)  # camera layout + action_mode for this dataset
        self.video_size = list(video_size)
        self.split = split
        self.fps_downsample_ratio = self.config.fps_downsample_ratio

        dataset_path = os.path.join(oxe_base_path, dataset_name)
        self.ann_dir = os.path.join(dataset_path, self.config.annotation_subdir, split)
        self.video_root = os.path.join(dataset_path, "videos", split)

        self.oxe_base_path = oxe_base_path
        self.oxe_lam_root = oxe_lam_root or oxe_base_path  # default LAM root to the main OXE tree if unset
        self.oxe_lam_subdir = oxe_lam_subdir
        # Only enforce cross-family eligibility when the caller actually cares
        # about chunk length (num_history/num_frames given) and asked for it.
        self._T_required = (num_history + num_frames) if (strict_eligibility and num_history and num_frames) else 0

        self.stat_path = os.path.join(dataset_meta_info_path, dataset_name, "stat.json")

        # Enumeration happens once up front so __len__/load(idx) are cheap random access afterward.
        self.episodes = self._enumerate_episodes()
        logger.info(f"📊 [{dataset_name}/{split}] {len(self.episodes)} episodes "
                    f"(strict_eligibility={'on' if self._T_required else 'off'})")

    def _resolve_ep_dir(self, base, episode_id):
        # Episode dirs are named either the raw id or a zero-padded 'episode_NNNNNN'; try both.
        key = str(episode_id)
        for cand in (key, f"episode_{int(episode_id):06d}"):  # try raw id, then zero-padded numeric form
            d = os.path.join(base, cand)
            if os.path.isdir(d):
                return d  # first existing candidate wins
        return os.path.join(base, key)  # neither exists; caller's isfile/isdir check will fail informatively

    def _slot_cams(self):
        """Camera id per vertical slot, in stacking order, for this dataset's stacking_mode."""
        cfg = self.config
        if cfg.stacking_mode == "four_view":
            return list(cfg.cam_ids)  # 4 distinct camera views
        if cfg.stacking_mode == "three_view":
            return [cfg.right_view_id, cfg.left_view_id, cfg.wrist_view_id]
        if cfg.stacking_mode == "two_view":
            return [cfg.right_view_id, cfg.left_view_id]
        return [cfg.cam_id]  # single camera

    def _video_paths_for(self, episode_id):
        ep_dir = self._resolve_ep_dir(self.video_root, episode_id)
        return [os.path.join(ep_dir, f"{cam}.mp4") for cam in self._slot_cams()]

    def _enumerate_episodes(self):
        if not os.path.isdir(self.ann_dir):
            logger.warning(f"⚠️ annotation dir missing: {self.ann_dir}")
            return []  # no annotations for this split -> no episodes
        ann_files = sorted(f for f in os.listdir(self.ann_dir) if f.endswith(".json"))
        episodes = []
        n_rejected = 0
        for fname in ann_files:
            ann_path = os.path.join(self.ann_dir, fname)
            # Read just enough of the annotation to validate + index this episode; full state
            # array is re-read in load() since not every enumerated episode gets loaded.
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                ep_id = ann.get("episode_id")
                n_state = len(ann.get("state", []))
                if ep_id is None or n_state == 0:
                    continue  # missing id or empty state -> not a usable episode
            except Exception:
                continue  # unreadable/corrupt annotation -> skip this episode

            ep_id = str(ep_id)
            video_paths = self._video_paths_for(ep_id)
            if not all(os.path.isfile(p) for p in video_paths):
                continue  # a camera file is missing; this episode can't be replayed

            if self._T_required:
                # Cross-family check: would this episode also have enough LAM/language samples?
                lengths = compute_data_lengths(
                    self.dataset_name, ep_id, self.oxe_base_path,
                    self.oxe_lam_root, self.oxe_lam_subdir, self.split,
                )
                if not chunk_eligible(lengths, self.fps_downsample_ratio, self._T_required):
                    n_rejected += 1
                    continue

            episodes.append({"ann_file": ann_path, "episode_id": ep_id, "n_state": n_state, "video_paths": video_paths})

        if self._T_required:
            logger.info(f"🔀 {n_rejected} eps rejected by cross-family eligibility "
                        f"(T_required={self._T_required}, d={self.fps_downsample_ratio})")
        return episodes

    def __len__(self):
        return len(self.episodes)

    def load_text(self, idx):
        """Just this episode's recorded instruction — reads the annotation JSON only,
        no video decode. Cheap enough to call once per episode when auto-discovering
        instructions (see `clap.rollout.deploy._resolve_episode_selection`)."""
        with open(self.episodes[idx]["ann_file"]) as f:
            return _extract_text(json.load(f))

    def load(self, idx):
        """Load one full episode: stacked video, per-frame state/gripper, text."""
        ep = self.episodes[idx]
        with open(ep["ann_file"]) as f:
            ann = json.load(f)

        d = self.fps_downsample_ratio
        readers = [VideoReader(p, ctx=cpu(0)) for p in ep["video_paths"]]  # one decord reader per camera
        # Cap at the shortest of: any camera's frame count, and the state array's length.
        n = min(min(len(vr) for vr in readers), ep["n_state"])
        native_ids = np.arange(0, n, d)  # downsample to this dataset's native replay rate

        # Load and stack each camera's frames at the same downsampled indices.
        slot_cams = self._slot_cams()
        views = [
            maybe_flip_bgr(reader.get_batch(native_ids).asnumpy(), self.dataset_name, cam)
            for reader, cam in zip(readers, slot_cams)
        ]
        stacker = STACKERS[self.config.stacking_mode] if len(views) > 1 else STACKERS[None]  # pick vertical-stack fn for this camera layout
        frames = (stacker(*views) if len(views) > 1 else stacker(views[0])).astype(np.uint8)
        video = frames_to_video_tensor(frames, self.video_size)  # (C, T, H, W) uint8

        # Slice state/gripper arrays at the same downsampled indices as the video.
        state_arr = np.array(ann["state"], dtype=np.float64)
        gripper_arr = np.array(ann.get("continuous_gripper_state", []), dtype=np.float64)
        states_sub = state_arr[native_ids]
        grippers_sub = gripper_arr[native_ids] if gripper_arr.size else None  # joint-space embodiments have none

        text = _extract_text(ann)

        return {
            "video": video,
            "states": states_sub,
            "grippers": grippers_sub,
            "text": text,
            "episode_id": ep["episode_id"],
            "dataset_name": self.dataset_name,
            "action_mode": self.config.action_mode,
            "n_use": len(native_ids),
            "stacking_mode_used": self.config.stacking_mode,
            "stat_path": self.stat_path,
        }
