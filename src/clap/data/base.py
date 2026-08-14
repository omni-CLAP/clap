"""Shared base class for embodiment dataset loaders.

`EmbodimentDataset` handles everything common across embodiments: p01/p99
normalization, history/future frame sampling, episode enumeration, per-camera
video/latent loading + stacking, and retrying on a bad sample instead of
crashing training. Each embodiment subclasses it and implements only
`_load_action`, since that's the one part that genuinely differs (EE-cartesian
deltas vs. LAM latents vs. raw joint state vs. captions).
"""

import json
import logging
import os
import random
import traceback
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from tqdm import tqdm

from clap.data.camera_stacking import STACKERS, frames_to_video_tensor
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

SVD_LATENT_SUBDIR = "latent_videos_svd"


def get_latent_video_subdir():
    """Latent-video subdir name, overridable via $CLAP_LATENT_VIDEO_SUBDIR.

    Lets a run point at a differently-named copy of the pre-encoded latents (e.g.
    one staged on faster local/scratch storage) instead of the default, without
    which every dataset falls back to raw video + on-the-fly VAE encoding for
    that subdir name -- a large per-step performance hit.
    """
    return os.environ.get("CLAP_LATENT_VIDEO_SUBDIR", SVD_LATENT_SUBDIR)


# Camera views confirmed to be stored with R/B channels swapped. cam_id keys are
# stringified so both int view-ids and str cam names match the same entry.
BGR_VIEWS = {
    "berkeley_autolab_ur5": {"0"},
}


def maybe_flip_bgr(frames, dataset_name, cam_id):
    """Swap R<->B channels for the specific camera views confirmed to need it."""
    if str(cam_id) in BGR_VIEWS.get(dataset_name, set()):
        return frames[..., ::-1]
    return frames


class VideoActionLengthMismatchError(RuntimeError):
    """Raised when too many episodes' video/latent frame counts disagree with their
    annotation's `state` array length -- see EmbodimentDataset._check_length_match.
    Propagates out of __getitem__ (not caught/retried like a one-off bad sample)."""


class _EpisodeLengthMismatch(RuntimeError):
    """One episode's video/latent frame count disagrees with its state length. Caught by
    __getitem__'s existing bad-sample handling -- the episode is skipped and a different
    random sample is retried, same as a corrupt file or any other per-episode failure."""


class BoundNormalizer:
    """p01/p99 -> [-1, 1] normalization, bounds loaded from a dataset's stat.json."""

    def __init__(self, stat_path):
        with open(stat_path) as f:  # load precomputed p01/p99 action-bound stats
            stat = json.load(f)
        self.data_min = np.array(stat["state_01"])[None, :]  # (1, D) p01 bound
        self.data_max = np.array(stat["state_99"])[None, :]  # (1, D) p99 bound

    def normalize(self, data, eps=1e-8):
        ndata = 2 * (data - self.data_min) / (self.data_max - self.data_min + eps) - 1
        return np.clip(ndata, -1, 1)

    def denormalize(self, data, eps=1e-8):
        return (data + 1) / 2 * (self.data_max - self.data_min + eps) + self.data_min


class TemporalSampler:
    """Builds the history+future frame-index window around an anchor frame.

    Future frames use a random stride (`skip`); history frames use a larger,
    independently-random stride (`skip_his`, occasionally 0 to include some
    near-anchor history), so the model sees varied time gaps at both ends.
    """

    def __init__(self, num_history, num_frames):
        self.num_history = num_history
        self.num_frames = num_frames

    def sample(self, frame_now, frame_len):
        skip = random.randint(1, 2)  # random future stride
        skip_his = int(skip * 4)  # history stride, coarser than the future stride
        if random.random() < 0.15:
            skip_his = 0  # occasionally collapse history to the anchor frame

        rgb_id = [int(frame_now - i * skip_his) for i in range(self.num_history, 0, -1)]  # history indices, oldest first
        rgb_id.append(frame_now)  # anchor/conditioning frame
        rgb_id += [int(frame_now + i * skip) for i in range(1, self.num_frames)]  # future frame indices
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()  # clamp to valid episode range
        return [int(x) for x in rgb_id]


class EmbodimentDataset(Dataset, ABC):
    """Base class for one embodiment/dataset's training samples.

    Directory layout expected under `<oxe_base_path>/<dataset_name>/`:
    `annotation*/<split>/*.json` (episode metadata + state), `videos/<split>/`
    (raw per-camera mp4s), `<latent_video_subdir>/<split>/` (pre-encoded SVD
    latents, preferred when present; subdir name defaults to "latent_videos_svd",
    overridable via `CLAP_LATENT_VIDEO_SUBDIR` -- see `get_latent_video_subdir`).

    Args:
        config: `EmbodimentConfig` — camera layout + action_mode for this dataset.
        stat_path: Path to the dataset's stat.json (p01/p99 action bounds), or
            None for embodiments that don't normalize (e.g. LAM).
    """

    def __init__(
        self,
        config,
        oxe_base_path,
        num_history,
        num_frames,
        video_size=(576, 320),
        mode="train",
        annotation_subdir="annotation",
        stat_path=None,
        debug=False,
    ):
        self.config = config
        self.dataset_name = config.name
        self.mode = mode
        self.split = "train" if mode == "train" else "val"  # annotation/video dirs use "val", not the mode name
        self.video_size = list(video_size)
        self.sampler = TemporalSampler(num_history, num_frames)  # builds history/future frame-index windows
        self.normalizer = BoundNormalizer(stat_path) if stat_path else None  # p01/p99 normalizer, if this embodiment uses one
        self._mismatched_episodes = set()  # episode_ids whose video/latent length disagrees with their state length

        dataset_path = os.path.join(oxe_base_path, config.name)
        self.ann_dir = os.path.join(dataset_path, annotation_subdir, self.split)
        self.video_root = os.path.join(dataset_path, "videos", self.split)
        self.latent_root = os.path.join(dataset_path, get_latent_video_subdir(), self.split)

        self.ann_files = sorted(
            os.path.join(self.ann_dir, fn) for fn in os.listdir(self.ann_dir) if fn.endswith(".json")
        )  # every episode's annotation JSON path, sorted for determinism
        if debug:
            self.ann_files = self.ann_files[:8]  # debug mode: only enumerate a handful of episodes
        self._num_episodes = len(self.ann_files)  # denominator for _check_length_match's ratio threshold

        self.samples = self._enumerate_samples()  # one training anchor per frame across all episodes
        logger.info(f"📊 {type(self).__name__} [{config.name}] ({mode}): "
                    f"{len(self.ann_files)} eps, {len(self.samples)} anchors")

    # ------------------------------------------------------------------
    # Episode enumeration — one training anchor per frame in every episode.
    # ------------------------------------------------------------------

    def _enumerate_one(self, ann_file):
        try:
            with open(ann_file) as f:
                ann = json.load(f)
        except Exception:
            return []  # unreadable/corrupt annotation -> episode contributes no samples
        n = len(ann.get("state", []))
        if n < 2:
            return []  # too short to sample a meaningful window from
        return [{"ann_file": ann_file, "episode_id": ann["episode_id"], "frame_id": idx} for idx in range(n)]  # one anchor per frame

    def _enumerate_samples(self):
        samples = []
        with ThreadPoolExecutor(16) as pool:  # parallelize per-episode JSON reads
            futs = {pool.submit(self._enumerate_one, ap): ap for ap in self.ann_files}
            for fut in tqdm(as_completed(futs), total=len(futs),
                             desc=f"enumerate {self.dataset_name} ({self.mode})", leave=False):
                samples.extend(fut.result())  # collect this episode's anchors as they finish
        samples.sort(key=lambda x: (x["ann_file"], x["frame_id"]))  # deterministic order regardless of completion order
        return samples

    def __len__(self):
        return len(self.samples)

    def _resolve_ep_dir(self, base, episode_id):
        """Episode dirs are named either the raw id or a zero-padded 'episode_NNNNNN'.

        Some datasets (e.g. egodex) use a non-numeric "task/id"-style episode_id,
        for which the zero-padded form doesn't apply — skip it rather than crash.
        """
        key = str(episode_id)
        candidates = [key]  # try the raw id first
        try:
            candidates.append(f"episode_{int(episode_id):06d}")  # also try the zero-padded numeric form
        except ValueError:
            pass  # non-numeric episode_id (e.g. egodex "task/id") -> only the raw form applies
        for cand in candidates:
            d = os.path.join(base, cand)
            if os.path.isdir(d):
                return d  # first existing candidate wins
        return os.path.join(base, key)  # neither candidate exists; caller's isfile/isdir check will fail informatively

    # ------------------------------------------------------------------
    # Video/action length consistency
    # ------------------------------------------------------------------

    # A single mismatched episode is treated like any other bad sample (logged, skipped,
    # retried elsewhere in this class) -- these thresholds are for escalating a *systemic*
    # problem (e.g. a preprocessing bug re-encoding videos at the wrong length) into a hard
    # error instead of letting it silently corrupt training via the min(i, n-1) clamps below.
    # Note: with DataLoader(num_workers>0), each worker process holds its own independent
    # copy of self._mismatched_episodes, so this threshold is evaluated per-worker, not
    # globally across the dataset -- a systemic problem will still eventually trip it in
    # every worker, just not necessarily on the very first mismatch observed process-wide.
    _LENGTH_MISMATCH_MIN_COUNT = 3   # never escalate on fewer than this many distinct bad episodes
    _LENGTH_MISMATCH_RATIO = 0.01    # ...unless mismatches exceed this fraction of the dataset's episodes

    def _check_length_match(self, episode_id, cam_id, expected_len, actual_len):
        """Enforce that this camera's video/latent frame count matches its annotation's
        `state` array length -- the two are assumed frame-aligned everywhere else in this
        class (__getitem__ builds `state_id` from the exact same `rgb_id` used to index
        video/latent frames), so a mismatch means the sampled frames wouldn't genuinely
        correspond to their claimed action/state entry.

        One mismatched episode is treated like any other bad sample: raises
        `_EpisodeLengthMismatch`, which __getitem__'s existing handling catches and skips
        (retries a different random sample), same as a corrupt file. If mismatches pile up
        across many distinct episodes, that's no longer a one-off bad file but a likely
        systemic problem (e.g. a preprocessing bug) -- raises `VideoActionLengthMismatchError`
        instead, which propagates all the way out instead of being silently skipped.
        """
        if actual_len == expected_len:
            return
        if episode_id not in self._mismatched_episodes:
            self._mismatched_episodes.add(episode_id)
            logger.warning(
                f"⚠️ [{self.dataset_name}] episode {episode_id} cam {cam_id}: "
                f"{actual_len} video/latent frames vs {expected_len} action/state entries -- skipping."
            )
        n_bad = len(self._mismatched_episodes)
        if n_bad >= self._LENGTH_MISMATCH_MIN_COUNT and n_bad >= self._LENGTH_MISMATCH_RATIO * self._num_episodes:
            raise VideoActionLengthMismatchError(
                f"[{self.dataset_name}] {n_bad}/{self._num_episodes} episodes have a video/latent "
                f"frame count that doesn't match their action/state array length -- this looks like "
                f"a systemic data problem (e.g. a preprocessing bug or a bad re-encode), not "
                f"one-off corrupt episodes. Affected episodes include: "
                f"{sorted(self._mismatched_episodes)[:10]}{'...' if n_bad > 10 else ''}. "
                f"Fix the underlying video/annotation data before continuing training."
            )
        raise _EpisodeLengthMismatch(
            f"[{self.dataset_name}] episode {episode_id} cam {cam_id}: {actual_len} video/latent "
            f"frames vs {expected_len} action/state entries"
        )

    # ------------------------------------------------------------------
    # Per-camera latent loading (preferred path — skips VAE encoding at train time).
    # ------------------------------------------------------------------

    def _load_one_latent(self, lat_dir, cam_id, rgb_id, episode_id, expected_len):
        pt_path = os.path.join(lat_dir, f"{cam_id}.pt")
        if not os.path.isfile(pt_path):
            return None
        with open(pt_path, "rb") as f:
            tensor = torch.load(f)
        tensor.requires_grad_(False)
        n = tensor.shape[0]
        self._check_length_match(episode_id, cam_id, expected_len, n)
        max_f = n - 1
        ids = [int(min(i, max_f)) for i in rgb_id]  # clamp: defensive only, lengths already verified equal above
        return tensor[ids].float()  # (T, 4, H_lat, W_lat)

    def _slot_cams(self):
        """Camera id per vertical slot, in stacking order, for this config's stacking_mode."""
        cfg = self.config
        if cfg.stacking_mode == "four_view":
            return list(cfg.cam_ids)
        if cfg.stacking_mode == "three_view":
            return [cfg.right_view_id, cfg.left_view_id, cfg.wrist_view_id]
        if cfg.stacking_mode == "two_view":
            return [cfg.right_view_id, cfg.left_view_id, cfg.left_view_id]  # wrist repeats middle+bottom
        return [cfg.cam_id, cfg.cam_id, cfg.cam_id]  # single camera, tiled

    def _load_latent_stacked(self, episode_id, rgb_id, expected_len):
        """Load each unique camera's latent once, then assemble stacked slots.

        Returns None if any required camera's latent file is missing on disk
        (caller falls back to raw video).
        """
        lat_dir = self._resolve_ep_dir(self.latent_root, episode_id)
        slot_cams = self._slot_cams()  # camera id per vertical slot, in stacking order

        unique_cams = list(dict.fromkeys(slot_cams))  # load each distinct camera only once
        cam_latents = {}
        for cam in unique_cams:
            lat = self._load_one_latent(lat_dir, cam, rgb_id, episode_id, expected_len)
            if lat is None:
                return None  # a required camera's latent is missing -> caller falls back to raw video
            cam_latents[cam] = lat

        parts = [cam_latents[c] for c in slot_cams]  # re-expand back to per-slot order (a camera may repeat)
        T, C, H_lat, W_lat = parts[0].shape
        n_slots = len(parts)
        stacked = torch.zeros((T, C, n_slots * H_lat, W_lat), dtype=torch.float32)  # slots stacked vertically
        for i, part in enumerate(parts):
            stacked[:, :, i * H_lat:(i + 1) * H_lat, :] = part  # place this slot's latent into its vertical band
        return stacked

    # ------------------------------------------------------------------
    # Per-camera raw video loading (fallback when latents aren't pre-encoded).
    # ------------------------------------------------------------------

    def _video_path(self, episode_id, cam_id):
        return os.path.join(self._resolve_ep_dir(self.video_root, episode_id), f"{cam_id}.mp4")

    def _load_one_video(self, path, cam_id, rgb_id, episode_id, expected_len):
        vr = VideoReader(path, ctx=cpu(0), num_threads=2)  # decord reader for this camera's mp4
        n = len(vr)
        self._check_length_match(episode_id, cam_id, expected_len, n)
        frames = vr.get_batch([int(min(i, n - 1)) for i in rgb_id]).asnumpy()  # clamp: defensive only, lengths already verified equal above
        return maybe_flip_bgr(frames, self.dataset_name, cam_id)  # correct for cameras confirmed to store BGR

    def _load_video_stacked(self, episode_id, rgb_id, expected_len):
        """Load each camera's raw frames and stack in pixel space (mirrors `_load_latent_stacked`)."""
        slot_cams = self._slot_cams()  # camera id per vertical slot, in stacking order
        unique_cams = list(dict.fromkeys(slot_cams))  # load each distinct camera only once
        cam_frames = {
            cam: self._load_one_video(self._video_path(episode_id, cam), cam, rgb_id, episode_id, expected_len)
            for cam in unique_cams
        }
        # Pass only the unique camera views (in slot order) -- the stacker functions do their
        # own internal duplication for a repeated slot (e.g. stack_two_view_tiled's wrist filling
        # 2 of 3 bands), so re-expanding to len(slot_cams) args here would pass too many
        # positional args whenever a camera repeats (e.g. two_view: 3 slots, 2 unique cams).
        unique_views = [cam_frames[c] for c in unique_cams]
        stacker = STACKERS[self.config.stacking_mode] if len(unique_cams) > 1 else STACKERS[None]  # pick vertical-stack fn for this camera layout
        stacked = stacker(*unique_views) if len(unique_cams) > 1 else stacker(unique_views[0])
        return frames_to_video_tensor(stacked, self.video_size)  # (C, T, H, W)

    # ------------------------------------------------------------------
    # Subclass hook + shared __getitem__
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_action(self, ann, state_id):
        """Return this embodiment's action tensor for the sampled frame indices."""
        raise NotImplementedError

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            with open(sample["ann_file"]) as f:
                ann = json.load(f)

            expected_len = len(ann["state"])
            frame_len = expected_len - 1
            frame_now = min(sample["frame_id"], frame_len)
            rgb_id = self.sampler.sample(frame_now, frame_len)
            state_id = rgb_id  # dataset state is recorded at the same rate as video

            action = self._load_action(ann, state_id)
            text = (ann.get("texts") or [""])[0]

            # Prefer pre-encoded SVD latents; fall back to raw video + on-the-fly encoding.
            # Both loaders verify each camera's frame count matches expected_len (see
            # _check_length_match) before ever indexing into it.
            latent = self._load_latent_stacked(ann["episode_id"], rgb_id, expected_len)
            if latent is not None:
                return {"latent": latent, "action": action, "text": text}
            video = self._load_video_stacked(ann["episode_id"], rgb_id, expected_len)
            return {"video": video, "action": action, "text": text}

        except VideoActionLengthMismatchError:
            raise  # systemic data problem across many episodes -- surface it, don't silently retry past it
        except Exception:
            warnings.warn(f"Bad sample {self.samples[index]['ann_file']}:\n{traceback.format_exc()}")
            return self[np.random.randint(len(self.samples))]  # skip bad samples, don't crash training
