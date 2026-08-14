"""LAM (latent-action-model) loader: action = a 32-dim learned latent, not a physical unit.

Episode discovery walks the LAM latent-action tree directly (one `.npy` per
episode) rather than annotation JSON files, and text instructions come from a
separately-built {episode -> caption} map instead of a per-sample annotation
lookup — different enough from `EEDataset` that this overrides enumeration and
`__getitem__`, but still reuses `EmbodimentDataset`'s camera-stacking/video
loading (the LAM camera ids match the EE ids for every shared dataset).
"""

import json
import logging
import os
import random
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from clap.data.base import EmbodimentDataset, TemporalSampler, VideoActionLengthMismatchError, get_latent_video_subdir
from clap.data.oxe_catalog import OXE_LAM_DATASET_ORDER, OXE_LAM_LAYOUT, OXE_LAM_SAMPLING_WEIGHTS, get_embodiment_config
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# egodex_lam_subdir substrings meaning "sample only egodex", skipping the rest of
# the mix — matched loosely (not by exact name).
EGODEX_ONLY_LAM_SUBDIR_MARKERS = ("dreamdojo",)


def is_egodex_only_lam_subdir(egodex_lam_subdir):
    return egodex_lam_subdir is not None and any(m in egodex_lam_subdir for m in EGODEX_ONLY_LAM_SUBDIR_MARKERS)


def build_lam_text_map(dataset_name, dataset_path, ann_dir, ann_split):
    """{normalized_episode_key: instruction} for a LAM dataset.

    egodex has no per-frame state annotations, so its instructions come from a
    flat manifest listing {task, episode_id} instead; every other dataset's
    instructions are read the normal way, from each episode's annotation JSON.
    """
    if dataset_name == "egodex":
        # egodex's manifest can live at either of two nesting depths; use whichever exists.
        candidates = [
            os.path.join(dataset_path, "annotations", f"{ann_split}.json"),
            os.path.join(dataset_path, "annotations", "annotations", f"{ann_split}.json"),
        ]
        manifest = next((c for c in candidates if os.path.isfile(c)), None)
        if manifest is None:
            return {}  # no manifest found -> no text available for this split
        try:
            with open(manifest) as f:
                entries = json.load(f)
        except Exception:
            return {}
        out = {}
        for e in entries:
            # Each manifest entry names one episode's task/id; the task string doubles as its instruction.
            task, ep_id = e.get("task") or "", e.get("episode_id")
            if task and ep_id is not None:
                out[_normalize_ep_key(f"{task}/{ep_id}")] = task
        return out

    # Every other dataset: walk the annotation directory and pull each episode's instruction text.
    out = {}
    if not os.path.isdir(ann_dir):
        return out  # no annotations for this split -> empty map, callers fall back to ""
    for fname in os.listdir(ann_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(ann_dir, fname)) as f:
                ann = json.load(f)
        except Exception:
            continue  # unreadable annotation file -> skip, this episode just has no text entry
        texts = ann.get("texts")
        # "texts" (list, use first) is the newer field; "text" (str) is the older one, kept as a fallback.
        txt = texts[0] if isinstance(texts, list) and texts else ann.get("text", "")
        if txt:
            out[_normalize_ep_key(ann.get("episode_id", fname[:-5]))] = txt
    return out


def _normalize_ep_key(rel):
    """Strip an 'episode_' prefix and leading zeros so annotation/LAM-tree keys line up."""
    head, tail = os.path.split(str(rel))  # split off the final path component to normalize
    if tail.startswith("episode_"):
        tail = tail[len("episode_"):]  # strip the "episode_" prefix
    if tail.isdigit():
        tail = str(int(tail))  # drop leading zeros from a numeric id
    return os.path.join(head, tail) if head else tail


class LAMDataset(EmbodimentDataset):
    """Latent-action conditioning: 32-dim per-frame LAM embedding instead of a physical action.

    LAM actions are relative (action[k] encodes the k-1 -> k transition), not
    p01/p99-normalized (the encoder's own embedding is already roughly unit-scale).

    Args:
        oxe_lam_root / oxe_lam_subdir: Root dir and subdir name holding the
            precomputed `latent_actions.npy` per episode.
        egodex_lam_subdir: Overrides `oxe_lam_subdir` for egodex specifically
            (different latent-action extractor runs use different subdir names);
            if this is a "this run is egodex-only" value, the mix drops every
            other dataset (see `EGODEX_ONLY_LAM_SUBDIR_MARKERS`/`is_egodex_only_lam_subdir`).
    """

    def __init__(
        self, dataset_name, oxe_base_path, oxe_lam_root, oxe_lam_subdir,
        num_history, num_frames, lam_action_dim=32, video_size=(576, 320),
        mode="train", debug=False, egodex_lam_subdir=None, lam_subdir_override=None,
    ):
        self.config = get_embodiment_config(dataset_name)
        self.dataset_name = dataset_name
        self.mode = mode
        self.split = "train" if mode == "train" else "val"
        self.video_size = list(video_size)
        self.lam_action_dim = lam_action_dim
        self.normalizer = None  # LAM latents are not p01/p99-normalized
        self._mismatched_episodes = set()  # episode_ids whose video/latent length disagrees with n_latents*fps_ratio

        self.sampler = TemporalSampler(num_history, num_frames)

        layout = OXE_LAM_LAYOUT.get(dataset_name, {})
        self.video_layout = layout.get("video_layout", "folder_stacked")
        self.fps_ratio = layout.get("fps_downsample_ratio", 1)

        self.dataset_path = dataset_path = os.path.join(oxe_base_path, dataset_name)
        if lam_subdir_override is not None and layout.get("lam_subdir") is not None:
            lam_subdir = lam_subdir_override
        elif dataset_name == "egodex" and egodex_lam_subdir is not None:
            lam_subdir = egodex_lam_subdir
        else:
            lam_subdir = layout.get("lam_subdir", oxe_lam_subdir)

        # egodex-style datasets (declared via max_episodes) may have no separate
        # val LAM dir; fall back to reading the tail of the train split as val.
        max_episodes = layout.get("max_episodes")
        val_lam_dir = os.path.join(oxe_lam_root, dataset_name, lam_subdir, self.split)
        self._use_train_tail_as_val = (
            mode != "train" and max_episodes is not None and not os.path.isdir(val_lam_dir)
        )
        # Videos/latents/annotations share that same train-tail fallback, or a
        # val-mode lookup would hit a videos/val/ dir that doesn't exist either.
        lam_split = "train" if self._use_train_tail_as_val else self.split
        self.video_root = os.path.join(dataset_path, "videos", lam_split)
        self.latent_root = os.path.join(dataset_path, get_latent_video_subdir(), lam_split)
        self.lam_dir = os.path.join(oxe_lam_root, dataset_name, lam_subdir, lam_split)
        self.ann_dir = os.path.join(dataset_path, self.config.annotation_subdir, lam_split)

        # egodex's manifest-based text uses the train split's manifest during
        # the train-tail-as-val fallback, same reasoning as lam_split above.
        ann_split = "train" if self._use_train_tail_as_val else self.split
        self.text_map = build_lam_text_map(dataset_name, dataset_path, self.ann_dir, ann_split)
        self.episodes = self._enumerate_episodes(max_episodes, debug=debug)
        self._num_episodes = len(self.episodes)  # denominator for _check_length_match's ratio threshold
        self.samples = self._build_samples()
        logger.info(f"📊 LAMDataset [{dataset_name}] ({mode}): {len(self.episodes)} eps, "
                    f"{len(self.samples)} anchors (lam_dir={self.lam_dir})")

    def _enumerate_episodes(self, max_episodes, debug=False):
        """Walk the LAM latent-action tree directly — one entry per `latent_actions.npy` found.

        debug=True stops the walk after 8 episodes for fast local iteration, which makes the
        production max_episodes head/tail train-tail-as-val split meaningless (slicing an
        8-item list by a ~100k offset always yields an empty train OR val set depending on
        direction) — so that split is skipped entirely in debug mode; train/val may then
        overlap, which is fine for a smoke test but not for a real train/val split.
        """
        if not os.path.isdir(self.lam_dir):
            return []  # no LAM latents for this dataset/split
        logger.info(f"🔍 [{self.dataset_name}] scanning {self.lam_dir} for episodes -- can take minutes on a "
                    f"large real tree over a network filesystem; set data.debug_dataset=true to cap this at "
                    f"8 episodes instead for a fast smoke test")
        eps = []
        n_walked = 0
        for root, _, files in os.walk(self.lam_dir):
            n_walked += 1
            if n_walked % 20000 == 0:
                logger.info(f"🔍 [{self.dataset_name}] still scanning ({n_walked} dirs walked, {len(eps)} episodes found so far)...")
            if "latent_actions.npy" not in files:
                continue  # not an episode's LAM folder
            npy_path = os.path.join(root, "latent_actions.npy")
            try:
                n = int(np.load(npy_path, mmap_mode="r").shape[0])  # mmap: just read the header for the frame count
            except Exception:
                continue  # unreadable/corrupt latent file -> skip this episode
            if n <= 0:
                continue  # empty latent array -> no usable frames
            eps.append({"rel": os.path.relpath(root, self.lam_dir), "npy_path": npy_path, "n_latents": n})
            if debug and len(eps) >= 8:
                break  # debug mode: stop early for fast local iteration
        eps.sort(key=lambda e: e["rel"])  # deterministic ordering
        if max_episodes is not None and not debug:
            eps = eps[max_episodes:] if self._use_train_tail_as_val else eps[:max_episodes]  # head/tail train/val split
        return eps

    def _build_samples(self):
        samples = []
        for ep_idx, ep in enumerate(self.episodes):
            # Step by fps_ratio so frame_id always lands on a LAM-frame boundary
            # (for fps_ratio=1 this is just range(n_latents)).
            for frame_id in range(0, ep["n_latents"] * self.fps_ratio, self.fps_ratio):
                samples.append({"ep_idx": ep_idx, "frame_id": frame_id})
        return samples

    def __len__(self):
        return len(self.samples)

    def _build_rgb_id(self, frame_now, frame_len):
        # Same window-building as TemporalSampler, but strides are scaled by
        # fps_ratio so they land on LAM-frame boundaries too.
        skip = random.randint(1, 2) * self.fps_ratio  # random future stride, scaled to LAM-frame boundaries
        skip_his = int(skip * 4)  # history stride, coarser than the future stride
        if random.random() < 0.15:
            skip_his = 0  # occasionally collapse history to the anchor frame
        rgb_id = [int(frame_now - i * skip_his) for i in range(self.sampler.num_history, 0, -1)]  # history indices, oldest first
        rgb_id.append(frame_now)  # anchor/conditioning frame
        rgb_id += [int(frame_now + i * skip) for i in range(1, self.sampler.num_frames)]  # future frame indices
        return [int(x) for x in np.clip(rgb_id, 0, frame_len).tolist()]  # clamp to valid episode range

    def _load_action(self, ep, rgb_id):
        """Per-frame LAM latents -> (T, 32).

        Slot 0 has no prior frame so it stays zero. Any slot whose rgb_id
        equals the previous slot's (clipped to the same source frame — no
        real motion) is also left zero rather than reading that frame's
        stored transition, which would encode spurious motion.
        """
        arr = np.load(ep["npy_path"])  # (n_latents, lam_action_dim) per-frame LAM latents
        last = len(arr) - 1
        out = np.zeros((len(rgb_id), self.lam_action_dim), dtype=np.float32)  # (T, 32); slot 0 stays zero (no prior frame)
        for k in range(1, len(rgb_id)):
            if rgb_id[k] == rgb_id[k - 1]:
                continue  # same source frame as previous slot -> no real motion, leave zero
            out[k] = arr[int(min(rgb_id[k] // self.fps_ratio, last))]  # map video-frame index back to a latent row
        return torch.from_numpy(out).float()

    def _video_path(self, episode_id, cam_id):
        """Override: egodex's "flat_mp4" layout stores one video directly per episode
        (e.g. videos/train/<task>/<episode_id>.mp4), not the per-camera-subdirectory
        layout `EmbodimentDataset._video_path` assumes for every other dataset."""
        if self.video_layout == "flat_mp4":
            return self._resolve_ep_dir(self.video_root, episode_id).rstrip("/") + ".mp4"
        return super()._video_path(episode_id, cam_id)

    def __getitem__(self, index):
        try:
            sample = self.samples[index]
            ep = self.episodes[sample["ep_idx"]]
            expected_len = ep["n_latents"] * self.fps_ratio  # video-frame space
            frame_len = expected_len - 1
            frame_now = min(sample["frame_id"], frame_len)

            rgb_id = self._build_rgb_id(frame_now, frame_len)
            action = self._load_action(ep, rgb_id)
            text = self.text_map.get(_normalize_ep_key(ep["rel"]), "")

            # Both loaders verify each camera's frame count matches expected_len (see
            # EmbodimentDataset._check_length_match) before ever indexing into it.
            latent = self._load_latent_stacked(ep["rel"], rgb_id, expected_len)
            if latent is not None:
                return {"latent": latent, "action": action, "text": text}
            video = self._load_video_stacked(ep["rel"], rgb_id, expected_len)
            return {"video": video, "action": action, "text": text}

        except VideoActionLengthMismatchError:
            raise  # systemic data problem across many episodes -- surface it, don't silently retry past it
        except Exception:
            warnings.warn(f"Bad LAM sample in {self.dataset_name}:\n{traceback.format_exc()}")
            return self[np.random.randint(len(self.samples))]


def build_oxe_lam_dataset(oxe_base_path, oxe_lam_root, oxe_lam_subdir, num_history, num_frames,
                           video_size, mode, debug=False, egodex_lam_subdir=None, lam_subdir_override=None):
    """Cross-embodiment LAM pretraining mix, or egodex-only if `egodex_lam_subdir` requests it."""
    egodex_only = is_egodex_only_lam_subdir(egodex_lam_subdir)
    dataset_order = ["egodex"] if egodex_only else OXE_LAM_DATASET_ORDER  # egodex-only run drops the rest of the mix
    sub_datasets = [
        LAMDataset(
            dataset_name=ds, oxe_base_path=oxe_base_path, oxe_lam_root=oxe_lam_root, oxe_lam_subdir=oxe_lam_subdir,
            num_history=num_history, num_frames=num_frames, video_size=video_size, mode=mode, debug=debug,
            egodex_lam_subdir=egodex_lam_subdir, lam_subdir_override=lam_subdir_override,
        )
        for ds in dataset_order
    ]  # one LAMDataset per embodiment in the mix
    concat = ConcatDataset(sub_datasets)
    concat.sampling_weights = [1.0] if egodex_only else list(OXE_LAM_SAMPLING_WEIGHTS)  # per-dataset sampling weight
    return concat
