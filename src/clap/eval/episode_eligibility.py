"""Cross-embodiment chunk-eligibility check.

An episode is only usable for one autoregressive-replay chunk of
`T = num_history + num_frames` timesteps if EE state and LAM latents both
have enough samples at their respective (possibly downsampled) rates.
Language conditioning builds its per-frame captions at runtime from the same
state array EE uses, so it needs no separate check. Centralizing this one
check keeps the eligible-episode set consistent across families evaluated on
the same test set.
"""

import json
import os
from typing import Dict, Optional

import numpy as np
from decord import VideoReader, cpu

from clap.data.oxe_catalog import get_embodiment_config


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _primary_video_path(oxe_base_path: str, dataset_name: str, split: str, ep_id: str) -> str:
    """One representative camera's mp4 (other cameras in a stack are assumed to match its frame count)."""
    cfg = get_embodiment_config(dataset_name)
    # Stacked layouts read their left camera as the representative one; single-camera datasets have only cam_id.
    cam = cfg.left_view_id if cfg.stacking_mode in ("three_view", "two_view") else cfg.cam_id
    return os.path.join(oxe_base_path, dataset_name, "videos", split, str(ep_id), f"{cam}.mp4")


def _candidate_lam_dirs(ep_id: str):
    """LAM episode folder naming conventions seen in practice: unpadded int, or zero-padded 'episode_NNNNNN'."""
    rels = [str(ep_id)]
    try:
        rels.append(f"episode_{int(ep_id):06d}")
    except (ValueError, TypeError):
        pass  # ep_id isn't numeric (e.g. egodex's "<task>/<id>" keys) -> only the unpadded form applies
    return rels


def compute_data_lengths(dataset_name, ep_id, oxe_base_path, oxe_lam_root, oxe_lam_subdir, split) -> Dict[str, Optional[int]]:
    """Read the 3 data-source lengths for one episode — cheap reads only (JSON, npy header, one VideoReader open).

    Each entry is an int if that source exists and was readable, else None.
    """
    out: Dict[str, Optional[int]] = {"n_video": None, "n_state": None, "n_lam": None}

    # n_state: length of the EE-state annotation array (native, non-downsampled rate).
    ann_path = os.path.join(
        oxe_base_path, dataset_name, get_embodiment_config(dataset_name).annotation_subdir, split, f"{ep_id}.json"
    )
    if os.path.isfile(ann_path):
        try:
            with open(ann_path) as f:
                out["n_state"] = len(json.load(f).get("state", []))
        except Exception:
            pass  # missing/corrupt annotation -> leave as None, episode is ineligible

    # n_lam: length of the precomputed LAM latent-action array (already at the downsampled rate).
    # Try both episode-folder naming conventions; stop at the first that exists.
    for rel in _candidate_lam_dirs(ep_id):
        lam_path = os.path.join(oxe_lam_root, dataset_name, oxe_lam_subdir, split, rel, "latent_actions.npy")
        if os.path.isfile(lam_path):
            try:
                out["n_lam"] = int(np.load(lam_path, mmap_mode="r").shape[0])  # mmap: read only the header/shape
                break
            except Exception:
                pass

    # n_video: frame count of one representative camera (native rate); the most expensive of the 3 reads.
    video_path = _primary_video_path(oxe_base_path, dataset_name, split, ep_id)
    if os.path.isfile(video_path):
        try:
            out["n_video"] = len(VideoReader(video_path, ctx=cpu(0)))
        except Exception:
            pass

    return out


def chunk_eligible(lengths: Dict[str, Optional[int]], d: int, T_required: int) -> bool:
    """True iff both EE and LAM's available chunk length reach T_required.

    Any missing/unreadable source (a None length) makes the episode
    ineligible, since parity across families can't otherwise be guaranteed.
    """
    n_video, n_state, n_lam = (lengths.get(k) for k in ("n_video", "n_state", "n_lam"))
    if None in (n_video, n_state, n_lam):
        return False  # a missing source means we can't verify parity for that family
    nv = _ceildiv(n_video, d)  # video length after downsampling by d
    ee_t = min(nv, _ceildiv(n_state, d))  # state is native-rate, so it needs the same downsampling
    lam_t = min(nv, n_lam)  # latents are already at the downsampled rate, no further division
    return ee_t >= T_required and lam_t >= T_required


def per_path_t_total(lengths: Dict[str, Optional[int]], d: int) -> Dict[str, Optional[int]]:
    """Per-family T_total values, for diagnostics (not used by the eligibility check itself)."""
    n_video, n_state, n_lam = (lengths.get(k) for k in ("n_video", "n_state", "n_lam"))
    out: Dict[str, Optional[int]] = {"ee": None, "lam": None}
    if n_video is None:
        return out  # without a video length, none of the per-family totals can be computed
    nv = _ceildiv(n_video, d)
    if n_state is not None:
        out["ee"] = min(nv, _ceildiv(n_state, d))
    if n_lam is not None:
        out["lam"] = min(nv, n_lam)
    return out
