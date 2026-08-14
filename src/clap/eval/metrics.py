"""Dataset-level video metrics: FVD and FID, computed over many episodes at once.

Per-episode PSNR/SSIM/LPIPS (overall and per camera view) already come from
`CLAPRolloutAgent.compute_metrics`; FVD/FID need a whole population of videos
at once (Fréchet distance between feature distributions), so they're computed
separately here, once per dataset, from the accumulated GT/prediction videos.
"""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_fvd_extractor = None
_fid_extractor = None
_fid_resizer = None


def _get_fvd_extractor():
    """Lazily construct the single shared R3D-18 feature extractor (loaded once per process)."""
    global _fvd_extractor
    if _fvd_extractor is None:
        _fvd_extractor = _FVDFeatureExtractor()  # first call: build and cache the model
    return _fvd_extractor  # subsequent calls: reuse the cached instance


def _get_fid_extractor():
    """Lazily construct the single shared clean-fid InceptionV3 feature extractor."""
    global _fid_extractor, _fid_resizer
    if _fid_extractor is None:
        from cleanfid.inception_torchscript import InceptionV3W
        from cleanfid.resize import build_resizer

        # clean-fid's own builder hardcodes a /tmp cache path; build the model
        # directly instead so CLEANFID_CACHE_DIR actually controls where weights land.
        cache = os.environ.get("CLEANFID_CACHE_DIR", os.path.expanduser("~/.cache/clap/cleanfid"))
        os.makedirs(cache, exist_ok=True)
        model = InceptionV3W(path=cache, download=True, resize_inside=False).to(_DEVICE).eval()
        _fid_extractor = model
        _fid_resizer = build_resizer("clean")
    return _fid_extractor, _fid_resizer


def _frechet_distance(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    """Fréchet distance between two Gaussian-fitted feature distributions (the F in FVD/FID)."""
    from scipy.linalg import sqrtm

    mu_r, mu_f = feats_real.mean(0), feats_fake.mean(0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_f = np.cov(feats_fake, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # sqrtm can return a tiny imaginary part from numerical noise
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


class _FVDFeatureExtractor:
    """R3D-18 (Kinetics-400 weights) video features, used as the FVD backbone.

    Not directly comparable to published I3D-based FVD numbers — useful for
    relative comparisons across this project's own checkpoints.
    """

    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

    def __init__(self):
        from torchvision.models.video import R3D_18_Weights, r3d_18

        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = torch.nn.Identity()  # use pre-logit features, not classification scores
        self.model = model.to(_DEVICE).eval()
        self.mean = self.MEAN.to(_DEVICE)
        self.std = self.STD.to(_DEVICE)

    @torch.no_grad()
    def features(self, video_uint8: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> (512,) feature vector."""
        v = torch.from_numpy(video_uint8).permute(3, 0, 1, 2).float() / 255.0  # (3, T, H, W)
        v = F.interpolate(v.unsqueeze(0), size=(v.shape[1], 112, 112), mode="trilinear", align_corners=False).to(_DEVICE)
        v = (v - self.mean) / self.std
        return self.model(v).squeeze(0).cpu()


def compute_fvd(gt_videos: List[np.ndarray], pred_videos: List[np.ndarray]) -> Tuple[float, int]:
    """FVD over paired (T, H, W, 3) uint8 video lists. Needs >=2 pairs (single-sample covariance is degenerate)."""
    n = len(gt_videos)
    if n < 2:
        return float("nan"), n
    extractor = _get_fvd_extractor()
    feats_real, feats_fake = [], []
    for gt_vid, pred_vid in zip(gt_videos, pred_videos):
        T = min(len(gt_vid), len(pred_vid))  # a chunk-count mismatch shouldn't crash the whole metric
        feats_real.append(extractor.features(gt_vid[:T]).numpy())
        feats_fake.append(extractor.features(pred_vid[:T]).numpy())
    return _frechet_distance(np.stack(feats_real), np.stack(feats_fake)), n


def compute_fid(gt_videos: List[np.ndarray], pred_videos: List[np.ndarray], skip_first: int = 1, batch_size: int = 64) -> Tuple[float, int]:
    """"Video FID": pool every frame (after skip_first) from every episode into one GT set and one
    prediction set, then compare their InceptionV3 feature distributions — one sample per frame,
    unlike FVD's one sample per video. Needs >=2 pooled frames.
    """
    import torchvision.transforms as T

    model, fn_resize = _get_fid_extractor()
    to_tensor = T.ToTensor()

    def _extract(videos):
        from cleanfid.fid import get_batch_features

        all_feats, batch = [], []
        for video in videos:
            for t in range(skip_first, len(video)):
                resized = fn_resize(video[t])  # (299, 299, 3) float32 in [0, 255]
                batch.append(to_tensor(resized))  # ToTensor on an already-float array: no /255 rescale, stays [0,255]
                if len(batch) >= batch_size:
                    all_feats.append(get_batch_features(torch.stack(batch), model, _DEVICE))
                    batch = []
        if batch:
            all_feats.append(get_batch_features(torch.stack(batch), model, _DEVICE))
        return np.concatenate(all_feats, axis=0) if all_feats else np.zeros((0, 2048))

    feats_gt = _extract(gt_videos)
    feats_pred = _extract(pred_videos)
    n = min(len(feats_gt), len(feats_pred))
    if n < 2:
        return float("nan"), n
    from cleanfid.fid import fid_from_feats
    return float(fid_from_feats(feats_gt[:n], feats_pred[:n])), n
