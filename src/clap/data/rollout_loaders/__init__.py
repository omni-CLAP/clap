"""Full-episode loaders for offline rollout replay/eval.

Unlike the training-time `EmbodimentDataset` (one random windowed sample per
`__getitem__`), these load an entire episode at once — `clap.rollout.agent`
autoregressively steps through it chunk by chunk.
"""

from clap.data.rollout_loaders.ee_loader import EEEpisodeLoader
from clap.data.rollout_loaders.lam_loader import LAMEpisodeLoader

# Dispatch table for clap.rollout.replay, keyed by family name. "language" uses
# the same loader as "ee" — its per-frame captions are built downstream from
# the same states/grippers/text this loader already returns. bridge/taco_play/etc.
# also use "ee" (not their own key) -- family only ever selects a LOADER CLASS here,
# and per-dataset camera/video handling is already looked up from dataset_name via
# clap.data.oxe_catalog, independent of family.
ROLLOUT_LOADERS = {
    "ee": EEEpisodeLoader,
    "bimanual_yam": EEEpisodeLoader,
    "g1_humanoid": EEEpisodeLoader,
    "language": EEEpisodeLoader,
    "lam": LAMEpisodeLoader,
}

__all__ = [
    "ROLLOUT_LOADERS",
    "EEEpisodeLoader",
    "LAMEpisodeLoader",
]
