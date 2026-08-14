"""Language-conditioned loader: per-frame CLIP captions instead of a numeric action.

Reuses EEDataset's enumeration, temporal sampling, and video/latent loading
entirely — the only addition is `action_caption_steps`, a per-frame caption
string list built from the same normalized EE action every sample already carries.
"""

import logging

from torch.utils.data import ConcatDataset

from clap.data.action_caption import format_action_caption, format_relative_action_caption, relativize_action_window
from clap.data.ee import EEDataset
from clap.data.oxe_catalog import OXE_LANG_DATASET_ORDER, OXE_LANG_SAMPLING_WEIGHTS
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class LanguageDataset(EEDataset):
    """EE-cartesian action + a per-frame text caption of that action.

    `batch["action"]` carries the normalized EE tensor as usual (unused by the
    model's language-conditioning branch, but harmless to populate);
    `batch["action_caption_steps"]` carries the per-frame caption strings the
    model actually conditions on.

    Args:
        action_caption_mode: "absolute" formats the normalized state directly;
            "relative" re-baselines each window against its anchor frame first
            (see `clap.data.action_caption`).
    """

    def __init__(self, dataset_name, oxe_base_path, meta_info_path, num_history, num_frames,
                 video_size=(576, 320), mode="train", debug=False, action_caption_mode="absolute"):
        assert action_caption_mode in ("absolute", "relative"), action_caption_mode  # only two supported caption formats
        self.action_caption_mode = action_caption_mode
        super().__init__(
            dataset_name=dataset_name, oxe_base_path=oxe_base_path, meta_info_path=meta_info_path,
            num_history=num_history, num_frames=num_frames, video_size=video_size, mode=mode, debug=debug,
        )  # reuses EEDataset's enumeration, sampling, and video/latent loading

    def __getitem__(self, index):
        sample = super().__getitem__(index)  # {"latent" or "video", "action", "text"}
        action_np = sample["action"].numpy()

        if self.action_caption_mode == "relative":
            # Anchor = the current/conditioning frame, i.e. index num_history
            # in the window (history frames first, then now, then future).
            caption_action = relativize_action_window(action_np, self.sampler.num_history)
            step_strings = [format_relative_action_caption(*row) for row in caption_action]
        else:
            step_strings = [format_action_caption(*row) for row in action_np]

        sample["action_caption_steps"] = step_strings
        return sample


def build_oxe_language_dataset(oxe_base_path, meta_info_path, num_history, num_frames,
                                video_size, mode, debug=False, action_caption_mode="absolute"):
    """Language-conditioning mix (same 7 real-action datasets as the EE mix; egodex has no numeric action)."""
    sub_datasets = []
    for ds in OXE_LANG_DATASET_ORDER:
        try:
            sub_datasets.append(
                LanguageDataset(
                    dataset_name=ds, oxe_base_path=oxe_base_path, meta_info_path=meta_info_path,
                    num_history=num_history, num_frames=num_frames, video_size=video_size,
                    mode=mode, debug=debug, action_caption_mode=action_caption_mode,
                )
            )
        except FileNotFoundError as e:
            logger.warning(f"⚠️ Skipping language dataset {ds}: {e}")  # dataset not present on disk -> drop from the mix
    concat = ConcatDataset(sub_datasets)
    concat.sampling_weights = list(OXE_LANG_SAMPLING_WEIGHTS)[:len(sub_datasets)]  # trim weights to match any dropped datasets
    return concat
