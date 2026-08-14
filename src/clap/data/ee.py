"""EE-cartesian and joint-space action loader.

Covers droid, bridge, bimanual_yam, and g1_humanoid — selected per-instance by
`EmbodimentConfig` (camera layout + action_mode), not by subclassing.
"""

import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from clap.data.base import EmbodimentDataset
from clap.data.oxe_catalog import OXE_EE_DATASET_ORDER, OXE_EE_SAMPLING_WEIGHTS, get_embodiment_config


class EEDataset(EmbodimentDataset):
    """EE-cartesian or joint-space action loader, selected by `config.action_mode`.

    "ee7" (default): action = state[:, :6] (xyz+rpy) + continuous_gripper_state,
    normalized via the dataset's stat.json. "joint14"/"joint26": action = the
    raw `state` array as-is (bimanual/humanoid joint angles, already includes
    any gripper/hand dims — no separate gripper concat or the EE convention's
    6-dim slice).
    """

    def __init__(self, dataset_name, oxe_base_path, meta_info_path, num_history, num_frames,
                 video_size=(576, 320), mode="train", debug=False):
        config = get_embodiment_config(dataset_name)  # camera layout + action_mode for this dataset
        stat_path = os.path.join(meta_info_path, dataset_name, "stat.json")  # p01/p99 action bounds
        super().__init__(
            config=config, oxe_base_path=oxe_base_path, num_history=num_history, num_frames=num_frames,
            video_size=video_size, mode=mode, annotation_subdir=config.annotation_subdir,
            stat_path=stat_path, debug=debug,
        )

    def _load_action(self, ann, state_id):
        state = np.array(ann["state"])[state_id]  # (T, D)
        if self.config.action_mode == "ee7":
            gripper = np.array(ann["continuous_gripper_state"])[state_id]  # (T,)
            action = np.concatenate([state[:, :6], gripper[:, None]], axis=-1)  # (T, 7)
        else:
            action = state  # joint-space: use the raw state array as the action
        action = self.normalizer.normalize(action)
        return torch.from_numpy(action).float()


def build_oxe_ee_dataset(oxe_base_path, meta_info_path, num_history, num_frames,
                          video_size, mode, debug=False):
    """Cross-embodiment OXE modeling mix (the 7 datasets in `OXE_EE_DATASET_ORDER`)."""
    sub_datasets = [
        EEDataset(
            dataset_name=ds, oxe_base_path=oxe_base_path, meta_info_path=meta_info_path,
            num_history=num_history, num_frames=num_frames, video_size=video_size,
            mode=mode, debug=debug,
        )
        for ds in OXE_EE_DATASET_ORDER
    ]  # one EEDataset per embodiment in the mix
    concat = ConcatDataset(sub_datasets)
    concat.sampling_weights = list(OXE_EE_SAMPLING_WEIGHTS)  # per-dataset sampling weight, read by the training sampler
    return concat
