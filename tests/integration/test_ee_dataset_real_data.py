"""Integration tests for clap.data.ee.EEDataset against real OXE data on disk."""

import os

import numpy as np
import pytest
import torch

from tests.conftest import requires_oxe_data


def _meta_info_path():
    return os.environ.get("CLAP_META_INFO_ROOT", "dataset_meta_info")


def _skip_if_no_stat(dataset_name):
    path = os.path.join(_meta_info_path(), dataset_name)
    if not os.path.isdir(path):
        pytest.skip(f"no stat.json for {dataset_name} at {path}")


@requires_oxe_data
@pytest.mark.parametrize("dataset_name", ["droid", "bridge"])
def test_ee_dataset_sample_shapes_and_dtypes(oxe_base_path, dataset_name):
    from clap.data.ee import EEDataset

    _skip_if_no_stat(dataset_name)
    ds = EEDataset(
        dataset_name=dataset_name, oxe_base_path=oxe_base_path, meta_info_path=_meta_info_path(),
        num_history=6, num_frames=5, mode="val", debug=True,
    )
    assert len(ds) > 0
    sample = ds[0]

    assert sample["action"].shape == (11, 7)  # num_history + num_frames, ee7
    assert sample["action"].dtype == torch.float32
    assert torch.isfinite(sample["action"]).all()
    assert sample["action"].min() >= -1.0 - 1e-4
    assert sample["action"].max() <= 1.0 + 1e-4

    visual_key = "latent" if "latent" in sample else "video"
    assert visual_key in sample
    assert sample[visual_key].shape[0] == 11 if visual_key == "latent" else sample[visual_key].shape[1] == 11


@requires_oxe_data
def test_ee_dataset_bimanual_yam_joint_action_not_normalized_via_ee7_slice(oxe_base_path):
    """joint14 action_mode uses the raw state array, not the state[:6]+gripper slice."""
    from clap.data.ee import EEDataset

    _skip_if_no_stat("bimanual_yam")
    ds = EEDataset(
        dataset_name="bimanual_yam", oxe_base_path=oxe_base_path, meta_info_path=_meta_info_path(),
        num_history=6, num_frames=5, mode="val", debug=True,
    )
    if len(ds) == 0:
        pytest.skip("no bimanual_yam val episodes available")
    sample = ds[0]
    assert sample["action"].shape == (11, 14)


@requires_oxe_data
def test_build_oxe_ee_dataset_mix_covers_all_seven_datasets(oxe_base_path):
    from clap.data.ee import build_oxe_ee_dataset
    from clap.data.oxe_catalog import OXE_EE_DATASET_ORDER

    concat = build_oxe_ee_dataset(
        oxe_base_path=oxe_base_path, meta_info_path=_meta_info_path(),
        num_history=6, num_frames=5, video_size=(576, 320), mode="val", debug=True,
    )
    assert len(concat.datasets) == len(OXE_EE_DATASET_ORDER)
    assert len(concat) > 0
