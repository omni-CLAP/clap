"""Integration tests for clap.data.lam.LAMDataset against real OXE data on disk."""

import pytest
import torch

from tests.conftest import requires_oxe_data


@requires_oxe_data
@pytest.mark.parametrize("dataset_name", ["droid", "bridge", "egodex"])
def test_lam_dataset_sample_shapes_and_dtypes(oxe_base_path, oxe_lam_subdir, dataset_name):
    from clap.data.lam import LAMDataset

    ds = LAMDataset(
        dataset_name=dataset_name, oxe_base_path=oxe_base_path, oxe_lam_root=oxe_base_path,
        oxe_lam_subdir=oxe_lam_subdir, num_history=6, num_frames=5, mode="train", debug=True,
    )
    assert len(ds) > 0
    sample = ds[0]

    assert sample["action"].shape == (11, 32)
    assert sample["action"].dtype == torch.float32
    assert torch.isfinite(sample["action"]).all()
    assert sample["action"][0].abs().sum() == 0.0  # anchor row always zero (no prior transition)

    visual_key = "latent" if "latent" in sample else "video"
    assert visual_key in sample


@requires_oxe_data
def test_lam_dataset_egodex_only_mode(oxe_base_path, oxe_lam_subdir, egodex_dreamdojo_lam_subdir):
    from clap.data.lam import build_oxe_lam_dataset

    concat = build_oxe_lam_dataset(
        oxe_base_path=oxe_base_path, oxe_lam_root=oxe_base_path, oxe_lam_subdir=oxe_lam_subdir,
        num_history=6, num_frames=5, video_size=(576, 320), mode="train", debug=True,
        egodex_lam_subdir=egodex_dreamdojo_lam_subdir,
    )
    assert len(concat.datasets) == 1
    assert concat.datasets[0].dataset_name == "egodex"
    assert concat.sampling_weights == [1.0]


@requires_oxe_data
def test_build_oxe_lam_dataset_mix_covers_all_eight_datasets(oxe_base_path, oxe_lam_subdir):
    from clap.data.lam import build_oxe_lam_dataset
    from clap.data.oxe_catalog import OXE_LAM_DATASET_ORDER

    concat = build_oxe_lam_dataset(
        oxe_base_path=oxe_base_path, oxe_lam_root=oxe_base_path, oxe_lam_subdir=oxe_lam_subdir,
        num_history=6, num_frames=5, video_size=(576, 320), mode="train", debug=True,
    )
    assert len(concat.datasets) == len(OXE_LAM_DATASET_ORDER)
    assert {sub.dataset_name for sub in concat.datasets} == set(OXE_LAM_DATASET_ORDER)
    assert len(concat) > 0
