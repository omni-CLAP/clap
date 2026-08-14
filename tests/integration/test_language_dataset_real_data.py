"""Integration tests for clap.data.language.LanguageDataset against real OXE data on disk."""

import os

import pytest

from tests.conftest import requires_oxe_data


def _meta_info_path():
    return os.environ.get("CLAP_META_INFO_ROOT", "dataset_meta_info")


@requires_oxe_data
@pytest.mark.parametrize("action_caption_mode", ["absolute", "relative"])
def test_language_dataset_action_caption_steps(oxe_base_path, action_caption_mode):
    from clap.data.language import LanguageDataset

    meta_info_path = _meta_info_path()
    if not os.path.isdir(os.path.join(meta_info_path, "droid")):
        pytest.skip(f"no stat.json for droid at {meta_info_path}")

    ds = LanguageDataset(
        dataset_name="droid", oxe_base_path=oxe_base_path, meta_info_path=meta_info_path,
        num_history=6, num_frames=5, mode="val", debug=True, action_caption_mode=action_caption_mode,
    )
    assert len(ds) > 0
    sample = ds[0]

    assert "action_caption_steps" in sample
    steps = sample["action_caption_steps"]
    assert len(steps) == 11  # num_history + num_frames
    assert all(isinstance(s, str) and s for s in steps)
    # x=/y=/z=/roll=/pitch=/yaw=/grip= fields present in every per-frame caption.
    assert all(field in steps[0] for field in ("x=", "y=", "z=", "roll=", "pitch=", "yaw=", "grip="))


@requires_oxe_data
def test_language_dataset_relative_anchor_frame_is_near_zero(oxe_base_path):
    """The anchor frame (index num_history) should caption ~zero motion in relative mode."""
    from clap.data.language import LanguageDataset

    meta_info_path = _meta_info_path()
    if not os.path.isdir(os.path.join(meta_info_path, "droid")):
        pytest.skip(f"no stat.json for droid at {meta_info_path}")

    ds = LanguageDataset(
        dataset_name="droid", oxe_base_path=oxe_base_path, meta_info_path=meta_info_path,
        num_history=6, num_frames=5, mode="val", debug=True, action_caption_mode="relative",
    )
    sample = ds[0]
    anchor_caption = sample["action_caption_steps"][6]  # index num_history
    # Relative captions use decimals=0 with a large scale, so ~zero motion formats as "0".
    assert "x=0 " in anchor_caption
    assert "y=0 " in anchor_caption
    assert "z=0 " in anchor_caption


@requires_oxe_data
def test_build_oxe_language_dataset_mix(oxe_base_path):
    from clap.data.language import build_oxe_language_dataset
    from clap.data.oxe_catalog import OXE_LANG_DATASET_ORDER

    meta_info_path = _meta_info_path()
    concat = build_oxe_language_dataset(
        oxe_base_path=oxe_base_path, meta_info_path=meta_info_path,
        num_history=6, num_frames=5, video_size=(576, 320), mode="val", debug=True,
    )
    assert len(concat.datasets) <= len(OXE_LANG_DATASET_ORDER)
    assert len(concat.datasets) == len(concat.sampling_weights)
