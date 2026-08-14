"""Unit tests for egodex-specific behavior in the LAM data path: catalog registration,
non-numeric episode-id resolution, and the flat-file video layout."""

from clap.data.base import EmbodimentDataset
from clap.data.lam import is_egodex_only_lam_subdir
from clap.data.oxe_catalog import get_embodiment_config


def test_egodex_is_registered():
    cfg = get_embodiment_config("egodex")
    assert cfg.name == "egodex"
    assert cfg.stacking_mode is None  # single-view tile_to_stack, same as bc_z/taco_play/fractal


def test_is_egodex_only_lam_subdir_matches_any_dreamdojo_variant():
    assert is_egodex_only_lam_subdir("dreamdojo_latent_actions_skip_1")
    assert is_egodex_only_lam_subdir("dreamdojo_latent_actions_skip_2")  # a different extractor run
    assert not is_egodex_only_lam_subdir("latent_actions")
    assert not is_egodex_only_lam_subdir(None)


def test_resolve_ep_dir_handles_non_numeric_episode_id(tmp_path):
    """Episode ids can be "task/id"-style strings (egodex), not just integers."""
    (tmp_path / "basic_fold").mkdir()
    (tmp_path / "basic_fold" / "5804").mkdir()
    resolved = EmbodimentDataset._resolve_ep_dir(object(), str(tmp_path), "basic_fold/5804")
    assert resolved == str(tmp_path / "basic_fold" / "5804")


def test_resolve_ep_dir_still_handles_numeric_episode_id(tmp_path):
    (tmp_path / "episode_000042").mkdir()
    resolved = EmbodimentDataset._resolve_ep_dir(object(), str(tmp_path), "42")
    assert resolved == str(tmp_path / "episode_000042")


def test_lam_dataset_video_path_flat_mp4_layout():
    """egodex stores one video per episode directly (task/<episode_id>.mp4), not
    the <episode_dir>/<cam_id>.mp4 layout every other LAM dataset uses."""
    from clap.data.lam import LAMDataset

    # Build a bare instance without running __init__ (avoids needing real data on disk)
    # to unit-test _video_path's branching logic in isolation.
    ds = LAMDataset.__new__(LAMDataset)
    ds.video_layout = "flat_mp4"
    ds.video_root = "/fake/videos/train"
    path = ds._video_path("basic_fold/5804", cam_id=0)
    assert path == "/fake/videos/train/basic_fold/5804.mp4"


def test_lam_dataset_video_path_folder_layout_unchanged():
    from clap.data.lam import LAMDataset

    ds = LAMDataset.__new__(LAMDataset)
    ds.video_layout = "folder_stacked"
    ds.video_root = "/fake/videos/train"
    path = ds._video_path("42", cam_id="0")
    assert path == "/fake/videos/train/42/0.mp4"
