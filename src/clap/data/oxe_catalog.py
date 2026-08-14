"""Per-embodiment `EmbodimentConfig` registry, keyed by name.

Merges what used to be three separate dicts (OXE_EE_CONFIGS, the LAM-only
config, and eval/datasets.py's DatasetSpec) into one registry both training
and eval read from.

Per-dataset directory-name overrides live in this same family: `_ann_subdir`
below reads `CLAP_<NAME>_ANNOTATION_SUBDIR` per dataset; the pre-encoded
SVD video-latent subdir (shared across all datasets, not per-dataset) is
`CLAP_LATENT_VIDEO_SUBDIR`, read by `clap.data.base.get_latent_video_subdir`.
"""

import os

from clap.config.data import EmbodimentConfig


def _ann_subdir(name: str) -> str:
    """Default "annotation" for every dataset, overridable per-dataset via
    CLAP_<NAME>_ANNOTATION_SUBDIR for
    a deployment whose own copy of that dataset's annotations live under a different
    directory name (e.g. a second/later annotation pass) — a site config choice, never
    hardcoded as a package default, since other deployments' data may not have it at all.
    """
    return os.environ.get(f"CLAP_{name.upper()}_ANNOTATION_SUBDIR", "annotation")


OXE_CATALOG = {
    "fractal": EmbodimentConfig(name="fractal", cam_id=0, fps_downsample_ratio=1, annotation_subdir=_ann_subdir("fractal")),
    "fmb": EmbodimentConfig(
        name="fmb", stacking_mode="three_view",
        wrist_view_id=4, left_view_id=0, right_view_id=2, annotation_subdir=_ann_subdir("fmb"),
    ),
    "bc_z": EmbodimentConfig(name="bc_z", cam_id=0, annotation_subdir=_ann_subdir("bc_z")),
    "taco_play": EmbodimentConfig(name="taco_play", cam_id=3, annotation_subdir=_ann_subdir("taco_play")),
    "furniture_bench": EmbodimentConfig(
        name="furniture_bench", stacking_mode="two_view", left_view_id=1, right_view_id=0,
        annotation_subdir=_ann_subdir("furniture_bench"),
    ),
    "bridge": EmbodimentConfig(name="bridge", cam_id="rgb", annotation_subdir=_ann_subdir("bridge")),  # bridge stores video as rgb.mp4
    "droid": EmbodimentConfig(
        name="droid", stacking_mode="three_view",
        wrist_view_id=2, left_view_id=0, right_view_id=1, annotation_subdir=_ann_subdir("droid"),
    ),
    # LAM-only: egocentric human video, single camera. stacking_mode=None (default) ->
    # tile_to_stack, the same single-view-tripled treatment bc_z/taco_play/fractal get.
    # LAMDataset is the only class that ever constructs this entry, via OXE_LAM_DATASET_ORDER.
    "egodex": EmbodimentConfig(name="egodex", cam_id=0, annotation_subdir=_ann_subdir("egodex")),
    # Held-out generalization targets: unseen during pretraining, used to probe
    # cross-dataset transfer. Same scene-top/wrist-bottom two_view convention.
    "austin_sailor": EmbodimentConfig(
        name="austin_sailor", stacking_mode="two_view", left_view_id=1, right_view_id=0,
        annotation_subdir=_ann_subdir("austin_sailor"),
    ),
    "berkeley_autolab_ur5": EmbodimentConfig(
        name="berkeley_autolab_ur5", stacking_mode="two_view", left_view_id=0, right_view_id=1,
        annotation_subdir=_ann_subdir("berkeley_autolab_ur5"),
    ),
    "stanford_hydra": EmbodimentConfig(
        name="stanford_hydra", stacking_mode="two_view", left_view_id=1, right_view_id=0,
        annotation_subdir=_ann_subdir("stanford_hydra"),
    ),
    "utaustin_mutex": EmbodimentConfig(
        name="utaustin_mutex", stacking_mode="two_view", left_view_id=1, right_view_id=0,
        annotation_subdir=_ann_subdir("utaustin_mutex"),
    ),
    # Bimanual YAM adaptation target: 3 cameras (top/scene + 2 wrists), 14-dim
    # joint-space action (6 joints + gripper per arm) instead of cartesian EE.
    "bimanual_yam": EmbodimentConfig(
        name="bimanual_yam", stacking_mode="three_view", action_mode="joint14",
        right_view_id=0, left_view_id=1, wrist_view_id=2, annotation_subdir=_ann_subdir("bimanual_yam"),
    ),
    # G1 humanoid adaptation target: 4 cameras (2 overhead + 2 wrist), 26-dim
    # absolute joint positions (7+7 arms, 6+6 hands) instead of cartesian EE.
    "g1_humanoid": EmbodimentConfig(
        name="g1_humanoid", stacking_mode="four_view", action_mode="joint26",
        cam_ids=[0, 1, 2, 3], hand_type="brainco",  # [right_high, left_high, right_wrist, left_wrist]
        annotation_subdir=_ann_subdir("g1_humanoid"),
    ),
}

# Cross-embodiment modeling mix: which datasets + how heavily to sample each.
OXE_EE_DATASET_ORDER = ["fractal", "fmb", "bc_z", "taco_play", "furniture_bench", "bridge", "droid"]
OXE_EE_SAMPLING_WEIGHTS = [2.00, 2.00, 2.00, 2.00, 2.00, 15.00, 75.00]  # bridge/droid oversampled

# LAM (latent-action) modeling mix additionally includes egodex (human video,
# no camera stacking needed) and drops nothing else from the EE mix.
OXE_LAM_DATASET_ORDER = ["egodex", "bridge", "fractal", "droid", "bc_z", "fmb", "taco_play", "furniture_bench"]
OXE_LAM_SAMPLING_WEIGHTS = [2.50, 15.00, 1.50, 75.00, 1.50, 1.50, 1.50, 1.50]

# Language-caption conditioning mix: same 7 real-action datasets as the EE mix
# (egodex has no numeric action to caption, so it's excluded here).
OXE_LANG_DATASET_ORDER = ["bc_z", "bridge", "droid", "fmb", "fractal", "furniture_bench", "taco_play"]
OXE_LANG_SAMPLING_WEIGHTS = [2.00, 15.00, 75.00, 2.00, 2.00, 2.00, 2.00]

# LAM-specific per-dataset video layout info (video_layout/cam_name/lam_subdir),
# not needed by EEDataset/LanguageDataset but read by LAMDataset.
OXE_LAM_LAYOUT = {
    "egodex": {"video_layout": "flat_mp4", "lam_subdir": "latent_actions_skip_1", "max_episodes": 101_000},
    "bridge": {"video_layout": "folder_single", "cam_name": "rgb"},
    "fractal": {"video_layout": "folder_single", "cam_name": "0", "lam_subdir": "latent_actions_skip_1"},
    "droid": {"video_layout": "folder_stacked", "stacking_mode": "three_view", "wrist_cam": "2", "left_cam": "0", "right_cam": "1"},
    "bc_z": {"video_layout": "folder_single", "cam_name": "0", "lam_subdir": "latent_actions_skip_1"},
    "fmb": {"video_layout": "folder_stacked", "stacking_mode": "three_view", "wrist_cam": "4", "left_cam": "0", "right_cam": "2", "lam_subdir": "latent_actions_skip_1"},
    "taco_play": {"video_layout": "folder_single", "cam_name": "3"},
    "furniture_bench": {"video_layout": "folder_stacked", "stacking_mode": "two_view", "left_cam": "1", "right_cam": "0", "lam_subdir": "latent_actions_skip_1"},
    "austin_sailor": {"video_layout": "folder_stacked", "stacking_mode": "two_view", "left_cam": "1", "right_cam": "0"},
    "berkeley_autolab_ur5": {"video_layout": "folder_stacked", "stacking_mode": "two_view", "left_cam": "0", "right_cam": "1"},
    "stanford_hydra": {"video_layout": "folder_stacked", "stacking_mode": "two_view", "left_cam": "1", "right_cam": "0"},
    "utaustin_mutex": {"video_layout": "folder_stacked", "stacking_mode": "two_view", "left_cam": "1", "right_cam": "0"},
}


def get_embodiment_config(name: str) -> EmbodimentConfig:
    """Look up an `EmbodimentConfig` by name, raising a clear error if unregistered."""
    if name not in OXE_CATALOG:
        raise KeyError(f"Unknown embodiment/dataset '{name}'. Registered: {sorted(OXE_CATALOG)}")
    return OXE_CATALOG[name]