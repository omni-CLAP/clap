from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EmbodimentConfig:
    """Per-dataset/embodiment config: camera layout + action representation.

    One instance per entry in `clap.data.oxe_catalog`; `EEDataset` reads these
    fields directly rather than each embodiment owning its own config class —
    `bridge` and `bimanual_yam` are just distinct `EmbodimentConfig` entries
    feeding the same `EEDataset`, not separate classes.

    Attributes:
        action_mode: "ee7" (state[:6]+gripper, the cartesian-EE convention) |
            "joint14"/"joint26" (raw state array as-is, for joint-space embodiments).
        stacking_mode: "four_view" | "three_view" | "two_view" | None (single
            camera, tiled into 3 identical slots). Selects the camera-stacking
            function in `clap.data.camera_stacking.STACKERS`.
        cam_id: Single-camera name/id, used when stacking_mode is None.
        left_view_id / right_view_id / wrist_view_id: Camera ids for
            two_view/three_view stacking (see `camera_stacking` for slot order).
        cam_ids: Camera ids for four_view stacking, in slot order
            [right_high, left_high, right_wrist, left_wrist].
        hand_type: Optional hand-hardware variant (e.g. "brainco" for
            g1_humanoid); not currently used to change behavior, only recorded.
        fps_downsample_ratio: Extra frame-skip beyond the dataset's native rate
            (e.g. 3 for 30fps sources being treated as ~10fps).
        annotation_subdir: Name of the per-dataset annotation directory under
            `<oxe_base_path>/<name>/`. Defaults to "annotation" for every
            dataset; a deployment whose own copy of a given dataset's
            annotations live under a different name (e.g. a second/later
            annotation pass) overrides this per-entry in `oxe_catalog.py`, or
            via a `CLAP_<NAME>_ANNOTATION_SUBDIR` env var read there — never
            hardcode a site-specific directory name as the package default.
    """

    name: str
    action_mode: str = "ee7"
    stacking_mode: Optional[str] = None
    cam_id: Optional[str] = None
    left_view_id: Optional[int] = None
    right_view_id: Optional[int] = None
    wrist_view_id: Optional[int] = None
    cam_ids: Optional[List[int]] = None
    hand_type: Optional[str] = None
    fps_downsample_ratio: int = 1
    annotation_subdir: str = "annotation"


@dataclass
class DataConfig:
    """Dataset-loading configuration, independent of model architecture.

    Attributes:
        conditioning: Selects which `clap.data.EMBODIMENTS` class to build —
            "ee"/"lam"/"language" (the cross-embodiment mixes) or a single-dataset
            key ("bridge", "bimanual_yam", "g1_humanoid", "droid", ...), which
            always builds an `EEDataset` for that one dataset. Post-training a
            LAM/language-pretrained checkpoint onto droid/bridge alone still uses
            plain conditioning="droid"/"bridge" (EE actions) — that's the normal
            post-train convention here, not an exception needing `single_dataset`.
        single_dataset: Only meaningful when conditioning is "lam"/"language"
            (conditioning="ee" already has this via the single-dataset keys
            above). When set, builds THAT one dataset with the LAM/language
            class instead of the full cross-embodiment mix — for a genuine
            LAM-latent or per-step-caption single-dataset training run, distinct
            from the ee7-action post-train convention.
        oxe_base_path / dataset_meta_info_path: Roots for video/annotation data
            and per-dataset stat.json normalization bounds, respectively.
        oxe_lam_root / oxe_lam_subdir: Only read when conditioning="lam".
            oxe_lam_subdir's default ("latent_actions") is a placeholder name —
            point it at a real extraction run via `CLAP_OXE_LAM_SUBDIR` or the
            experiment YAML, not by editing this default.
        egodex_lam_subdir: Only read when conditioning="lam". Overrides
            oxe_lam_subdir for egodex specifically (different latent-action
            extractor runs use different subdir names). When it matches
            `clap.data.lam.EGODEX_ONLY_LAM_SUBDIR_MARKERS` (e.g. contains
            "dreamdojo"), the LAM mix drops every other dataset and trains on
            egodex alone — this is how `cross_embodiment_oxe_lam_dreamdojo` differs
            from `cross_embodiment_oxe_lam_clap`, not a per-dataset subdir swap.
        action_caption_mode: Only read when conditioning="language"; see
            `clap.data.action_caption`.
        val_sample_ids: Which validation samples to render during training,
            keyed by sub-dataset name (local index within that sub-dataset).
        debug_dataset: Truncates each sub-dataset to 8 episodes for fast iteration.
    """

    conditioning: str = "ee"
    single_dataset: Optional[str] = None
    oxe_base_path: str = ""
    dataset_meta_info_path: str = "dataset_meta_info"
    oxe_lam_root: str = ""
    oxe_lam_subdir: str = "latent_actions"
    egodex_lam_subdir: Optional[str] = None
    height: int = 576  # stacked-frame height (3 camera slots x 192, or 4 x 192 for four_view)
    width: int = 320
    num_workers: int = 4  # dataloader worker processes
    debug_dataset: bool = False
    action_caption_mode: str = "relative"
    val_sample_ids: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def video_size(self):
        """(height, width), the shape dataset loaders resize stacked frames to."""
        return (self.height, self.width)
