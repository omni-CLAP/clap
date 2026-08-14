"""Per-dataset eval-only defaults: rollout length caps and static-prefix trimming.

Camera layout (stacking_mode) already lives in `clap.data.oxe_catalog` and its
pixel-space view boxes come from `clap.data.camera_stacking.view_slices_for_stacking`
— this module only adds the handful of fields that are eval-specific, layered
on top of that one shared registry rather than duplicating it.
"""

from dataclasses import dataclass, field
from typing import Optional

from clap.data.oxe_catalog import OXE_CATALOG


@dataclass
class DatasetSpec:
    """Eval-only per-dataset defaults, applied unless the caller explicitly overrides them.

    Attributes:
        save_fps: Frame rate for written comparison videos (independent of the
            dataset's own native/replay rate).
        max_chunks: Cap on autoregressive chunks per episode; 0 = unlimited.
        trim_static_prefix / skip_first_n_frames: See `CLAPRolloutAgent`'s
            corresponding parameters.
    """

    name: str
    stacking_mode: Optional[str] = None
    save_fps: int = 4
    fps_downsample_ratio: int = 1
    max_chunks: int = 0
    trim_static_prefix: bool = False
    skip_first_n_frames: int = 0


# Per-dataset overrides where the eval defaults above aren't appropriate.
_OVERRIDES = {
    "stanford_hydra": {"skip_first_n_frames": 5},  # first few frames are a fixed camera-settle period
}

DATASET_SPECS = {
    name: DatasetSpec(
        name=name, stacking_mode=cfg.stacking_mode, fps_downsample_ratio=cfg.fps_downsample_ratio,
        **_OVERRIDES.get(name, {}),
    )
    for name, cfg in OXE_CATALOG.items()
}


def get_spec(name: str) -> DatasetSpec:
    """Look up a `DatasetSpec`, falling back to all-default values for an unregistered name."""
    return DATASET_SPECS.get(name, DatasetSpec(name=name))
