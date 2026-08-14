"""Embodiment dataset registry and the `build_dataset` training-time dispatcher."""

from clap.data.ee import EEDataset, build_oxe_ee_dataset
from clap.data.lam import LAMDataset, build_oxe_lam_dataset
from clap.data.language import LanguageDataset, build_oxe_language_dataset
from clap.data.oxe_catalog import OXE_CATALOG

# Which dataset class backs each conditioning value. Multiple keys can point
# at the same class (see EEDataset's docstring) when the only difference is
# which EmbodimentConfig entry gets selected.
EMBODIMENTS = {
    "ee": EEDataset,
    "bridge": EEDataset,
    "bimanual_yam": EEDataset,
    "g1_humanoid": EEDataset,
    "lam": LAMDataset,
    "language": LanguageDataset,
}

# conditioning values that build a cross-embodiment mix rather than a single dataset.
_MIX_CONDITIONINGS = {"ee", "lam", "language"}


def build_dataset(data_config, model_config, mode, egodex_lam_subdir=None, lam_subdir_override=None):
    """Build the train/val dataset for `data_config.conditioning`.

    "ee"/"lam"/"language" build their cross-embodiment mixes, UNLESS
    `data_config.single_dataset` is set (then that one dataset is built with
    the LAM/language class instead — a genuine LAM-latent or per-step-caption
    single-dataset run, not the usual post-train convention). Any other
    `conditioning` value naming a single `OXE_CATALOG` entry (e.g. "droid",
    "bridge", "bimanual_yam", "g1_humanoid") builds that one dataset as an
    `EEDataset` — this is the normal way post-training/adaptation runs target
    a single embodiment with EE actions, regardless of which family the
    checkpoint being finetuned was pretrained with.

    Args:
        data_config: `DataConfig` — paths, image size, conditioning.
        model_config: `CLAPModelConfig` — only `num_history`/`num_frames` are read.
        egodex_lam_subdir / lam_subdir_override: Only meaningful when
            conditioning="lam" and single_dataset is unset; see `LAMDataset`.
    """
    cond = data_config.conditioning
    single = data_config.single_dataset
    common = dict(
        num_history=model_config.num_history,
        num_frames=model_config.num_frames,
        video_size=data_config.video_size,
        mode=mode,
        debug=data_config.debug_dataset,
    )

    if cond == "ee":  # cross-embodiment EE mix, always multi-dataset
        return build_oxe_ee_dataset(
            oxe_base_path=data_config.oxe_base_path, meta_info_path=data_config.dataset_meta_info_path, **common,
        )
    if cond == "lam":
        if single:  # single LAM-latent dataset run
            return LAMDataset(
                dataset_name=single, oxe_base_path=data_config.oxe_base_path,
                oxe_lam_root=data_config.oxe_lam_root, oxe_lam_subdir=data_config.oxe_lam_subdir, **common,
            )
        return build_oxe_lam_dataset(  # cross-embodiment LAM mix
            oxe_base_path=data_config.oxe_base_path, oxe_lam_root=data_config.oxe_lam_root,
            oxe_lam_subdir=data_config.oxe_lam_subdir, egodex_lam_subdir=egodex_lam_subdir,
            lam_subdir_override=lam_subdir_override, **common,
        )
    if cond == "language":
        if single:  # single per-step-caption dataset run
            return LanguageDataset(
                dataset_name=single, oxe_base_path=data_config.oxe_base_path,
                meta_info_path=data_config.dataset_meta_info_path,
                action_caption_mode=data_config.action_caption_mode, **common,
            )
        return build_oxe_language_dataset(  # cross-embodiment language mix
            oxe_base_path=data_config.oxe_base_path, meta_info_path=data_config.dataset_meta_info_path,
            action_caption_mode=data_config.action_caption_mode, **common,
        )
    if cond in OXE_CATALOG:  # post-train/adaptation: one named embodiment, EE actions
        return EEDataset(
            dataset_name=cond, oxe_base_path=data_config.oxe_base_path,
            meta_info_path=data_config.dataset_meta_info_path, **common,
        )
    raise ValueError(f"Unknown conditioning '{cond}'. Choose from {_MIX_CONDITIONINGS | set(OXE_CATALOG)}.")


__all__ = [
    "EMBODIMENTS",
    "EEDataset",
    "LAMDataset",
    "LanguageDataset",
    "build_dataset",
    "build_oxe_ee_dataset",
    "build_oxe_lam_dataset",
    "build_oxe_language_dataset",
]