"""Checkpoint registry for `clap-eval`: maps a short experiment name to how to load and run it.

Organized into 5 categories: baselines (single-embodiment training),
cross-embodiment models (6 conditioning variants), post-train DROID (6),
post-train Bridge (6), and novel-embodiment adaptation (bimanual YAM, G1
humanoid). Checkpoint-root paths are relative to `PathConfig.checkpoint_root`,
resolved at eval time — nothing here hardcodes an absolute path.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ExperimentSpec:
    """One entry in the registry.

    Attributes:
        ckpts_root: Directory holding `checkpoint-<N>.pt`/`last.pt`, relative
            to `PathConfig.checkpoint_root` (or absolute).
        family: Which `clap.data.rollout_loaders` entry + `CLAPRolloutAgent`
            conditioning family this checkpoint expects.
        default_datasets: Datasets to evaluate when the caller doesn't specify any.
        action_caption_mode: Only meaningful when family="language".
        lam_subdir_override: Only meaningful when family="lam" (selects which
            latent-action extractor run to read, e.g. a "dreamdojo" variant).
    """

    ckpts_root: str
    family: str
    default_datasets: List[str]
    action_dim: int
    conditioning: str
    action_caption_mode: str = "absolute"
    lam_subdir_override: Optional[str] = None


_OXE_EE_DATASETS = ["fractal", "fmb", "bc_z", "taco_play", "furniture_bench", "bridge", "droid"]
_OXE_LAM_DATASETS = ["egodex", "bridge", "fractal", "droid", "bc_z", "fmb", "taco_play", "furniture_bench"]
_OXE_LANG_DATASETS = ["bridge", "fractal", "droid", "bc_z", "fmb", "taco_play", "furniture_bench"]

# 0. Baselines: single-embodiment models with no cross-embodiment modeling.
_BASELINES = {
    "baseline_droid": ExperimentSpec("baseline_droid", "ee", ["droid"], 7, "ee"),
    "baseline_bridge": ExperimentSpec("baseline_bridge", "ee", ["bridge"], 7, "ee"),
}

# 1. Cross-embodiment models: same OXE mix, 6 conditioning variants.
_CROSS_EMBODIMENT = {
    "cross_embodiment_oxe_ee": ExperimentSpec("cross_embodiment_oxe_ee", "ee", _OXE_EE_DATASETS, 7, "ee"),
    "cross_embodiment_oxe_lam_clap": ExperimentSpec("cross_embodiment_oxe_lam_clap", "lam", _OXE_LAM_DATASETS, 32, "lam"),
    "cross_embodiment_oxe_lam_dreamdojo": ExperimentSpec(
        "cross_embodiment_oxe_lam_dreamdojo", "lam", _OXE_LAM_DATASETS, 32, "lam",
        lam_subdir_override="dreamdojo_latent_actions_skip_1",
    ),
    "cross_embodiment_oxe_curriculum_lam_ee": ExperimentSpec("cross_embodiment_oxe_curriculum_lam_ee", "ee", _OXE_EE_DATASETS, 7, "ee"),
    "cross_embodiment_oxe_language_absolute": ExperimentSpec("cross_embodiment_oxe_language_absolute", "language", _OXE_LANG_DATASETS, 7, "language"),
    "cross_embodiment_oxe_language_relative": ExperimentSpec(
        "cross_embodiment_oxe_language_relative", "language", _OXE_LANG_DATASETS, 7, "language", action_caption_mode="relative",
    ),
}

# Shared across post-train DROID/Bridge: each finetunes from the same 6 cross-embodiment
# bases (name matches the corresponding _CROSS_EMBODIMENT suffix, minus the "cross_embodiment_"
# prefix), onto droid alone or bridge alone respectively. Deriving both the dict key and
# ckpts_root from one shared list (instead of separate (name, suffix) pairs) guarantees they
# can't drift apart.
_POST_TRAIN_BASES = ["oxe_ee", "oxe_lam_clap", "oxe_lam_dreamdojo", "oxe_curriculum_lam_ee", "oxe_language_absolute", "oxe_language_relative"]

# 2. Post-train DROID: 6 variants finetuned onto droid alone.
_POST_TRAIN_DROID = {
    f"post_train_{base}_droid": ExperimentSpec(f"post_train_{base}_droid", "ee", ["droid"], 7, "ee")
    for base in _POST_TRAIN_BASES
}

# 3. Post-train Bridge: 6 variants finetuned onto bridge alone. The bridge-only reference
# point for this category is _BASELINES["baseline_bridge"] above — not duplicated here.
_POST_TRAIN_BRIDGE = {
    f"post_train_{base}_bridge": ExperimentSpec(f"post_train_{base}_bridge", "ee", ["bridge"], 7, "ee")
    for base in _POST_TRAIN_BASES
}

# 4. Novel-embodiment adaptation.
_ADAPTATION = {
    "adapt_bimanual_yam": ExperimentSpec("adapt_bimanual_yam", "ee", ["bimanual_yam"], 14, "ee"),
    "adapt_g1_humanoid": ExperimentSpec("adapt_g1_humanoid", "ee", ["g1_humanoid"], 26, "ee"),
}

EXPERIMENTS: Dict[str, ExperimentSpec] = {**_BASELINES, **_CROSS_EMBODIMENT, **_POST_TRAIN_DROID, **_POST_TRAIN_BRIDGE, **_ADAPTATION}

_CATEGORIES = {
    "baselines": _BASELINES, "cross_embodiment": _CROSS_EMBODIMENT,
    "droid": _POST_TRAIN_DROID, "bridge": _POST_TRAIN_BRIDGE, "adaptation": _ADAPTATION,
}

# Short-form aliases for EXPERIMENTS keys, for CLI/env-var ergonomics (e.g.
# CKPT_NAME=clap-ee instead of the full cross_embodiment_oxe_ee). "clap-*"
# names the flagship cross-embodiment tier by conditioning (ee/lam/lang);
# "clap-curr" is the curriculum LAM->EE variant; -droid/-bridge suffixes are
# the post-train counterparts; adapt-* are the novel-embodiment targets.
# Resolve through resolve_ckpt_name() rather than indexing this dict directly.
CKPT_ALIASES: Dict[str, str] = {
    # baselines
    "baseline-droid": "baseline_droid",
    "baseline-bridge": "baseline_bridge",

    # cross-embodiment (flagship tier -- these are the defaults getting_started/*.sh point at)
    "clap-ee": "cross_embodiment_oxe_ee",
    "clap-lam": "cross_embodiment_oxe_lam_clap",
    "clap-curr": "cross_embodiment_oxe_curriculum_lam_ee",
    "clap-lang": "cross_embodiment_oxe_language_relative",
    "clap-lang-abs": "cross_embodiment_oxe_language_absolute",
    "clap-lam-dreamdojo": "cross_embodiment_oxe_lam_dreamdojo",

    # post-train DROID
    "clap-ee-droid": "post_train_oxe_ee_droid",
    "clap-lam-droid": "post_train_oxe_lam_clap_droid",
    "clap-curr-droid": "post_train_oxe_curriculum_lam_ee_droid",
    "clap-lang-droid": "post_train_oxe_language_relative_droid",
    "clap-lang-abs-droid": "post_train_oxe_language_absolute_droid",
    "clap-lam-dreamdojo-droid": "post_train_oxe_lam_dreamdojo_droid",

    # post-train Bridge
    "clap-ee-bridge": "post_train_oxe_ee_bridge",
    "clap-lam-bridge": "post_train_oxe_lam_clap_bridge",
    "clap-curr-bridge": "post_train_oxe_curriculum_lam_ee_bridge",
    "clap-lang-bridge": "post_train_oxe_language_relative_bridge",
    "clap-lang-abs-bridge": "post_train_oxe_language_absolute_bridge",
    "clap-lam-dreamdojo-bridge": "post_train_oxe_lam_dreamdojo_bridge",

    # novel-embodiment adaptation
    "adapt-yam": "adapt_bimanual_yam",
    "adapt-g1": "adapt_g1_humanoid",
}


def resolve_ckpt_name(name: str) -> str:
    """Resolve a CKPT_ALIASES short form (or an already-full EXPERIMENTS name,
    passed through unchanged) to a real EXPERIMENTS key."""
    return CKPT_ALIASES.get(name, name)


def display_ckpt_name(name: str) -> str:
    """Short display label for a resolved (full) experiment name, e.g. "CLAP-EE" for
    cross_embodiment_oxe_ee -- reverse lookup against CKPT_ALIASES (uppercased), falling
    back to the raw name itself (uppercased) if no alias maps to it. Purely cosmetic,
    for the teleop/deploy live-view pages -- not meant to round-trip through resolve_ckpt_name."""
    for alias, full in CKPT_ALIASES.items():
        if full == name:
            return alias.upper()
    return name.upper()


def get_experiment(name: str) -> ExperimentSpec:
    name = resolve_ckpt_name(name)
    if name not in EXPERIMENTS:
        categories = "\n".join(f"{cat}: {sorted(entries)}" for cat, entries in _CATEGORIES.items())
        raise KeyError(f"Unknown experiment {name!r}.\n{categories}")
    return EXPERIMENTS[name]


def list_experiments(category: Optional[str] = None) -> List[str]:
    if category is None:
        return sorted(EXPERIMENTS)  # every registered experiment name
    if category not in _CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; choose from {list(_CATEGORIES)}")
    return sorted(_CATEGORIES[category])  # names within just this category