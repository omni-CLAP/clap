"""Named test-set registry: exact, deterministic episode lists shared across every conditioning family.

Each entry names which dataset(s) to draw episodes from and how many, so
`clap-eval --test-set <name>` always evaluates the same videos regardless of
which checkpoint/family is being scored. `clap.eval.build_test_sets` turns an
entry here into a `clap/eval/test_sets_cache/<name>.json` snapshot; `clap.eval.evaluate`
reads that snapshot, never this registry directly.
"""

from typing import Dict, List, Tuple

TestSetSpec = Dict
"""
{
    "split": str,
    "selection": List[Tuple[str, int]],   # (dataset_name, num_episodes)
    "description": str,
    "strict_eligibility": bool,  # default False; see build_test_sets._filter_eligible
    "t_required": int,          # default 18 (= num_history + num_frames)
}
"""

TEST_SETS: Dict[str, TestSetSpec] = {
    "bridge_100": {
        "split": "val",
        "selection": [("bridge", 100)],
        "description": "First 100 bridge val episodes with both EE state and LAM latents.",
    },
    "droid_100": {
        "split": "val",
        "selection": [("droid", 100)],
        "description": "First 100 droid val episodes with both EE state and LAM latents.",
    },
    "oxe_mix_100": {
        "split": "val",
        # Equal-ish split across the 7 OXE EE datasets, strict eligibility on. fractal
        # shrinks 15->14 to total 99 (matches bridge_100/droid_100's episode count).
        "selection": [
            ("bridge", 15),
            ("droid", 15),
            ("fractal", 14),
            ("fmb", 15),
            ("bc_z", 15),
            ("taco_play", 15),
            ("furniture_bench", 10),
        ],
        "strict_eligibility": True,
        "t_required": 18,
        "description": "Equal-ish OXE mix totalling 99 episodes, filtered by strict "
                       "cross-family chunk eligibility (T_required=18) so every key "
                       "survives rollout-time chunk-length checks under any family.",
    },
    # Held-out generalization targets: unseen during cross-embodiment modeling, used to
    # probe transfer. Data lives under a separate root (CLAP_OXE_HELD_OUT_PATH, not
    # CLAP_OXE_BASE_PATH) -- point --oxe-base-path there when rebuilding these via
    # clap-build-test-sets, and swap CLAP_OXE_BASE_PATH for the same when evaluating.
    "austin_sailor_100": {
        "split": "train",
        "selection": [("austin_sailor", 100)],
        "description": "100 austin_sailor train episodes stride-sampled across the full "
                       "train set (228 episodes total, episode range 0-237).",
    },
    "berkeley_autolab_ur5_100": {
        "split": "train",
        "selection": [("berkeley_autolab_ur5", 100)],
        "description": "100 berkeley_autolab_ur5 train episodes stride-sampled across the "
                       "full train set (807 episodes total, episode range 0-886).",
    },
    "stanford_hydra_100": {
        "split": "train",
        "selection": [("stanford_hydra", 100)],
        "description": "100 stanford_hydra train episodes stride-sampled across the full "
                       "train set (542 episodes total, episode range 0-564).",
    },
    "utaustin_mutex_100": {
        "split": "train",
        "selection": [("utaustin_mutex", 100)],
        "description": "100 utaustin_mutex train episodes stride-sampled across the full "
                       "train set (1425 episodes total, episode range 0-1484).",
    },
    # Novel-embodiment adaptation targets: single-embodiment, EE-only (no lam/language
    # trees), so every val episode qualifies -- no cross-family intersection needed.
    "bimanual_yam_val": {
        "split": "val",
        "selection": [("bimanual_yam", 34)],
        "description": "All 34 bimanual_yam val episodes.",
    },
    "g1_humanoid_val": {
        "split": "val",
        "selection": [("g1_humanoid", 104)],
        "description": "All 104 g1_humanoid val episodes.",
    },
}


def get_test_set(name: str) -> TestSetSpec:
    """Look up a registered `TestSetSpec` by name, raising if it isn't registered."""
    if name not in TEST_SETS:
        raise KeyError(f"Unknown test set {name!r}; known: {sorted(TEST_SETS)}")
    return TEST_SETS[name]


def list_test_sets() -> List[str]:
    """All registered test-set names, sorted."""
    return sorted(TEST_SETS)
