"""Build/refresh test-set JSON snapshots: `clap-build-test-sets [--names ...]`.

For each (dataset, num_episodes) pair in a `clap.eval.test_sets` entry:
  1. Enumerate canonical episode keys under the EE-state annotation dir and
     the LAM latent-action dir (language conditioning needs no separate check
     — its per-step captions are built at runtime from the same state array
     EE uses).
  2. Intersect the two sets — only episodes usable by both families stay.
  3. Sort the intersection (numeric-first) and take the first `num_episodes`
     (or, if `strict_eligibility`, the first `num_episodes` that also pass
     `clap.eval.episode_eligibility.chunk_eligible`).
  4. Write `clap/eval/test_sets_cache/<name>.json` with the chosen keys per
     dataset, so every experiment pins to the exact same episodes.

Re-run whenever the underlying data changes; the JSON is a content-addressable
snapshot meant to be committed alongside code.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Set

from clap.data.lam import _normalize_ep_key
from clap.data.oxe_catalog import get_embodiment_config
from clap.eval.episode_eligibility import chunk_eligible, compute_data_lengths
from clap.eval.test_sets import TEST_SETS, get_test_set, list_test_sets
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_sets_cache")


def _ann_keys(base: Path, ds: str, split: str) -> Set[str]:
    d = base / ds / get_embodiment_config(ds).annotation_subdir / split  # dataset's EE-state annotation dir
    if not d.is_dir():
        return set()  # dataset/split not present on disk
    return {_normalize_ep_key(f[: -len(".json")]) for f in os.listdir(d) if f.endswith(".json")}  # strip .json, normalize episode key


def _lam_keys(base: Path, ds: str, split: str, lam_subdir: str) -> Set[str]:
    d = base / ds / lam_subdir / split  # dataset's latent-action dir
    if not d.is_dir():
        return set()  # dataset/split not present on disk
    out = set()
    for root, _dirs, files in os.walk(d):
        if "latent_actions.npy" in files:  # this leaf dir is one episode
            out.add(_normalize_ep_key(os.path.relpath(root, d)))
    return out


def _sort_keys(keys: Set[str]) -> List[str]:
    """Numeric keys sort numerically (e.g. "53" < "1002") and come first; the rest sort lexicographically."""
    nums, others = [], []
    for k in keys:
        (nums if k.isdigit() else others).append(k)  # split into purely-numeric vs other keys
    return [str(n) for n in sorted(int(x) for x in nums)] + sorted(others)  # numeric-sorted numbers, then lexicographic rest


def _filter_eligible(ds: str, sorted_keys: List[str], oxe_base: Path, split: str, lam_subdir: str, t_required: int, target_n: int) -> List[str]:
    """Walk sorted_keys in order, keeping only chunk-eligible ones, stopping once target_n are found."""
    d = get_embodiment_config(ds).fps_downsample_ratio  # this dataset's FPS downsample ratio, needed for the eligibility check

    def check(k):
        lengths = compute_data_lengths(ds, k, str(oxe_base), str(oxe_base), lam_subdir, split)  # per-source sequence lengths for this episode
        return k, chunk_eligible(lengths, d, t_required)

    eligible: List[str] = []
    i = 0
    batch_size = max(64, target_n * 2)  # scan in batches so we stop early once target_n is reached
    with ThreadPoolExecutor(8) as ex:  # eligibility checks hit disk, so parallelize them
        while i < len(sorted_keys) and len(eligible) < target_n:
            batch = sorted_keys[i : i + batch_size]  # next slice of keys to check
            i += batch_size
            for k, ok in (f.result() for f in [ex.submit(check, k) for k in batch]):  # submit batch, then collect in order
                if ok:
                    eligible.append(k)
                    if len(eligible) >= target_n:
                        break  # stop once target reached, even mid-batch
    return eligible


def build_one(name: str, oxe_base: Path, lam_subdir: str, out_dir: Path, no_lam_intersection: bool = False) -> Dict:
    spec = get_test_set(name)  # test-set definition (selection, split, eligibility settings)
    split = spec["split"]
    strict = bool(spec.get("strict_eligibility", False))
    t_required = int(spec.get("t_required", 18))
    out: Dict = {"name": name, "split": split, "description": spec.get("description", ""), "datasets": {}}
    logger.info(f"\n🎬 === {name}  (split={split}, strict={strict}, T_req={t_required}) ===")

    for ds, n in spec["selection"]:
        ann_keys = _ann_keys(oxe_base, ds, split)  # episodes with EE-state annotations
        if no_lam_intersection:
            # Skip requiring LAM latent-action data entirely -- for datasets that never
            # compute it (e.g. bimanual_yam/g1_humanoid, joint-space-only embodiments),
            # ann_keys & lam_keys would otherwise always be empty.
            common, lam_note = ann_keys, "skipped"
        else:
            lam_keys = _lam_keys(oxe_base, ds, split, lam_subdir)  # episodes with latent actions
            common, lam_note = ann_keys & lam_keys, str(len(lam_keys))  # only episodes usable by both families
        sorted_common = _sort_keys(common)

        if strict:
            chosen = _filter_eligible(ds, sorted_common, oxe_base, split, lam_subdir, t_required, n)  # first n that also pass chunk eligibility
            tag = f"strict (T_req={t_required})"
        else:
            chosen = sorted_common[:n]  # first n by sort order, no eligibility filtering
            tag = "first-N"

        status = "warn" if len(chosen) < n else "ok"  # warn if we couldn't find n eligible episodes
        logger.info(f"📊   [{status}] {ds}: {len(chosen)}/{n} keys (state={len(ann_keys)} lam={lam_note} "
              f"intersection={len(common)}, {tag})")
        out["datasets"][ds] = chosen

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(out, indent=2))  # persist the snapshot so every experiment pins to the same episodes
    total = sum(len(v) for v in out["datasets"].values())
    logger.info(f"💾   -> {out_path}  ({total} episodes total)")
    return out


def cli():
    p = argparse.ArgumentParser(description=__doc__)  # parses which test sets to (re)build and where to read/write data
    p.add_argument("--names", nargs="*", default=None, help="Test sets to build (default: all in clap.eval.test_sets).")
    p.add_argument("--oxe-base-path", type=Path, default=Path(os.environ.get("CLAP_OXE_BASE_PATH", "")))
    p.add_argument("--oxe-lam-subdir", default=os.environ.get("CLAP_OXE_LAM_SUBDIR", "latent_actions"))
    p.add_argument("--out-dir", type=Path, default=Path(_CACHE_DIR))
    p.add_argument("--no-lam-intersection", action="store_true",
                    help="Select episodes from EE-state annotations alone, skipping the "
                         "intersection with LAM latent-action data. Needed for datasets that "
                         "never compute LAM (e.g. bimanual_yam/g1_humanoid) -- ann_keys & "
                         "lam_keys would otherwise always be empty for them.")
    return p.parse_args()


def main():
    args = cli()
    if not args.oxe_base_path.is_dir():
        raise SystemExit(f"OXE base path not found: {args.oxe_base_path}")

    names = args.names or list_test_sets()  # default to every registered test set
    logger.info(f"🚀 OXE base: {args.oxe_base_path}\nOut dir:  {args.out_dir}\nBuilding: {names}")
    for name in names:
        build_one(name, args.oxe_base_path, args.oxe_lam_subdir, args.out_dir, args.no_lam_intersection)


if __name__ == "__main__":
    main()
