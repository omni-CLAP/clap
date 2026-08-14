"""Verify that different experiments evaluated the exact same episode IDs per test set.

Reads `info/*.json` under each `<eval-outputs>/<experiment>/<test_set>/<iter>/`
(handling the optional one-level `iter_*/<uuid>/info` nesting some runs
produce), normalizes episode IDs via `clap.data.lam._normalize_ep_key` (so
LAM's "episode_000142" compares equal to EE/language's "142"), and prints
per-(test_set, dataset) whether the ID sets are identical across experiments
— with a missing/extra sample list when they diverge.

Usage:
    clap-eval-check-episode-sets --eval-outputs eval_outputs \\
        --experiments cross_embodiment_oxe_ee:iter_last cross_embodiment_oxe_lang_absolute:iter_last \\
        --test-sets bridge_100 droid_100 oxe_mix_100
"""

import argparse
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from clap.data.lam import _normalize_ep_key
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_DEFAULT_TEST_SETS = ["bridge_100", "droid_100", "oxe_mix_100"]


def _parse_runs(specs: List[str]) -> Dict[str, str]:
    """['exp:iter', ...] -> {exp: iter}, preserving CLI order."""
    out: Dict[str, str] = {}
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"--experiments entries must be 'name:iter'; got {s!r}")
        name, it = s.split(":", 1)  # split on first ':' only
        out[name] = it
    return out


def _info_dirs(root: Path, exp: str, ts: str, it: str) -> List[Path]:
    """Some runs store info/ directly under iter_*; some nest one uuid level (iter_*/<uuid>/info)."""
    direct = root / exp / ts / it / "info"
    if direct.is_dir():
        return [direct]  # common layout: info/ directly under iter_*
    return [Path(p) for p in sorted(glob.glob(str(root / exp / ts / it / "*" / "info")))]  # fallback: one uuid level of nesting


def _delivered(root: Path, exp: str, ts: str, it: str) -> Dict[str, Set[str]]:
    """{dataset_name: {normalized_episode_key, ...}} for one experiment+test set."""
    out: Dict[str, Set[str]] = defaultdict(set)
    for d in _info_dirs(root, exp, ts, it):
        for p in sorted(d.glob("*.json")):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue  # skip unreadable/corrupt info file
            ep = j.get("episode_id")
            ds = j.get("dataset_name")
            if ep is not None and ds:
                out[ds].add(_normalize_ep_key(str(ep)))  # normalize so LAM/EE/language ID formats compare equal
    return out


def _report_for_test_set(ts: str, runs: Dict[str, str], root: Path, missing_sample: int) -> bool:
    """Print a table for one test set; returns True iff every dataset matches across all experiments."""
    logger.info(f"\n🔍 === {ts} ===")
    sets = {exp: _delivered(root, exp, ts, it) for exp, it in runs.items()}  # per-experiment delivered episode IDs
    available = {e: s for e, s in sets.items() if any(s.values())}  # drop experiments with no info dirs for this test set
    if not available:
        logger.warning("⚠️   no info dirs found in any of the requested experiments")
        return True

    all_ds = sorted({ds for s in available.values() for ds in s})
    logger.info(f"📊   {'dataset':18s}" + "".join(f"  {e:>26s}" for e in available))
    for ds in all_ds:
        logger.info(f"📊   {ds:18s}" + "".join(f"  {len(available[e].get(ds, set())):>26d}" for e in available))  # episode count per experiment/dataset

    logger.info("🔍   -- ID set match across experiments listed above --")
    all_match = True
    for ds in all_ds:
        per_exp = {e: available[e].get(ds, set()) for e in available}  # {experiment: episode_id set} for this dataset
        union = set().union(*per_exp.values())
        inter = set.intersection(*per_exp.values())
        if all(per_exp[e] == inter for e in per_exp) and inter == union:  # every experiment's set equals the intersection AND the union — i.e. all identical
            logger.info(f"✅     {ds:18s} IDENTICAL across all ({len(inter)} eps)")
        else:
            all_match = False
            logger.warning(f"⚠️     {ds:18s} DIFFER: |union|={len(union)}  |intersection|={len(inter)}")
            for e in per_exp:
                miss = sorted(union - per_exp[e])[:missing_sample]  # episodes other experiments have but this one lacks
                extra = sorted(per_exp[e] - inter)[:missing_sample]  # episodes this experiment has that aren't shared by all
                logger.warning(f"⚠️       {e:30s}  has {len(per_exp[e]):3d}  missing[:{missing_sample}]={miss}  extra[:{missing_sample}]={extra}")
    if all_match:
        logger.info("✅   ALL DATASETS MATCH ACROSS ALL AVAILABLE EXPERIMENTS")
    return all_match


def cli():
    p = argparse.ArgumentParser(description=__doc__)  # parses which experiments/test sets to cross-check
    p.add_argument("--eval-outputs", type=Path, default=Path("eval_outputs"), help="Root containing <experiment>/<test_set>/<iter>/info/*.json.")
    p.add_argument("--experiments", nargs="+", required=True, help="exp:iter pairs, e.g. cross_embodiment_oxe_ee:iter_last.")
    p.add_argument("--test-sets", nargs="+", default=_DEFAULT_TEST_SETS)
    p.add_argument("--missing-sample", type=int, default=5, help="How many missing/extra ids to print per experiment on divergence.")
    return p.parse_args()


def main() -> int:
    args = cli()
    runs = _parse_runs(args.experiments)
    if not args.eval_outputs.is_dir():
        raise SystemExit(f"eval-outputs root not found: {args.eval_outputs}")
    logger.info(f"🚀 eval_outputs root: {args.eval_outputs}\nexperiments: {runs}\ntest sets: {args.test_sets}")

    # Not `all(...)` over a generator: that would short-circuit and skip printing
    # reports for test sets after the first mismatch.
    results = [_report_for_test_set(ts, runs, args.eval_outputs, args.missing_sample) for ts in args.test_sets]
    all_ok = all(results)
    (logger.info if all_ok else logger.warning)(f"\n{'✅' if all_ok else '⚠️'} OVERALL: {'MATCH' if all_ok else 'MISMATCH'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
