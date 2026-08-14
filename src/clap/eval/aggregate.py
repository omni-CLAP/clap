"""Aggregate one experiment's per-checkpoint eval results into a table + plots: `clap-eval-aggregate`.

Reads `<results-root>/iter_*/{aggregate,per_view_summary}.json`, one iter_*
directory per checkpoint evaluated (however those directories got there —
this module only consumes the standard `clap.eval.evaluate` output layout).
Writes, under `<results-root>/_aggregate`:
    all_checkpoints.csv                one row per (dataset, view, iter)
    plots/<dataset>/fvd.png            FVD vs iter
    plots/<dataset>/<view>/{psnr,ssim,lpips}.png
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_ITER_RE = re.compile(r"^iter_(\d+|last)(?:_(.+))?$")


def _iter_sort_key(name: str):
    m = _ITER_RE.match(name)
    if not m:
        return (10**12, name)  # non-matching names sort last, alphabetically among themselves
    raw = m.group(1)
    return (10**12 if raw == "last" else int(raw), m.group(2) or "")  # "last" sorts after every numeric iter


def _load_iter_dir(d: Path) -> Optional[Dict]:
    per_view_path = d / "per_view_summary.json"
    if not per_view_path.is_file():
        return None  # per_view_summary.json is required; skip dirs that don't have it
    agg_path = d / "aggregate.json"
    return {
        "iter_name": d.name,
        "iter_num": _iter_sort_key(d.name)[0],  # numeric iter for sorting/plotting
        "per_view": json.loads(per_view_path.read_text()),
        "aggregate": json.loads(agg_path.read_text()) if agg_path.is_file() else None,  # optional file
    }


def _discover_iters(results_root: Path) -> List[Path]:
    dirs = [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("iter_")]
    return sorted(dirs, key=lambda p: _iter_sort_key(p.name))


def _plot_metric_vs_iter(iters, vals, title, ylabel, out_path: Path):
    if not iters:
        return  # nothing to plot
    fig, ax = plt.subplots(figsize=(6, 4))  # new figure per metric plot
    ax.plot(list(iters), list(vals), "o-")  # line with point markers
    ax.set_xlabel("training iter")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)  # free figure memory before the next plot


def main():
    p = argparse.ArgumentParser(description=__doc__)  # parses --results-root/--out-dir/--datasets
    p.add_argument("--results-root", type=Path, required=True, help="Directory containing iter_*/ subdirs.")
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <results-root>/_aggregate")
    p.add_argument("--datasets", nargs="*", default=None, help="Restrict to these datasets (default: all found).")
    args = p.parse_args()

    out_dir = args.out_dir or (args.results_root / "_aggregate")
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # load every iter_* dir that has a per_view_summary.json, skipping the rest
    recs = [r for d in _discover_iters(args.results_root) if (r := _load_iter_dir(d)) is not None]
    if not recs:
        logger.warning(f"⚠️ No iter_*/per_view_summary.json found under {args.results_root}.")
        return

    all_ds = {ds for r in recs for ds in r["per_view"]}  # union of datasets seen across all iters
    if args.datasets:
        all_ds &= set(args.datasets)  # restrict to requested subset
    if not all_ds:
        logger.warning("⚠️ No datasets found in summaries.")
        return
    datasets = sorted(all_ds)

    rows = []
    for r in recs:
        for ds, ds_sum in r["per_view"].items():
            if ds not in datasets:
                continue
            for view, m in ds_sum.get("views", {}).items():
                rows.append({
                    "dataset": ds, "view": view, "iter_name": r["iter_name"], "iter": r["iter_num"],
                    "n_episodes": ds_sum["n_episodes"], "psnr": m["psnr_mean"], "ssim": m["ssim_mean"],
                    "lpips": m["lpips_mean"], "fvd": ds_sum.get("fvd") if view == "stacked" else "",  # FVD is dataset-level, only attach it to the "stacked" view row
                })
    rows.sort(key=lambda r: (r["dataset"], r["view"], r["iter"]))

    csv_path = out_dir / "all_checkpoints.csv"
    with open(csv_path, "w") as f:
        f.write("dataset,view,iter_name,iter,n_episodes,psnr,ssim,lpips,fvd\n")
        for r in rows:
            fvd_str = "" if r["fvd"] in ("", None) else f"{r['fvd']:.3f}"  # blank fvd column for non-stacked views
            f.write(f"{r['dataset']},{r['view']},{r['iter_name']},{r['iter']},{r['n_episodes']},"
                    f"{r['psnr']:.4f},{r['ssim']:.4f},{r['lpips']:.4f},{fvd_str}\n")
    logger.info(f"💾 Wrote {csv_path}")

    for ds in datasets:
        ds_recs = [r for r in recs if ds in r["per_view"]]  # iters that evaluated this dataset
        if not ds_recs:
            continue
        ds_plot = plot_dir / ds
        ds_plot.mkdir(parents=True, exist_ok=True)
        iters = [r["iter_num"] for r in ds_recs]

        fvds = [r["per_view"][ds].get("fvd", float("nan")) for r in ds_recs]
        _plot_metric_vs_iter(iters, fvds, f"{ds} — FVD vs iter", "FVD (lower = better)", ds_plot / "fvd.png")

        view_names = sorted({v for r in ds_recs for v in r["per_view"][ds].get("views", {})})  # every camera view seen for this dataset
        for view in view_names:
            v_plot = ds_plot / view
            v_plot.mkdir(parents=True, exist_ok=True)
            for metric, key, better in [("PSNR", "psnr_mean", "higher"), ("SSIM", "ssim_mean", "higher"), ("LPIPS", "lpips_mean", "lower")]:
                vals = [r["per_view"][ds]["views"].get(view, {}).get(key, float("nan")) for r in ds_recs]  # per-iter value for this metric/view
                _plot_metric_vs_iter(iters, vals, f"{ds} [{view}] — {metric} vs iter", f"{metric} ({better} = better)", v_plot / f"{metric.lower()}.png")
        logger.info(f"💾 [{ds}] plots under {ds_plot}")


if __name__ == "__main__":
    main()
