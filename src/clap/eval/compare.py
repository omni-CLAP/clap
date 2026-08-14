"""Cross-experiment comparison from one or more aggregate.json files: `clap-eval-compare --config <path>`.

Config format (JSON):
    {
        "experiments": {
            "oxe_ee @58k":  "eval_outputs/cross_embodiment_oxe_ee/val/iter_000058000/aggregate.json",
            "oxe_lam @60k": "eval_outputs/cross_embodiment_oxe_lam_clap/val/iter_000060000/aggregate.json"
        },
        "exclude_datasets": ["egodex"],
        "title": "Best-checkpoint comparison",
        "out_path": "eval_outputs/_compare/best_per_experiment.png"
    }
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

METRICS = [("psnr_mean", "PSNR (higher = better)"), ("ssim_mean", "SSIM (higher = better)"), ("lpips_mean", "LPIPS (lower = better)")]


def _per_dataset_means(agg_path: Path, exclude: set) -> Dict[str, Dict]:
    agg = json.loads(Path(agg_path).read_text())
    buckets: Dict[str, list] = defaultdict(list)
    for t in agg.get("per_traj", []):  # per-episode records from the aggregate.json
        ds = t.get("dataset_name")
        if ds and ds not in exclude:
            buckets[ds].append(t)
    return {
        ds: {
            "psnr_mean": float(np.mean([x["psnr_mean"] for x in items])),  # mean over episodes for this dataset
            "ssim_mean": float(np.mean([x["ssim_mean"] for x in items])),
            "lpips_mean": float(np.mean([x["lpips_mean"] for x in items])),
            "n": len(items),
        }
        for ds, items in buckets.items()
    }


def _grouped_bar(ax, table: Dict[str, Dict], ds_order: List[str], metric: str, title: str):
    labels = list(table.keys())  # one bar-group series per experiment
    width = 0.8 / max(1, len(labels))  # bar width so all series fit within one dataset's x-slot
    x = np.arange(len(ds_order))
    colors = plt.get_cmap("tab10").colors

    for i, lab in enumerate(labels):
        vals = [(table[lab].get(ds) or {}).get(metric, np.nan) for ds in ds_order]  # this experiment's value per dataset, NaN if missing
        offset = (i - (len(labels) - 1) / 2) * width  # center this series' bars within the group
        bars = ax.bar(x + offset, vals, width, label=lab, color=colors[i % 10])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if metric != "psnr_mean" else f"{v:.2f}", ha="center", va="bottom", fontsize=7)  # value label above each bar
    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, rotation=20)
    ax.set_title(title)
    ax.set_ylabel(title.split(" ")[0])
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def _build_table(experiments: Dict[str, Path], exclude: set):
    raw = {label: _per_dataset_means(p, exclude) for label, p in experiments.items()}  # per-experiment, per-dataset means
    all_ds = set().union(*[set(v.keys()) for v in raw.values()]) if raw else set()  # every dataset seen in any experiment
    common_ds = set.intersection(*[set(v.keys()) for v in raw.values()]) if raw else set()  # datasets present in every experiment

    avg_label = "Average*"
    table = {}
    for label, per_ds in raw.items():
        per_ds = dict(per_ds)
        if common_ds:
            per_ds[avg_label] = {m: float(np.mean([per_ds[d][m] for d in common_ds])) for m, _ in METRICS}  # average only over the fairly-comparable common datasets
        table[label] = per_ds

    priority = ["bridge", "droid"]
    ds_order = [d for d in priority if d in all_ds] + sorted(d for d in all_ds if d not in priority)  # bridge/droid first, rest alphabetical
    if common_ds:
        ds_order = ds_order + [avg_label]  # average column goes last
    return table, ds_order, sorted(common_ds)


def _print_table(name: str, table: Dict, ds_order: List[str]):
    logger.info(f"\n📊 === {name} ===")
    logger.info("📊 " + " | ".join(f"{h:>22}" for h in ["Experiment"] + ds_order))  # header row
    for lab, rows in table.items():
        line = [f"{lab:>22}"]
        for ds in ds_order:
            v = rows.get(ds)
            line.append(f"{'-':>22}" if v is None else f"P{v['psnr_mean']:.2f}/S{v['ssim_mean']:.3f}/L{v['lpips_mean']:.3f}".rjust(22))  # dash if this experiment lacks the dataset
        logger.info("📊 " + " | ".join(line))


def main():
    p = argparse.ArgumentParser(description=__doc__)  # parses --config, the JSON comparison spec
    p.add_argument("--config", type=Path, required=True, help="JSON config; see module docstring for format.")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    experiments = {k: Path(v) for k, v in cfg["experiments"].items()}
    exclude = set(cfg.get("exclude_datasets", []))
    title = cfg.get("title", "Cross-experiment comparison")
    out_path = Path(cfg.get("out_path", "eval_outputs/_compare/comparison.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table, ds_order, common_ds = _build_table(experiments, exclude)
    _print_table(title, table, ds_order)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))  # one subplot per metric (PSNR/SSIM/LPIPS)
    for ax, (m, label) in zip(axes, METRICS):
        _grouped_bar(ax, table, ds_order, m, label)
    handles, labs = axes[0].get_legend_handles_labels()  # legend is shared across all 3 subplots
    fig.legend(handles, labs, loc="upper center", ncol=len(labs), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(title, y=1.06, fontsize=14)
    if common_ds:
        fig.text(0.5, -0.02, "Average* = mean over datasets present in every experiment: " + ", ".join(common_ds), ha="center", va="top", fontsize=9, style="italic")  # footnote explaining the Average column
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    logger.info(f"\n💾 Wrote {out_path}")


if __name__ == "__main__":
    main()