"""Compare runs on the episodes they'd ALL still be evaluating under a stricter chunk cap.

Longer episodes accumulate more autoregressive drift, so a run with a looser
`max_chunks` can look worse purely from evaluating longer rollouts, not from a
worse model. This recomputes aggregate metrics using only the intersection of
episodes whose actual `num_chunks` is below `--max-chunks` in EVERY run being
compared, so all runs are scored on the same (shorter) rollouts.

Usage:
    clap-eval-capped-chunk-metrics --run-dirs <iter_dir1> <iter_dir2> ... \\
        --output-dir eval_outputs/_capped/max20 --max-chunks 20
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_run(iter_dir: Path) -> Dict[str, dict]:
    """{episode_key: traj_dict} for every episode in `iter_dir/info/*.json`."""
    info_dir = iter_dir / "info"
    if not info_dir.is_dir():
        raise FileNotFoundError(f"No info/ subdir in {iter_dir}")
    trajs = {}
    for p in sorted(info_dir.glob("*.json")):  # one json per evaluated episode
        d = json.loads(p.read_text())
        m = d.get("metrics", {})
        trajs[p.stem] = {  # filename stem (dataset_episodeid) is identical across runs of the same test set
            "dataset_name": d.get("dataset_name"), "num_chunks": d.get("num_chunks"),
            "psnr_mean": m.get("psnr_mean"), "ssim_mean": m.get("ssim_mean"),
            "lpips_mean": m.get("lpips_mean"), "latent_mse": m.get("latent_mse"),
            "psnr_per_frame": m.get("psnr_per_frame", []), "ssim_per_frame": m.get("ssim_per_frame", []),
            "lpips_per_frame": m.get("lpips_per_frame", []),
        }
    return trajs


def aggregate(trajs: List[dict]) -> dict:
    if not trajs:
        return {}  # nothing to aggregate (e.g. empty shared-episode intersection)

    def nanmean(vals):
        v = [x for x in vals if x is not None]  # drop episodes missing this metric
        return float(np.mean(v)) if v else None

    def per_frame_mean(key):
        arrs = [t[key] for t in trajs if t[key]]  # episodes that have this per-frame metric at all
        if not arrs:
            return [], []
        max_len = max(len(a) for a in arrs)  # longest episode sets the frame axis length
        mat = np.full((len(arrs), max_len), np.nan)  # (n_episodes, max_len), NaN-padded for shorter episodes
        for i, a in enumerate(arrs):
            mat[i, : len(a)] = a
        return np.nanmean(mat, axis=0).tolist(), np.sum(~np.isnan(mat), axis=0).astype(int).tolist()  # per-frame mean, and how many episodes contributed to it

    psnr_pf, n_per_t = per_frame_mean("psnr_per_frame")
    ssim_pf, _ = per_frame_mean("ssim_per_frame")
    lpips_pf, _ = per_frame_mean("lpips_per_frame")

    return {
        "n_trajs": len(trajs),
        "psnr_mean": nanmean(t["psnr_mean"] for t in trajs),
        "ssim_mean": nanmean(t["ssim_mean"] for t in trajs),
        "lpips_mean": nanmean(t["lpips_mean"] for t in trajs),
        "latent_mse_mean": nanmean(t["latent_mse"] for t in trajs),
        "psnr_per_frame_mean": psnr_pf, "ssim_per_frame_mean": ssim_pf, "lpips_per_frame_mean": lpips_pf,
        "n_episodes_per_frame": n_per_t,
    }


def run_label(iter_dir: Path) -> str:
    """Last 3 path components, so sibling <experiment>/<split>/<iter> dirs get distinct labels."""
    return "__".join(iter_dir.resolve().parts[-3:])


def cli():
    p = argparse.ArgumentParser(description=__doc__)  # parses run dirs to compare and the shared chunk cap
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-chunks", type=int, default=20)
    return p.parse_args()


def main():
    args = cli()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_data = {run_label(Path(rd)): (Path(rd), load_run(Path(rd))) for rd in args.run_dirs}  # {label: (dir, {episode_key: traj})}
    kept_per_run = {
        lbl: {k for k, t in trajs.items() if t["num_chunks"] is not None and t["num_chunks"] < args.max_chunks}  # episodes under the chunk cap for this run
        for lbl, (_, trajs) in run_data.items()
    }
    shared_keys = set.intersection(*kept_per_run.values()) if kept_per_run else set()  # episodes under the cap in EVERY run

    logger.info(f"📊 max_chunks < {args.max_chunks}")
    logger.info(f"📊 {'Run':<60}  {'total':>6}  {'eligible':>8}  {'shared':>6}  {'PSNR':>8}  {'SSIM':>8}  {'LPIPS':>8}  {'lat_mse':>10}")
    logger.info("📊 " + "-" * 120)

    fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A  "
    summary = {}
    for lbl, (iter_dir, trajs) in run_data.items():
        shared_trajs = [trajs[k] for k in sorted(shared_keys)]  # this run's trajectories, restricted to the shared episode set
        agg = aggregate(shared_trajs)
        logger.info(f"📊 {lbl:<60}  {len(trajs):>6}  {len(kept_per_run[lbl]):>8}  {len(shared_keys):>6}  "
              f"{fmt(agg.get('psnr_mean')):>8}  {fmt(agg.get('ssim_mean')):>8}  {fmt(agg.get('lpips_mean')):>8}  {fmt(agg.get('latent_mse_mean')):>10}")

        result = {
            "run_dir": str(iter_dir), "max_chunks_filter": args.max_chunks,
            "n_total": len(trajs), "n_eligible": len(kept_per_run[lbl]), "n_shared": len(shared_keys),
            **{k: agg.get(k) for k in ("psnr_mean", "ssim_mean", "lpips_mean", "latent_mse_mean",
                                        "psnr_per_frame_mean", "ssim_per_frame_mean", "lpips_per_frame_mean", "n_episodes_per_frame")},
            "shared_episode_keys": sorted(shared_keys),
        }
        (out_dir / f"{lbl}.json").write_text(json.dumps(result, indent=2))  # per-run detail file
        summary[lbl] = {k: v for k, v in result.items() if k != "shared_episode_keys" and "per_frame" not in k and k != "n_episodes_per_frame"}  # compact scalars-only view

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))  # compact cross-run summary
    logger.info(f"\n💾 Results written to {out_dir}")


if __name__ == "__main__":
    main()
