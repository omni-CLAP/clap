"""High-level checkpoint evaluator: `clap-eval --config <path> --experiment <name> --ckpt <ckpt>`.

Wraps `clap.rollout.agent.CLAPRolloutAgent` in a per-dataset loop: replays
every selected episode, writes GT/prediction videos + per-episode metrics
(same layout `clap.rollout.replay` uses), then adds what a single replay run
doesn't compute — per-dataset FVD/FID and a cross-dataset aggregate — plus a
`--resume` mode that reloads already-written episodes instead of re-running
them, so a killed job can be cheaply completed.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import mediapy
import numpy as np

from clap.config import PathConfig, load_config
from clap.config.rollout import RolloutReplayConfig
from clap.eval.dataset_specs import get_spec as get_dataset_spec
from clap.eval.experiments import get_experiment
from clap.eval.metrics import compute_fid, compute_fvd
from clap.utils import setup_logging

# Deferred (not at module level): clap.rollout.replay imports clap.data.rollout_loaders,
# which imports clap.eval.episode_eligibility — importing clap.rollout here at module level
# would cycle back into this still-initializing clap.eval package whenever clap.rollout is
# the entry point (e.g. `import clap.rollout.deploy` before anything else has touched clap.eval).

setup_logging()
logger = logging.getLogger(__name__)

_TEST_SETS_CACHE = os.path.join(os.path.dirname(__file__), "test_sets_cache")


def _resolve_ckpt(ckpts_root: str, ckpt: str, paths: PathConfig) -> str:
    """"last" / a step number resolve against the experiment's ckpts_root; anything else
    (absolute, or relative to checkpoint_root) goes through `PathConfig.resolve_checkpoint`.
    """
    root = paths.resolve_checkpoint(ckpts_root)
    if ckpt == "last":
        return os.path.join(root, "last.pt")
    if ckpt.isdigit():
        return os.path.join(root, f"checkpoint-{ckpt}.pt")
    return paths.resolve_checkpoint(ckpt)


def _load_test_set(name: str, cache_dir: str = _TEST_SETS_CACHE) -> dict:
    """Return {"split": ..., "datasets": {ds: [ep_id, ...]}} from a cached test-set JSON.

    cache_dir defaults to the package's own clap/eval/test_sets_cache/ (where every local
    test set lives); pass clap-build-test-sets' own --out-dir here to read a test set built
    somewhere else instead (e.g. a scratch dir, so as not to overwrite a local one).
    """
    path = os.path.join(cache_dir, f"{name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Test-set cache not found: {path}\nRun: clap-build-test-sets --names {name}")
    with open(path) as f:
        return json.load(f)


def _load_cached_episode(save_dir: Path, ds_name: str, ep_id: str):
    """Reload a previously-written episode's info/video for --resume, or None if nothing is cached yet.

    Returns (info_dict, gt_u8, pred_u8).
    """
    info_path = save_dir / "info" / f"{ds_name}_{ep_id}.json"
    video_path = save_dir / "video" / f"{ds_name}_{ep_id}.mp4"
    if not (info_path.is_file() and video_path.is_file()):
        return None
    try:
        info = json.loads(info_path.read_text())
        video_cat = np.asarray(mediapy.read_video(str(video_path)))
    except Exception as e:
        logger.warning(f"⚠️ [{ds_name}] resume: cache for ep={ep_id} unreadable ({e}), re-running")
        return None
    half = video_cat.shape[1] // 2  # GT (top) / prediction (bottom) stacked along H at write time
    return info, video_cat[:, :half], video_cat[:, half:]


def _per_frame_agg(trajs: List[dict], key: str):
    """Mean (ignoring padding) of a per-frame metric across episodes of different lengths."""
    arrs = [t[key] for t in trajs]
    max_len = max(len(a) for a in arrs)  # longest episode sets the frame axis length
    stacked = np.full((len(arrs), max_len), np.nan)  # (n_episodes, max_len), NaN-padded for shorter episodes
    for i, a in enumerate(arrs):
        stacked[i, : len(a)] = a
    counts = np.sum(~np.isnan(stacked), axis=0).astype(int).tolist()  # how many episodes contributed to each frame index
    return np.nanmean(stacked, axis=0).tolist(), counts


def evaluate(
    experiment: str,
    config_path: str,
    ckpt: str = "last",
    datasets: Optional[List[str]] = None,
    episode_ids_by_dataset: Optional[Dict[str, List[str]]] = None,
    test_set: Optional[str] = None,
    test_sets_cache_dir: str = _TEST_SETS_CACHE,
    split: str = "val",
    save_dir: Optional[str] = None,
    skip_fvd: bool = False,
    skip_fid: bool = False,
    resume: bool = False,
    max_episodes_per_dataset: int = 0,
    paths: Optional[PathConfig] = None,
    **rollout_overrides,
) -> dict:
    """Run full evaluation for one checkpoint: rollout + PSNR/SSIM/LPIPS/FVD/FID.

    Args:
        experiment: Key in `clap.eval.experiments.EXPERIMENTS`.
        config_path: TrainingRunConfig-shaped YAML (model:/data: sections) the
            checkpoint was trained with.
        ckpt: "last", a step number, or a path — resolved against the
            experiment's `ckpts_root` (see `_resolve_ckpt`).
        datasets: Defaults to the experiment's `default_datasets`.
        test_set: Name of a cached test set (`<test_sets_cache_dir>/<name>.json`);
            overrides `datasets`/`episode_ids_by_dataset` when given.
        test_sets_cache_dir: Where `test_set` is read from. Defaults to the package's own
            `clap/eval/test_sets_cache/`; point at a scratch dir (matching `clap-build-test-sets
            --out-dir`) to read a test set built elsewhere instead, without touching the
            local repo ones.
        resume: Reload already-written episodes from `save_dir` instead of
            re-running them, to cheaply complete a job that died mid-run.
        rollout_overrides: Any `RolloutReplayConfig` field, or `trim_static_prefix`/
            `skip_first_n_frames` (otherwise taken per-dataset from `dataset_specs`).

    Returns:
        {"aggregate": ..., "per_view_summary": ..., "per_episode": ...}
    """
    from clap.rollout.agent import CLAPRolloutAgent  # deferred import, see module-level note above
    from clap.rollout.replay import build_loader, ep_id_str, select_indices

    paths = paths or PathConfig()  # default path resolution if caller didn't supply one
    exp = get_experiment(experiment)  # registry lookup: family/conditioning/checkpoint root for this experiment
    config = load_config(config_path)  # TrainingRunConfig the checkpoint was trained with
    config.model.conditioning = exp.conditioning  # override with the experiment's declared conditioning
    config.model.action_dim = exp.action_dim
    if exp.family == "lam" and exp.lam_subdir_override:
        config.data.oxe_lam_subdir = exp.lam_subdir_override  # point at a specific latent-action extractor run
    if exp.family == "language":
        config.data.action_caption_mode = exp.action_caption_mode

    ckpt_path = _resolve_ckpt(exp.ckpts_root, ckpt, paths)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # trim_static_prefix/skip_first_n_frames are per-dataset (clap.eval.dataset_specs), not
    # RolloutReplayConfig fields, so they're pulled out of rollout_overrides before the rest
    # goes into RolloutReplayConfig below.
    dataset_overrides = {
        k: rollout_overrides.pop(k) for k in ("trim_static_prefix", "skip_first_n_frames") if k in rollout_overrides
    }

    if test_set is not None:
        ts = _load_test_set(test_set, test_sets_cache_dir)  # cached {split, datasets: {ds: [ep_id, ...]}} snapshot
        split = ts.get("split", split)
        ts_datasets = list(ts["datasets"].keys())
        if datasets:  # --datasets filters a test set down to a subset (e.g. debugging one dataset)
            ts_datasets = [d for d in ts_datasets if d in datasets]
        datasets = ts_datasets
        episode_ids_by_dataset = {ds: eps for ds, eps in ts["datasets"].items() if eps}  # drop datasets with an empty episode list

    datasets = datasets or exp.default_datasets  # fall back to the experiment's default dataset list
    if not datasets:
        raise ValueError(f"No datasets given and experiment {experiment!r} has no default_datasets.")

    if save_dir is None:
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp so repeated runs don't collide
        save_dir = os.path.join("eval_outputs", experiment, ts_str)
    save_dir = Path(save_dir)
    (save_dir / "video").mkdir(parents=True, exist_ok=True)
    (save_dir / "info").mkdir(parents=True, exist_ok=True)
    logger.info(f"[bold cyan]💾 writing videos/info to {save_dir}[/bold cyan] (aggregate.json/per_view_summary.json land here once the run finishes)")

    replay_config = RolloutReplayConfig(family=exp.family, ckpt_path=ckpt_path, **rollout_overrides)  # replay knobs (inference steps, guidance, etc.)
    agent = CLAPRolloutAgent(config.model, ckpt_path, family=exp.family, action_caption_mode=config.data.action_caption_mode)  # loads the checkpoint once, reused across every dataset below

    all_traj_summary: List[dict] = []  # per-episode metric records across all datasets
    per_view_summary: Dict[str, dict] = {}  # per-dataset {n_episodes, stacking_mode, views, fvd, fid}

    for ds_name in datasets:
        spec = get_dataset_spec(ds_name)  # per-dataset defaults (trim/skip/max_chunks)
        trim_static_prefix = dataset_overrides.get("trim_static_prefix", spec.trim_static_prefix)  # CLI override wins over the dataset spec
        skip_first_n_frames = dataset_overrides.get("skip_first_n_frames", spec.skip_first_n_frames)
        max_chunks = replay_config.max_chunks if replay_config.max_chunks else spec.max_chunks  # 0/None in config means "use dataset default"

        loader = build_loader(exp.family, ds_name, config.data, config.model, split=split)  # this dataset's episode loader
        if len(loader) == 0:
            logger.warning(f"⚠️ [{ds_name}] no episodes — skipping")
            continue
        indices = select_indices(loader, episode_ids=(episode_ids_by_dataset or {}).get(ds_name), max_episodes=max_episodes_per_dataset)  # specific episode IDs, or capped-count selection
        if not indices:
            logger.warning(f"⚠️ [{ds_name}] no episodes selected — skipping")
            continue

        gt_videos_ds: List[np.ndarray] = []  # accumulated for this dataset's FVD/FID
        pred_videos_ds: List[np.ndarray] = []
        stacking_mode_ds = None

        for i in indices:
            ep_id_guess = ep_id_str(loader.episodes[i])  # ID without loading the full episode, for the resume cache lookup
            if resume:
                cached = _load_cached_episode(save_dir, ds_name, ep_id_guess)
                if cached is not None:
                    info, gt_u8, pred_u8 = cached
                    logger.info(f"♻️ [{ds_name}] resume: skip ep={ep_id_guess} (cached)")
                    all_traj_summary.append({"dataset_name": ds_name, "episode_id": info["episode_id"], **info["metrics"]})
                    gt_videos_ds.append(gt_u8)
                    pred_videos_ds.append(pred_u8)
                    stacking_mode_ds = stacking_mode_ds or info.get("stacking_mode_used")
                    continue  # cache hit: skip rollout entirely for this episode

            try:
                ep = loader.load(i)  # full episode (frames, actions, text, ...)
            except Exception as e:
                logger.warning(f"⚠️ [{ds_name}] load failed idx={i}: {e}")
                continue

            ep_id = ep_id_str(ep)
            text = ep.get("text") or ep.get("task", "")  # task description used as conditioning/logging
            logger.info(f"🎬 {ds_name} [{i}/{len(loader)}] ep={ep_id} text={str(text)[:60]!r}")

            try:
                rollout = agent.autoregressive_replay(  # roll the checkpoint forward chunk-by-chunk over the whole episode
                    ep, agent.model.config.num_history, agent.model.config.num_frames,
                    num_inference_steps=replay_config.num_inference_steps, guidance_scale=replay_config.guidance_scale,
                    decode_chunk_size=replay_config.decode_chunk_size, max_chunks=max_chunks or 0,
                    gt_cond=replay_config.gt_cond, history_idx=replay_config.history_idx,
                    trim_static_prefix=trim_static_prefix, skip_first_n_frames=skip_first_n_frames,
                )
            except Exception as e:
                logger.warning(f"⚠️ rollout failed: {e}")
                continue

            metrics, pred_u8, gt_u8 = agent.compute_metrics(  # PSNR/SSIM/LPIPS/latent-MSE (overall and per view), plus decoded uint8 videos
                rollout["pred_latents"], rollout["gt_aligned"],
                stacking_mode=ep.get("stacking_mode_used"), decode_chunk_size=replay_config.decode_chunk_size,
            )

            video_cat = np.concatenate([gt_u8, pred_u8], axis=1)  # GT on top, prediction below
            video_path = save_dir / "video" / f"{ds_name}_{ep_id}.mp4"
            mediapy.write_video(str(video_path), video_cat, fps=4)

            info = {
                "dataset_name": ds_name, "episode_id": ep_id, "num_chunks": rollout["num_chunks"],
                "text": text, "metrics": metrics, "stacking_mode_used": ep.get("stacking_mode_used"),
                "video_path": str(video_path.relative_to(save_dir)), "ckpt_path": ckpt_path, "family": exp.family,
            }
            (save_dir / "info" / f"{ds_name}_{ep_id}.json").write_text(json.dumps(info, indent=2))  # per-episode record, read back on --resume

            logger.info(f"📊 PSNR={metrics['psnr_mean']:.3f}  SSIM={metrics['ssim_mean']:.4f}  "
                        f"LPIPS={metrics['lpips_mean']:.4f}  latent_MSE={metrics['latent_mse']:.6f}")

            all_traj_summary.append({"dataset_name": ds_name, "episode_id": ep_id, **metrics})
            gt_videos_ds.append(gt_u8)
            pred_videos_ds.append(pred_u8)
            stacking_mode_ds = stacking_mode_ds or ep.get("stacking_mode_used")

        if not gt_videos_ds:
            continue  # nothing succeeded for this dataset — no FVD/FID/summary to compute

        ds_traj = [t for t in all_traj_summary if t["dataset_name"] == ds_name]  # this dataset's episode records
        view_names = sorted({v for t in ds_traj for v in t.get("psnr_mean_per_view", {})})  # every camera view seen for this dataset
        views = {
            v: {
                "psnr_mean": float(np.mean([t["psnr_mean_per_view"][v] for t in ds_traj if v in t.get("psnr_mean_per_view", {})])),  # mean over episodes that have this view
                "ssim_mean": float(np.mean([t["ssim_mean_per_view"][v] for t in ds_traj if v in t.get("ssim_mean_per_view", {})])),
                "lpips_mean": float(np.mean([t["lpips_mean_per_view"][v] for t in ds_traj if v in t.get("lpips_mean_per_view", {})])),
            }
            for v in view_names
        }
        ds_summary: dict = {"n_episodes": len(gt_videos_ds), "stacking_mode": stacking_mode_ds, "views": views}
        if skip_fvd:
            ds_summary["fvd"], ds_summary["fvd_n_pairs"] = float("nan"), 0  # skip the (expensive) FVD computation
        else:
            logger.info(f"🧮 [{ds_name}] computing FVD over {len(gt_videos_ds)} episodes ...")
            ds_summary["fvd"], ds_summary["fvd_n_pairs"] = compute_fvd(gt_videos_ds, pred_videos_ds)  # population-level metric over all episodes at once
            logger.info(f"📈 FVD={ds_summary['fvd']:.2f}  (n={ds_summary['fvd_n_pairs']})")
        if skip_fid:
            ds_summary["fid"], ds_summary["fid_n_frames"] = float("nan"), 0  # skip the (expensive) FID computation
        else:
            logger.info(f"🧮 [{ds_name}] computing FID over {len(gt_videos_ds)} episodes ...")
            ds_summary["fid"], ds_summary["fid_n_frames"] = compute_fid(gt_videos_ds, pred_videos_ds)
            logger.info(f"📈 FID={ds_summary['fid']:.2f}  (n_frames={ds_summary['fid_n_frames']})")
        per_view_summary[ds_name] = ds_summary

    aggregate: dict = {}
    if all_traj_summary:  # at least one episode succeeded somewhere
        psnr_pf, n_per_t = _per_frame_agg(all_traj_summary, "psnr_per_frame")  # per-frame mean across ALL episodes/datasets
        ssim_pf, _ = _per_frame_agg(all_traj_summary, "ssim_per_frame")
        lpips_pf, _ = _per_frame_agg(all_traj_summary, "lpips_per_frame")

        all_view_names = sorted({v for t in all_traj_summary for v in t.get("psnr_mean_per_view", {})})  # every camera view seen across all datasets
        mk_pv = lambda key: {
            v: float(np.mean([t[key][v] for t in all_traj_summary if v in t.get(key, {})]))
            for v in all_view_names
        }

        aggregate = {
            "n_trajs": len(all_traj_summary),
            "psnr_mean": float(np.mean([t["psnr_mean"] for t in all_traj_summary])),
            "ssim_mean": float(np.mean([t["ssim_mean"] for t in all_traj_summary])),
            "lpips_mean": float(np.mean([t["lpips_mean"] for t in all_traj_summary])),
            "latent_mse_mean": float(np.mean([t["latent_mse"] for t in all_traj_summary])),
            "psnr_mean_per_view": mk_pv("psnr_mean_per_view"),
            "ssim_mean_per_view": mk_pv("ssim_mean_per_view"),
            "lpips_mean_per_view": mk_pv("lpips_mean_per_view"),
            "psnr_per_frame_mean": psnr_pf, "ssim_per_frame_mean": ssim_pf, "lpips_per_frame_mean": lpips_pf,
            "n_episodes_per_frame": n_per_t, "per_traj": all_traj_summary,
            "ckpt_path": ckpt_path, "experiment": experiment, "family": exp.family,
        }
        if not skip_fvd:
            aggregate["fvd_by_dataset"] = {ds: s["fvd"] for ds, s in per_view_summary.items() if "fvd" in s}
        if not skip_fid:
            aggregate["fid_by_dataset"] = {ds: s["fid"] for ds, s in per_view_summary.items() if "fid" in s}

        (save_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2))  # cross-dataset summary, consumed by clap.eval.aggregate/compare
        (save_dir / "per_view_summary.json").write_text(json.dumps(per_view_summary, indent=2))  # per-dataset/per-view summary

        logger.info(f"📊 aggregate over {aggregate['n_trajs']} episodes: "
                    f"PSNR={aggregate['psnr_mean']:.3f}  SSIM={aggregate['ssim_mean']:.4f}  "
                    f"LPIPS={aggregate['lpips_mean']:.4f}  latent_MSE={aggregate['latent_mse_mean']:.6f}")
        for ds, s in per_view_summary.items():
            logger.info(f"📈 FVD[{ds}]={s.get('fvd', float('nan')):.2f}  FID[{ds}]={s.get('fid', float('nan')):.2f}")
        logger.info(f"✅ wrote results to {save_dir}")
    else:
        logger.warning("⚠️ no episodes were rolled out successfully")

    return {"aggregate": aggregate, "per_view_summary": per_view_summary, "per_episode": all_traj_summary}


def cli():
    p = argparse.ArgumentParser(description=__doc__)  # parses all evaluate() knobs for the command-line entry point
    p.add_argument("--config", required=True, help="TrainingRunConfig YAML the checkpoint was trained with.")
    p.add_argument("--experiment", required=True, help="Key in clap.eval.experiments.EXPERIMENTS.")
    p.add_argument("--ckpt", default="last", help="'last', a step number, or a path.")
    p.add_argument("--datasets", nargs="+", default=None, help="Defaults to the experiment's default_datasets.")
    p.add_argument("--test-set", default=None, help="Name of a cached test set (<test-sets-cache-dir>/<name>.json).")
    p.add_argument("--test-sets-cache-dir", default=_TEST_SETS_CACHE,
                   help="Where --test-set is read from (default: the package's own clap/eval/test_sets_cache/). "
                        "Point at a scratch dir matching clap-build-test-sets --out-dir to read a test set "
                        "built elsewhere, without touching the committed ones.")
    p.add_argument("--split", default="val", choices=["val", "train"])
    p.add_argument("--save-dir", default=None)
    p.add_argument("--max-episodes-per-dataset", type=int, default=0)
    p.add_argument("--no-fvd", action="store_true")
    p.add_argument("--no-fid", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip rollout for episodes already written to --save-dir, reloading their "
                        "cached metrics instead — to cheaply complete a job that died mid-run.")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--max-chunks", type=int, default=0)
    p.add_argument("--decode-chunk-size", type=int, default=7)
    p.add_argument("--gt-cond", action="store_true")
    p.add_argument("--history-idx", type=int, nargs="+", default=None)
    p.add_argument("--trim-static-prefix", action="store_true", default=False,
                   help="Force trim_static_prefix=True for every dataset (default: per-dataset spec).")
    p.add_argument("--skip-first-n-frames", type=int, default=None,
                   help="Force skip_first_n_frames for every dataset (default: per-dataset spec).")
    return p.parse_args()


def main():
    args = cli()

    overrides = {}  # only pass along flags the user actually set, so evaluate()'s own defaults apply otherwise
    if args.gt_cond:
        overrides["gt_cond"] = True
    if args.history_idx:
        overrides["history_idx"] = args.history_idx
    if args.trim_static_prefix:
        overrides["trim_static_prefix"] = True
    if args.skip_first_n_frames is not None:
        overrides["skip_first_n_frames"] = args.skip_first_n_frames

    evaluate(
        experiment=args.experiment, config_path=args.config, ckpt=args.ckpt,
        datasets=args.datasets, test_set=args.test_set, test_sets_cache_dir=args.test_sets_cache_dir,
        split=args.split, save_dir=args.save_dir,
        skip_fvd=args.no_fvd, skip_fid=args.no_fid, resume=args.resume,
        max_episodes_per_dataset=args.max_episodes_per_dataset,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        max_chunks=args.max_chunks, decode_chunk_size=args.decode_chunk_size, **overrides,
    )


if __name__ == "__main__":
    main()
