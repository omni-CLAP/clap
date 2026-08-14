"""Offline autoregressive replay: `clap-rollout-replay --config <path>`.

Rolls a checkpoint out against recorded episodes and writes side-by-side
GT/prediction videos + per-episode/aggregate metrics — the eval-time
counterpart to `clap.training.train`.
"""

import argparse
import json
import logging
import os
from datetime import datetime

import mediapy
import numpy as np

from clap.data.rollout_loaders import ROLLOUT_LOADERS
from clap.eval.dataset_specs import get_spec
from clap.rollout.agent import CLAPRolloutAgent
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def ep_id_str(ep):
    """Canonical per-episode key: LAM episodes use 'rel' (no episode_id), everything else uses episode_id."""
    return str(ep.get("episode_id") or ep.get("rel", "unknown")).replace("/", "_")


def build_loader(family, dataset_name, data_config, model_config, split="val", strict_eligibility=False):
    """Instantiate the right episode loader for `family`, from shared data/model config."""
    loader_cls = ROLLOUT_LOADERS[family]  # loader class registered for this conditioning family
    common = dict(
        dataset_name=dataset_name, oxe_base_path=data_config.oxe_base_path,
        video_size=data_config.video_size, split=split,
        num_history=model_config.num_history, num_frames=model_config.num_frames,
    )
    if family == "lam":
        return loader_cls(
            oxe_lam_root=data_config.oxe_lam_root or data_config.oxe_base_path,  # LAM-specific dataset root
            oxe_lam_subdir=data_config.oxe_lam_subdir,
            strict_eligibility=strict_eligibility, **common,
        )
    return loader_cls(
        oxe_lam_root=data_config.oxe_lam_root, oxe_lam_subdir=data_config.oxe_lam_subdir,  # unused by non-LAM loaders but accepted
        strict_eligibility=strict_eligibility, dataset_meta_info_path=data_config.dataset_meta_info_path, **common,
    )


def select_indices(loader, episode_ids=None, episode_indices=None, max_episodes=0):
    """Pick which episodes to replay: explicit ids > explicit indices > everything, then
    max_episodes (if > 0) additionally truncates whichever of those was selected -- so
    --test-set + --max-episodes-per-dataset together still caps the pinned episode list.
    """
    if episode_ids:
        from clap.data.lam import _normalize_ep_key
        wanted = {_normalize_ep_key(e) for e in episode_ids}  # normalized set of requested ids
        indices = [
            i for i, ep in enumerate(loader.episodes)
            if _normalize_ep_key(str(ep.get("episode_id") or ep.get("rel", ""))) in wanted
        ]
        found = {_normalize_ep_key(str(loader.episodes[i].get("episode_id") or loader.episodes[i].get("rel", ""))) for i in indices}
        missing = wanted - found
        if missing:
            logger.warning(f"⚠️ {len(missing)} requested episode id(s) not found: {sorted(missing)[:5]}...")
    elif episode_indices:
        indices = [i for i in episode_indices if 0 <= i < len(loader)]  # keep only in-range indices
    else:
        indices = list(range(len(loader)))  # every episode
    return indices[:max_episodes] if max_episodes > 0 else indices


def replay_dataset(agent, loader, indices, replay_config, save_dir, trim_static_prefix=None, skip_first_n_frames=None):
    """Replay every selected episode of one dataset; write video+info, return per-episode summaries.

    trim_static_prefix/skip_first_n_frames: None means "use this dataset's
    clap.eval.dataset_specs default"; an explicit True/int forces it for
    every dataset (same override convention as clap-eval).
    """
    spec = get_spec(loader.dataset_name)  # per-dataset defaults (trim/skip), same registry clap-eval uses
    trim_static_prefix = spec.trim_static_prefix if trim_static_prefix is None else trim_static_prefix
    skip_first_n_frames = spec.skip_first_n_frames if skip_first_n_frames is None else skip_first_n_frames

    summaries = []
    for i in indices:
        try:
            ep = loader.load(i)  # load one episode's video/states/actions
        except Exception as e:
            logger.warning(f"⚠️ failed to load episode index {i}: {e}")
            continue

        try:
            rollout = agent.autoregressive_replay(
                ep, agent.model.config.num_history, agent.model.config.num_frames,
                num_inference_steps=replay_config.num_inference_steps, guidance_scale=replay_config.guidance_scale,
                decode_chunk_size=replay_config.decode_chunk_size, max_chunks=replay_config.max_chunks or 0,
                gt_cond=replay_config.gt_cond, history_idx=replay_config.history_idx,
                trim_static_prefix=trim_static_prefix, skip_first_n_frames=skip_first_n_frames,
            )
        except Exception as e:
            logger.warning(f"⚠️ rollout failed for episode {ep_id_str(ep)}: {e}")
            continue

        metrics, pred_u8, gt_u8 = agent.compute_metrics(
            rollout["pred_latents"], rollout["gt_aligned"],
            stacking_mode=ep.get("stacking_mode_used"), decode_chunk_size=replay_config.decode_chunk_size,
        )

        eid = ep_id_str(ep)
        video_cat = np.concatenate([gt_u8, pred_u8], axis=1)  # GT on top, prediction below
        mediapy.write_video(os.path.join(save_dir, "video", f"{ep['dataset_name']}_{eid}.mp4"), video_cat, fps=4)
        with open(os.path.join(save_dir, "info", f"{ep['dataset_name']}_{eid}.json"), "w") as f:
            json.dump({
                "dataset_name": ep["dataset_name"], "episode_id": eid, "num_chunks": rollout["num_chunks"],
                "text": ep.get("text", ""), "metrics": metrics, "stacking_mode_used": ep.get("stacking_mode_used"),
            }, f, indent=2)  # per-episode metrics + metadata

        summaries.append({"dataset_name": ep["dataset_name"], "episode_id": eid, **metrics})
    return summaries  # one dict per successfully replayed episode


def cli():
    """Parse command-line arguments for the replay CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a TrainingRunConfig-shaped YAML (model:/data: sections only).")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--family", required=True, choices=list(ROLLOUT_LOADERS))
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--max-episodes-per-dataset", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--gt-cond", action="store_true")
    parser.add_argument("--trim-static-prefix", action="store_true", default=False,
                         help="Force trim_static_prefix=True for every dataset (default: per-dataset clap.eval.dataset_specs).")
    parser.add_argument("--skip-first-n-frames", type=int, default=None,
                         help="Force skip_first_n_frames for every dataset (default: per-dataset clap.eval.dataset_specs).")
    return parser.parse_args()


def main():
    from clap.config import load_config
    from clap.config.rollout import RolloutReplayConfig

    args = cli()  # parse CLI args
    config = load_config(args.config)  # load model/data config from YAML
    replay_config = RolloutReplayConfig(
        family=args.family, ckpt_path=args.ckpt, num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, max_chunks=args.max_chunks, gt_cond=args.gt_cond,
    )

    save_dir = args.save_dir or os.path.join("eval_outputs", "replay", datetime.now().strftime("%Y%m%d_%H%M%S"))  # timestamped default output dir
    os.makedirs(os.path.join(save_dir, "video"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "info"), exist_ok=True)

    agent = CLAPRolloutAgent(config.model, args.ckpt, family=args.family, action_caption_mode=config.data.action_caption_mode)  # load the checkpoint once

    all_summaries = []
    for dataset_name in args.datasets:
        loader = build_loader(args.family, dataset_name, config.data, config.model, split=args.split)  # per-dataset episode loader
        if len(loader) == 0:
            logger.warning(f"⚠️ [{dataset_name}] no episodes found, skipping")
            continue
        indices = select_indices(loader, max_episodes=args.max_episodes_per_dataset)
        logger.info(f"🎬 [{dataset_name}] replaying {len(indices)} episodes")
        all_summaries.extend(replay_dataset(
            agent, loader, indices, replay_config, save_dir,
            trim_static_prefix=True if args.trim_static_prefix else None,  # only override when explicitly forced; None keeps the per-dataset spec default
            skip_first_n_frames=args.skip_first_n_frames,
        ))

    if all_summaries:
        aggregate = {
            "n_episodes": len(all_summaries),
            "psnr_mean": float(np.mean([s["psnr_mean"] for s in all_summaries])),
            "ssim_mean": float(np.mean([s["ssim_mean"] for s in all_summaries])),
            "lpips_mean": float(np.mean([s["lpips_mean"] for s in all_summaries])),
            "latent_mse_mean": float(np.mean([s["latent_mse"] for s in all_summaries])),
            "per_episode": all_summaries,
        }
        with open(os.path.join(save_dir, "aggregate.json"), "w") as f:
            json.dump(aggregate, f, indent=2)  # dataset-level aggregate metrics
        logger.info(f"✅ wrote {len(all_summaries)} episode results to {save_dir}")


if __name__ == "__main__":
    main()
