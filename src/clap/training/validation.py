"""Periodic video-generation preview, logged during training (not the training loss itself)."""

import logging
import math
import os

import mediapy
import numpy as np
import torch

from clap.models.pipeline import CLAPDiffusionPipeline
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _get_latents_for_val(model, sample, device):
    """Return a (T, 4, h, w) latent tensor for one validation sample.

    Uses the sample's pre-encoded latent if present; otherwise VAE-encodes
    the raw video frames on the fly.
    """
    if "latent" in sample:
        return sample["latent"].to(device)
    video = sample["video"]
    if video.dim() == 4:
        video = video.unsqueeze(0)  # add a batch dim; encode_video_to_latent expects (B, C, T, H, W)
    return model.encode_video_to_latent(video.to(device))[0]


def _resolve_val_ids(val_dataset, ids_override):
    """Turn `ids_override` into flat indices into `val_dataset` (a ConcatDataset).

    Accepts either a flat list of indices, or a `{sub_dataset_name: [local_ids]}`
    dict — the latter is what `DataConfig.val_sample_ids` uses, since a fixed
    global index would silently point at the wrong sample if the mix of
    sub-datasets ever changes between runs.
    """
    if not ids_override:  # None, or DataConfig.val_sample_ids' default {} -- either means "no override"
        return None
    if isinstance(ids_override, list):
        return [i % len(val_dataset) for i in ids_override]

    name_to_offset, offset = {}, 0
    for sub in val_dataset.datasets:
        name = getattr(sub, "dataset_name", None)
        if name is None:
            raise ValueError("val sub-dataset is missing a 'dataset_name' attribute")
        name_to_offset[name] = (offset, len(sub))
        offset += len(sub)

    flat = []
    for ds_name, local_ids in ids_override.items():
        if ds_name not in name_to_offset:
            continue  # this run's mix doesn't include ds_name; skip rather than error
        base, sz = name_to_offset[ds_name]
        if sz == 0:
            logger.warning(f"⚠️ [val_sample_ids] {ds_name} has 0 val samples in this run; skipping")
            continue
        for li in local_ids:
            if not (0 <= li < sz):
                raise ValueError(f"local id {li} out of range for {ds_name} (val size {sz})")
            flat.append(base + li)
    return flat


def validate_video_generation(model, val_dataset, config, train_steps, videos_dir, preview_id, accelerator):
    """Roll out `config.training.video_num` validation clips and write a side-by-side GT/prediction video.

    Args:
        preview_id: Selects which pair of validation samples this call renders
            (the caller loops preview_id over 0..num_previews to cover video_num clips).
    """
    device = accelerator.device
    underlying = model.module if accelerator.num_processes > 1 else model
    pipeline: CLAPDiffusionPipeline = underlying.pipeline
    videos_col = 2  # clips rendered per preview_id

    ids_override = _resolve_val_ids(val_dataset, config.data.val_sample_ids)
    if ids_override:
        batch_id = ids_override[preview_id * videos_col:(preview_id + 1) * videos_col]
    else:
        # No explicit ids: stride uniformly across the val set.
        stride = max(1, int(len(val_dataset) / config.training.video_num / videos_col))
        batch_id = list(range(0, len(val_dataset), stride))[preview_id * videos_col:(preview_id + 1) * videos_col]
    batch_list = [val_dataset[idx] for idx in batch_id]

    latent_list = [_get_latents_for_val(underlying, t, device) for t in batch_list]
    video_gt = torch.stack(latent_list, dim=0)  # (B, T, 4, h, w)

    text = [t.get("text", "") for t in batch_list]
    if underlying.conditioning == "language":
        steps = [t["action_caption_steps"] for t in batch_list]
        actions = None
    else:
        actions = torch.stack([t["action"] for t in batch_list], dim=0).to(device)

    num_history = config.model.num_history
    his_latent_gt, future_latent_gt = video_gt[:, :num_history], video_gt[:, num_history:]
    current_latent = future_latent_gt[:, 0]  # conditioning frame for the rollout
    _, lat_h, lat_w = current_latent.shape[1:]
    pix_h, pix_w = lat_h * 8, lat_w * 8  # VAE downsamples 8x; recover the pixel-space size the pipeline expects

    with torch.no_grad():
        if underlying.conditioning == "language":
            num_total = num_history + config.model.num_frames
            action_hidden = underlying._forward_language({"action_caption_steps": steps, "text": text}, device, underlying.unet.dtype, num_total)
        else:
            adapter = getattr(underlying, "action_adapter", None)
            if adapter is not None:
                adapter_dtype = next(adapter.parameters()).dtype
                actions = adapter(actions.to(adapter_dtype)).to(underlying.unet.dtype)
            action_hidden = underlying.action_encoder(actions, text, underlying.tokenizer, underlying.text_encoder, config.model.frame_level_cond)

        _, pred_latents = CLAPDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            action_hidden=action_hidden,
            width=pix_w, height=pix_h,
            num_frames=config.model.num_frames,
            history=his_latent_gt,
            num_inference_steps=config.training.num_inference_steps,
            decode_chunk_size=config.training.decode_chunk_size,
            max_guidance_scale=config.training.guidance_scale,
            fps=config.model.fps,
            motion_bucket_id=config.model.motion_bucket_id,
            output_type="latent",
            return_dict=False,
            frame_level_cond=config.model.frame_level_cond,
            his_cond_zero=config.model.his_cond_zero,
        )

    # Single-camera latents decode directly; multi-camera stacks are already
    # 3+ channels tall in latent space and decode the same way per-frame.
    decode_chunk_size = config.training.decode_chunk_size
    video_gt = _decode_latents(pipeline, video_gt, decode_chunk_size)
    videos = _decode_latents(pipeline, pred_latents, decode_chunk_size)

    # [-1, 1] -> uint8 pixels, then lay history+GT above history+prediction for a side-by-side comparison.
    video_gt = ((video_gt / 2.0 + 0.5).clamp(0, 1) * 255).to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(pipeline.unet.dtype).detach().cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)
    videos = np.concatenate([video_gt[:, :num_history], videos], axis=1)  # prepend history to the prediction too
    videos = np.concatenate([video_gt, videos], axis=-3)  # GT stacked above prediction
    videos = np.concatenate([video for video in videos], axis=-2)  # lay the batch out side by side

    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    ids_tag = "_".join(str(i) for i in batch_id)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{preview_id}_ids_{ids_tag}.mp4"
    mediapy.write_video(filename, videos, fps=2)


def _decode_latents(pipeline, latents, decode_chunk_size):
    """VAE-decode a (B, T, C, h, w) latent stack in chunks, preserving batch/time layout."""
    b, t = latents.shape[:2]
    flat = latents.flatten(0, 1)  # (B*T, C, h, w): decode ignores the batch/time split
    decoded = []
    for i in range(0, flat.shape[0], decode_chunk_size):  # decode_chunk_size frames at a time to bound memory
        chunk = flat[i:i + decode_chunk_size] / pipeline.vae.config.scaling_factor  # undo the VAE's latent scaling
        decoded.append(pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    decoded = torch.cat(decoded, dim=0)  # (B*T, C_pix, H, W)
    return decoded.reshape(b, t, *decoded.shape[1:])  # restore (B, T, C_pix, H, W)
