from typing import Callable, Dict, List, Optional, Union

import einops
import torch
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from clap.models.pipeline_svd import StableVideoDiffusionPipeline, _append_dims


def _tensor_to_video(video: torch.Tensor, processor, output_type="np"):
    """(B, C, T, H, W) -> per-sample post-processed frames, via the pipeline's video processor."""
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)  # (T, C, H, W)
        outputs.append(processor.postprocess(batch_vid, output_type))
    return outputs


class CLAPDiffusionPipeline(StableVideoDiffusionPipeline):
    """Inference-time autoregressive rollout pipeline for `CLAPModel`.

    Unlike the base SVD pipeline, conditioning is a pre-computed action/language
    embedding (`action_hidden`, produced by `CLAPModel.action_encoder` or
    `_forward_language`) rather than a CLIP image embedding, and the image
    condition may already be a VAE latent (skip re-encoding) or a history of
    past latent frames to prepend.
    """

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        action_hidden: torch.Tensor,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        history: Optional[torch.Tensor] = None,
        frame_level_cond: bool = False,
        his_cond_zero: bool = False,
    ):
        """Denoise a video clip conditioned on `image` and `action_hidden`.

        Args:
            image: Conditioning frame. Either raw pixels `(B, 3, H, W)` (VAE-encoded
                here) or an already-encoded VAE latent `(B, 4, h, w)` (used as-is).
            action_hidden: Pre-computed conditioning embedding, `(B, 1 or T, dim)` —
                see `CLAPModel.action_encoder` / `_forward_language`.
            history: Optional `(B, num_history, 4, h, w)` past latent frames,
                prepended to the denoised clip and stripped from the output
                (the model conditions on them but they aren't part of the result).
            his_cond_zero: If True, zero out the image condition for history frames
                (mirrors `CLAPModel.forward`'s `his_cond_zero` training option).
            frame_level_cond: Forwarded to the UNet — True means `action_hidden`
                carries one token per frame instead of one token for the whole clip.

        Returns:
            `StableVideoDiffusionPipelineOutput` if `return_dict=True`, else a
            `(frames, latents)` tuple.
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        device = self.unet.device
        do_classifier_free_guidance = max_guidance_scale > 1.0

        image_embeddings = action_hidden
        batch_size = image_embeddings.shape[0]
        if do_classifier_free_guidance:
            # Batch the unconditional (zero) and conditional embeddings so CFG
            # only needs one UNet forward pass per step instead of two.
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if image.shape[-3] == 3:
            # Raw pixels: preprocess + VAE-encode.
            image = self.video_processor.preprocess(image, height=height, width=width)
            if needs_upcasting:
                self.vae.to(dtype=torch.float32)
            image_latents = self._encode_vae_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
            image_latents = image_latents.to(image_embeddings.dtype)
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            # Already a VAE latent (e.g. reusing an encoded history frame) — use as-is.
            image_latents = image / self.vae.config.scaling_factor
            if do_classifier_free_guidance:
                image_latents = torch.cat([image_latents] * 2)
            image_latents = image_latents.to(image_embeddings.dtype)

        # Broadcast the single conditioning-image latent across all frames (history + future).
        if history is not None:
            B, num_his, C, H, W = history.shape
            num_frames_all = num_frames + num_his
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames_all, 1, 1, 1)  # (B, T_all, 4, h, w)
            if his_cond_zero:
                image_latents[:, :num_his] = 0.0
        else:
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # (B, T, 4, h, w)

        added_time_ids = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype,
            batch_size, num_videos_per_prompt, do_classifier_free_guidance,
        ).to(device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # num_channels_latents=8 (4 noise + 4 image-cond); prepare_latents only samples the first half.
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_frames, num_channels_latents,
            height, width, image_embeddings.dtype, device, generator, latents,
        )  # (B, T, 4, h, w)

        # Per-frame CFG scale, ramped linearly across the clip.
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order  # accounts for scheduler.order > 1
        self._num_timesteps = len(timesteps)
        if history is not None:
            history = torch.cat([history] * 2) if do_classifier_free_guidance else history  # match CFG-doubled batch

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # latents: (B, T, 4, h, w) — only the future frames being denoised.
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # (B or 2B, T, 4, h, w)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if history is not None:
                    # Prepend history so the UNet sees (history + being-denoised future) as one clip.
                    latent_model_input = torch.cat([history, latent_model_input], dim=1)  # (.., num_his+T, 4, h, w)
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)  # (.., 8, h, w): 4+4 channels

                latent_model_input = latent_model_input.to(self.unet.dtype)
                image_embeddings = image_embeddings.to(self.unet.dtype)
                # Predict the noise residual for every frame (history included, if present).
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                    frame_level_cond=frame_level_cond,
                )[0]  # (.., num_his+T or T, 4, h, w)

                if history is not None:
                    noise_pred = noise_pred[:, num_his:]  # drop history frames, keep only the predicted future

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)  # each (B, T, 4, h, w)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # Step the scheduler: noise_pred + current latents -> less-noisy latents for t-1.
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample  # (B, T, 4, h, w)

                if callback_on_step_end is not None:
                    # Let the caller inspect/modify latents (or other tracked tensors) between steps.
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                # Advance the progress bar once per "real" step, not once per scheduler substep.
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type != "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            latents = latents.to(self.vae.dtype)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = _tensor_to_video(frames, self.video_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return frames, latents
        return StableVideoDiffusionPipelineOutput(frames=frames)
