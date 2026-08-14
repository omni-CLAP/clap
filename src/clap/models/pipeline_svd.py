# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from diffusers' StableVideoDiffusionPipeline to use CLAP's action-
# conditioned UNet in place of diffusers' stock UNetSpatioTemporalConditionModel.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from clap.models.unet import UNetSpatioTemporalConditionModel

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from clap.models.pipeline_svd import StableVideoDiffusionPipeline
        >>> from diffusers.utils import load_image, export_to_video

        >>> pipe = StableVideoDiffusionPipeline.from_pretrained(
        ...     "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        ... )
        >>> pipe.to("cuda")

        >>> image = load_image("example.jpeg").resize((1024, 576))
        >>> frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        >>> export_to_video(frames, "generated.mp4", fps=7)
        ```
"""


def _append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]  # index with trailing None's to add size-1 dims for broadcasting


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """Call `scheduler.set_timesteps` and return the resulting timestep schedule.

    Exactly one of `timesteps`, `sigmas`, or `num_inference_steps` should be given
    to select custom vs. default timestep spacing.

    Returns:
        `(timesteps, num_inference_steps)` — the scheduler's timestep tensor and
        the (possibly recomputed) number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        # Custom timestep schedule: only supported if this scheduler's set_timesteps accepts it.
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)  # recompute since a custom schedule may not match the requested count
    elif sigmas is not None:
        # Custom sigma schedule: same idea, but noise-level based instead of timestep based.
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        # Default path: scheduler picks an evenly-spaced schedule of num_inference_steps.
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class StableVideoDiffusionPipelineOutput(BaseOutput):
    """Output of `StableVideoDiffusionPipeline`.

    Args:
        frames: `List[List[PIL.Image.Image]]`, `np.ndarray`, or `torch.Tensor` of
            shape `(batch_size, num_frames, height, width, num_channels)`.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class StableVideoDiffusionPipeline(DiffusionPipeline):
    """Image-to-video pipeline: encodes a conditioning image, then denoises a
    latent video clip conditioned on it (plus CLAP's action/text conditioning
    when used as `CLAPModel.pipeline`).

    Args:
        unet: `UNetSpatioTemporalConditionModel` — CLAP's action-conditioned
            variant, not diffusers' stock UNet.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        # register_modules makes these submodules visible to diffusers' pipeline
        # machinery (save_pretrained/from_pretrained, .to(device), etc.) as a group.
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        # Each VAE downsampling block halves spatial resolution once; total
        # downsampling factor = 2^(num_blocks - 1) between pixels and latents.
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Pixel<->latent resize/normalize helper, shared by every encode/decode call below.
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        """CLIP-encode the conditioning image -> (B, 1, embed_dim) cross-attention conditioning."""
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)
            # Normalize to [-1, 1] before resizing (matches the original SVD implementation), then back to [0, 1].
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # CLIP's own preprocessing (already resized above, so skip resize/crop here).
        image = self.feature_extractor(
            images=image, do_normalize=True, do_center_crop=False, do_resize=False, do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)  # (B, 1, embed_dim)

        # Expand for multiple samples per conditioning image.
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # Batch the unconditional (zero) and conditional embeddings together
            # so CFG only needs one UNet forward pass instead of two.
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        """VAE-encode the conditioning image -> latent (mode, not a sampled draw — deterministic)."""
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()  # deterministic (no sampling noise)
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)  # (B*num_videos_per_prompt, C, h, w)

        if do_classifier_free_guidance:
            # Prepend a zero-latent "unconditional" copy so CFG only needs one UNet pass.
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        """Pack SVD's 3 extra conditioning scalars into one tensor for the UNet's `added_time_ids`."""
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        # Sanity-check the UNet was actually built to accept 3 added-time scalars.
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)
        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        return add_time_ids

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        """VAE-decode denoised latents back to pixel frames."""
        latents = latents.flatten(0, 1)  # (B, T, C, H, W) -> (B*T, C, H, W)
        latents = 1 / self.vae.config.scaling_factor * latents

        # torch.compile wraps the module; unwrap to inspect the real forward signature.
        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # Decode decode_chunk_size frames at a time to avoid OOM.
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i:i + decode_chunk_size].shape[0]
            decode_kwargs = {"num_frames": num_frames_in} if accepts_num_frames else {}
            frames.append(self.vae.decode(latents[i:i + decode_chunk_size], **decode_kwargs).sample)
        frames = torch.cat(frames, dim=0)  # (B*T, C, H, W)

        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        return frames.float()  # cast to fp32; cheap and safe under bfloat16 too

    def check_inputs(self, image, height, width):
        """Validate `image`'s type and that `height`/`width` are divisible by 8 (VAE stride)."""
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.Tensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        """Sample the initial noisy latents, or reuse caller-provided ones.

        Args:
            num_channels_latents: UNet `in_channels` (8 = 4 noise + 4 image-cond);
                only the first half is actually sampled as noise (see below).
            latents: If given, used as-is (moved to device, then rescaled) instead
                of sampling fresh noise.
        """
        # num_channels_latents is the UNet's in_channels (8 = 4 noise + 4 image-cond),
        # so the sampled noise only needs the first half.
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (B, T, 4, h, w) latent-space shape
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # sample fresh gaussian noise
        else:
            latents = latents.to(device)  # caller supplied latents; just move to the right device

        latents = latents * self.scheduler.init_noise_sigma  # scale to the scheduler's expected initial noise level
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        # guidance_scale == 1 means no CFG (unconditional branch is skipped).
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.Tensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
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
    ):
        """Generate a video conditioned on `image`.

        Args:
            min_guidance_scale / max_guidance_scale: CFG scale is linearly ramped
                across frames from min to max (more guidance on later frames).
            motion_bucket_id: Higher values bias toward more motion in the output.
            noise_aug_strength: Noise added to the conditioning image; higher values
                let the video diverge further from the init image.
            decode_chunk_size: VAE-decode this many frames at a time to bound memory;
                defaults to decoding all frames at once.

        Returns:
            `StableVideoDiffusionPipelineOutput` if `return_dict=True`, else a
            `tuple` whose first element is the generated frames.

        Examples:
        """
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        self._guidance_scale = max_guidance_scale

        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # SVD was trained conditioned on fps - 1.
        fps = fps - 1

        image = self.video_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise  # matches training-time noise augmentation on the conditioning frame

        # Some VAEs are numerically unstable in fp16; upcast for encode/decode, then cast back.
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image, device=device, num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Broadcast the (single) conditioning-image latent across all output frames
        # so it can be concatenated with the per-frame noise along the channel dim.
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # (B, T, C, H, W)

        added_time_ids = self._get_add_time_ids(
            fps, motion_bucket_id, noise_aug_strength, image_embeddings.dtype,
            batch_size, num_videos_per_prompt, self.do_classifier_free_guidance,
        ).to(device)

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt, num_frames, num_channels_latents,
            height, width, image_embeddings.dtype, device, generator, latents,
        )

        # Per-frame CFG scale, ramped linearly across the clip.
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        # num_warmup_steps accounts for schedulers whose .order > 1 (e.g. multistep
        # solvers need >1 internal step per returned timestep) when deciding which
        # iterations count as a full progress-bar step.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # latents: (B, T, 4, h, w)
                # Duplicate the batch for CFG: one unconditional pass, one conditional.
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents  # (B or 2B, T, 4, h, w)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)  # scheduler-specific input scaling
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)  # (B or 2B, T, 8, h, w): + image cond

                # Predict the noise residual at this timestep.
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]  # (B or 2B, T, 4, h, w)

                if self.do_classifier_free_guidance:
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
                if XLA_AVAILABLE:
                    xm.mark_step()  # flush the lazily-built XLA graph after each step

        if output_type != "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()
        if not return_dict:
            return frames
        return StableVideoDiffusionPipelineOutput(frames=frames)


# --- Bicubic-with-antialiasing resize, used to match SVD's CLIP-image preprocessing ---

def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    """Downscale `input` to `size`, blurring first to reduce aliasing (naive interpolate does not)."""
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])  # downscale factor per spatial dim

    # Gaussian blur sigma/kernel-size chosen to approximate a 2-pass antialiased resize.
    sigmas = (max((factors[0] - 1.0) / 2.0, 0.001), max((factors[1] - 1.0) / 2.0, 0.001))
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)
    return torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


def _compute_padding(kernel_size):
    """Symmetric-as-possible padding (left, right, top, bottom) for an even/odd kernel."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]  # total padding needed per dim to keep "same" output size
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]  # iterate dims in reverse (F.pad expects last-dim-first order)
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front  # absorbs the extra pixel when computed_tmp is odd
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def _filter2d(input, kernel):
    """Apply a 2D convolution kernel per-channel (depthwise, reflect-padded) to `input`."""
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)  # (B_k, 1, kh, kw)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)  # (B_k, c, kh, kw): broadcast the kernel across channels
    height, width = tmp_kernel.shape[-2:]

    padding_shape: List[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")  # reflect-pad to keep "same" output size

    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)  # (B_k*c, 1, kh, kw): one group per channel
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))  # (1, B_k*c, H+pad, W+pad)
    # groups=tmp_kernel.size(0) makes this a depthwise conv: each channel only sees its own kernel.
    output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    return output.view(b, c, h, w)


def _gaussian(window_size: int, sigma):
    """1D Gaussian kernel of length `window_size`, normalized to sum to 1."""
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5  # even-length kernels have no exact center tap; offset to re-center
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    """Separable Gaussian blur: 1D blur along x, then along y (equivalent to a 2D blur, cheaper)."""
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)  # (1, 2): single (sigma_y, sigma_x) pair
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))  # (bs, kx) 1D kernel for the x pass
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))  # (bs, ky) 1D kernel for the y pass
    out_x = _filter2d(input, kernel_x[..., None, :])  # blur along width (kernel shape (bs, 1, kx))
    return _filter2d(out_x, kernel_y[..., None])  # then blur along height (kernel shape (bs, ky, 1))
