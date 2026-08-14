import contextlib
import logging

import einops
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModelWithProjection

from clap.config.model import CLAPModelConfig
from clap.models.action_adapter import build_action_adapter
from clap.models.action_encoder import ActionEncoder, clip_encode_strings
from clap.models.pipeline_svd import StableVideoDiffusionPipeline
from clap.models.unet import UNetSpatioTemporalConditionModel
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_transformers_load_report():
    """Silence transformers' `from_pretrained` "LOAD REPORT" (transformers>=5), which
    logs at WARNING from the `transformers.modeling_utils` logger. Expected/benign here:
    CLIPTextModelWithProjection.from_pretrained loads a CLIPTextModelWithProjection
    checkpoint that also contains the vision-tower weights of the full CLIPModel it was
    exported from, so those show up as "unexpected" even though nothing is actually wrong.
    """
    modeling_utils_logger = logging.getLogger("transformers.modeling_utils")
    previous_level = modeling_utils_logger.level
    modeling_utils_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        modeling_utils_logger.setLevel(previous_level)


class CLAPModel(nn.Module):
    """Action-conditioned video world model: an SVD backbone with an action-conditioned UNet.

    Composed of a frozen VAE/image-encoder/CLIP text encoder (from the SVD
    pipeline), a swapped-in `UNetSpatioTemporalConditionModel` that accepts
    action conditioning + frame-level pose, and an `ActionEncoder` mapping raw
    actions to the UNet's cross-attention hidden states. `forward` implements
    an EDM-style diffusion loss over video latents.

    Args:
        config: See `CLAPModelConfig` for the conditioning/adapter knobs that
            change which branch of `forward` runs.
    """

    def __init__(self, config: CLAPModelConfig):
        super().__init__()
        self.config = config
        self.conditioning = config.conditioning
        self.action_dim = config.action_dim

        # Load the pretrained SVD pipeline for its VAE/image-encoder/scheduler weights,
        # then swap in CLAP's action-conditioned UNet (loading what overlaps by shape;
        # the extra conditioning layers start from their own random init).
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(config.svd_model_path)
        unet = UNetSpatioTemporalConditionModel()
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        self.pipeline.unet = unet

        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # VAE and CLIP image-encoder stay frozen (used only to encode/decode pixels);
        # the UNet is the only pretrained component actually being trained here.
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        self.unet.enable_gradient_checkpointing()

        # Separate frozen CLIP text tower, used for task-description/language conditioning.
        # "LOAD REPORT" is suppressed: the checkpoint is a full CLIPModel, so its vision-tower
        # weights are always "unexpected" here by design -- see _suppress_transformers_load_report.
        with _suppress_transformers_load_report():
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(config.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.clip_model_path, use_fast=False)
        self.text_encoder.requires_grad_(False)

        # Maps the raw per-frame action (dim = action_dim) to the UNet's
        # cross-attention hidden dim (1024); trained jointly with the UNet.
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim,
            action_num=config.num_total_frames,
            hidden_size=1024,
            text_cond=config.text_cond,
            deep=config.deep_action_encoder,
        )
        if self.conditioning == "language":
            # Language conditioning uses _forward_language instead of action_encoder,
            # so its params would never receive grads and DDP would error on step 1.
            self.action_encoder.requires_grad_(False)

        # Optional adapter: projects an embodiment's own action representation into
        # the LAM-latent space a pretrained action_encoder was trained on, for
        # cross-embodiment adaptation without retraining action_encoder itself.
        self.action_adapter = None
        if config.train_action_adapter:
            # Only the adapter trains, against an otherwise-frozen, LAM-pretrained backbone.
            self.unet.requires_grad_(False)
        elif config.use_action_adapter:
            # Reconstructs the adapter module so a checkpoint trained with one loads correctly.
            self.action_adapter = build_action_adapter(
                arch=config.adapter_arch,
                in_dim=config.adapter_input_dim,
                out_dim=self.action_dim,
                hidden_dim=config.adapter_hidden_dim,
                num_layers=config.adapter_num_layers,
                num_heads=config.adapter_num_heads,
                max_seq_len=config.num_total_frames,
                dropout=config.adapter_dropout,
            )

    def encode_video_to_latent(self, video, view_h=192, encode_chunk_size=64):
        """Encode raw frames (B, C, T, H, W) uint8 to (B, T, 4, H/8, W/8) latents.

        Multi-camera stacked video (H = n_cams * view_h) is encoded one camera
        at a time and restacked along H, matching the layout of pre-encoded
        per-camera latents.

        Args:
            encode_chunk_size: max frames per VAE call; caps peak VRAM on
                long episodes at the cost of more kernel launches.
        """
        # uint8 pixels -> [-1, 1] float; already-float input is assumed pre-normalized.
        if video.dtype == torch.uint8:
            video = video.to(dtype=self.vae.dtype) / 127.5 - 1.0
        else:
            video = video.to(dtype=self.vae.dtype)

        B, C, T, H, W = video.shape
        n_cams = H // view_h  # stacked cameras share one tensor, split by height

        def _encode_flat(frames_bcthw):
            # frames_bcthw: (B, C, T, H, W)
            b, c, t, h, w = frames_bcthw.shape
            flat = einops.rearrange(frames_bcthw, "b c t h w -> (b t) c h w")  # (B*T, C, H, W)
            with torch.no_grad():
                chunk_size = max(1, min(encode_chunk_size, t))
                # Encode in chunks of chunk_size frames to bound peak VRAM.
                parts = [
                    self.vae.encode(flat[i:i + chunk_size]).latent_dist.sample()
                    for i in range(0, flat.shape[0], chunk_size)
                ]
                lat = torch.cat(parts, dim=0) * self.vae.config.scaling_factor  # (B*T, 4, H/8, W/8)
            return einops.rearrange(lat, "(b t) c h w -> b t c h w", b=b, t=t)  # (B, T, 4, H/8, W/8)

        if n_cams <= 1:
            return _encode_flat(video)

        # Multi-camera: encode each view separately (VAE has no notion of the
        # stacked layout), then restack along H to match pre-encoded latents.
        cam_latents = [
            _encode_flat(video[:, :, :, i * view_h:(i + 1) * view_h, :])
            for i in range(n_cams)
        ]
        return torch.cat(cam_latents, dim=3)

    def _forward_language(self, batch, device, dtype, num_total):
        """CLIP-encode per-step captions -> (B, T, 1024) conditioning.

        Expects `batch["action_caption_steps"]` as a (B, T) list of strings.
        """
        steps = batch.get("action_caption_steps")
        assert steps is not None, "language conditioning requires 'action_caption_steps' in batch"

        # Flatten (B, T) captions into one list so CLIP encodes them in a single batched call.
        B = len(steps)
        flat_strs = []
        for b in range(B):
            assert len(steps[b]) == num_total, f"expected {num_total} per-step strings, got {len(steps[b])}"
            flat_strs.extend(steps[b])

        flat_emb = clip_encode_strings(flat_strs, self.tokenizer, self.text_encoder)  # (B*T, clip_dim)
        flat_emb = einops.repeat(flat_emb, "bt c -> bt (n c)", n=2)  # (B*T, 1024): match action_encoder's hidden dim
        action_hidden = einops.rearrange(flat_emb, "(b t) c -> b t c", b=B, t=num_total)  # (B, T, 1024)

        # Optionally add a shared task-description embedding on top of every per-step caption.
        if self.config.text_cond and batch.get("text") is not None:
            task_emb = clip_encode_strings(batch["text"], self.tokenizer, self.text_encoder)  # (B, clip_dim)
            task_emb = einops.repeat(task_emb, "b c -> b 1 (n c)", n=2)  # (B, 1, 1024)
            action_hidden = action_hidden + task_emb

        return action_hidden.to(dtype=dtype)

    def _forward_numeric_action(self, batch, device, dtype):
        """Encode a numeric per-frame action (EE/LAM/joint-space/...) -> conditioning."""
        action = batch["action"].to(device)  # (B, T, action_dim)
        if self.action_adapter is not None:
            # batch['action'] is the embodiment-specific action; the adapter
            # projects it into the LAM-latent space action_encoder expects.
            adapter_dtype = next(self.action_adapter.parameters()).dtype
            action = self.action_adapter(action.to(adapter_dtype)).to(dtype)
        texts = batch.get("text")  # optional shared task description, added inside action_encoder
        return self.action_encoder(
            action, texts, self.tokenizer, self.text_encoder,
            frame_level_cond=self.config.frame_level_cond,
        )

    def forward(self, batch):
        """Compute the training loss for one batch.

        Returns:
            (loss, aux): aux is an unused placeholder (always 0.0), kept only so
            callers can unpack a 2-tuple; only `loss` is ever consumed.
        """
        dtype = self.unet.dtype
        device = self.unet.device
        P_mean, P_std = 0.7, 1.6
        noise_aug_strength = 0.0

        num_history = self.config.num_history
        num_total = self.config.num_total_frames

        # Most datasets pre-encode frames to VAE latents at prep time; only
        # encode on the fly here if raw pixel frames were given instead.
        if "latent" in batch:
            latents = batch["latent"].to(device)
        else:
            latents = self.encode_video_to_latent(batch["video"].to(device)).to(dtype=dtype)

        bsz, num_frames = latents.shape[:2]  # latents: (B, T, 4, H/8, W/8)
        assert num_frames == num_total, f"expected {num_total} frames, got {num_frames}"

        # Last history frame becomes the per-frame image condition (broadcast to
        # every output frame). Lightly noised (small random sigma, not the main
        # diffusion sigma below) so the model doesn't learn to trivially copy a
        # pristine conditioning frame instead of actually predicting motion.
        current_img = latents[:, num_history:(num_history + 1)][:, 0]  # (B, 4, H/8, W/8)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in * (current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, "b c h w -> b f c h w", f=num_frames)  # (B, T, 4, H/8, W/8)
        if self.config.his_cond_zero:
            # Ablation/robustness option: drop the image condition entirely for
            # history frames, forcing more reliance on the action conditioning.
            condition_latent[:, :num_history] = 0.0

        if self.conditioning == "language":
            action_hidden = self._forward_language(batch, device, dtype, num_total)
        else:
            action_hidden = self._forward_numeric_action(batch, device, dtype)

        # Classifier-free guidance: drop conditioning to zero on ~5% of samples.
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device) > 0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden * text_mask + uncond_hidden_states * (~text_mask)

        # EDM (Karras et al.) preconditioning: sample log-normal sigma, derive
        # skip/out/in scales and the loss weight that keeps the target ~unit variance.
        rnd_normal = torch.randn([bsz, 1, 1, 1, 1], device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        c_skip = 1 / (sigma**2 + 1)
        c_out = -sigma / (sigma**2 + 1) ** 0.5
        c_in = 1 / (sigma**2 + 1) ** 0.5
        c_noise = (sigma.log() / 4).reshape([bsz])
        loss_weight = (sigma**2 + 1) / sigma**2
        noisy_latents = latents + torch.randn_like(latents) * sigma

        # History frames get their own (lighter, fixed-scale) noise so the model
        # learns to condition on slightly-corrupted past frames, not ground truth.
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        history = latents[:, :num_history]
        noisy_history = 1 / (sigma_h**2 + 1) ** 0.5 * (history + sigma_h * torch.randn_like(history))
        input_latents = torch.cat([noisy_history, c_in * noisy_latents[:, num_history:]], dim=1)  # (B, T, 4, H/8, W/8)
        # Concat the (unnoised) image condition along the channel dim -> 8-channel UNet input.
        input_latents = torch.cat([input_latents, condition_latent / self.vae.config.scaling_factor], dim=2)  # (B, T, 8, H/8, W/8)

        # SVD's extra micro-conditioning (fps/motion_bucket_id/noise_aug_strength),
        # sinusoidally embedded inside the UNet and added to the timestep embedding.
        # do_classifier_free_guidance=False since CFG here is handled via action_hidden
        # masking above, not by doubling the batch.
        added_time_ids = self.pipeline._get_add_time_ids(
            self.config.fps, self.config.motion_bucket_id, noise_aug_strength,
            action_hidden.dtype, bsz, 1, False,
        ).to(device)

        # UNet predicts the denoising residual; combine with the noisy input via
        # EDM's c_skip/c_out to get the model's estimate of the clean latents (x0).
        model_pred = self.unet(
            input_latents, c_noise,
            encoder_hidden_states=action_hidden,
            added_time_ids=added_time_ids,
            frame_level_cond=self.config.frame_level_cond,
        ).sample
        predict_x0 = c_out * model_pred + c_skip * noisy_latents

        # Only future frames are supervised; history is context, not a prediction target.
        loss = ((predict_x0[:, num_history:] - latents[:, num_history:]) ** 2 * loss_weight).mean()
        return loss, torch.tensor(0.0, device=device, dtype=dtype)
