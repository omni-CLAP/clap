"""Autoregressive rollout stepping core, shared by offline replay and (later) real-robot deployment.

Loads a `CLAPModel` checkpoint once, then exposes video<->latent conversion,
one full episode's autoregressive chunked rollout, and GT-vs-prediction metrics.
"""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from clap.data.action_caption import format_action_caption, format_relative_action_caption, relativize_action_window
from clap.data.base import BoundNormalizer
from clap.data.camera_stacking import VIEW_H, view_slices_for_stacking
from clap.models import CLAPModel, CLAPDiffusionPipeline

_lpips_model = None


def _get_lpips(device):
    """Lazily construct a single frozen LPIPS(alex) instance, shared across calls."""
    global _lpips_model
    if _lpips_model is None:  # first call: build and cache the model
        import lpips
        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()  # frozen AlexNet-based perceptual metric
        for p in _lpips_model.parameters():
            p.requires_grad_(False)  # freeze weights, inference-only
    return _lpips_model  # cached instance


def _select_history(buffer, history_idx):
    """Pick custom history frames from `buffer` by index; negative indices count from the end (clipped to [0, L))."""
    L = buffer.shape[0]  # number of frames currently in the buffer
    idx = [min(i, L - 1) if i >= 0 else max(0, L + i) for i in history_idx]  # resolve/clip each requested index
    return buffer[idx]  # gathered history frames


class CLAPRolloutAgent:
    """Loads a checkpoint once and autoregressively replays episodes against it.

    Args:
        family: Which conditioning family this checkpoint/config pair expects
            ("ee" | "bimanual_yam" | "g1_humanoid" | "language" | "lam") -- selects a
            `clap.data.rollout_loaders.ROLLOUT_LOADERS` class, not a per-dataset value;
            bridge/taco_play/etc. all use "ee" too (dataset-specific camera/video handling
            comes from `clap.data.oxe_catalog`, keyed by dataset_name, not family).
        action_caption_mode: Only read when family="language"; see `clap.data.action_caption`.
    """

    def __init__(self, model_config, ckpt_path, family, action_caption_mode="absolute", device=None, dtype=torch.bfloat16):
        """Load `ckpt_path` into a fresh `CLAPModel`, non-strict, and set it to eval mode.

        Args:
            device: Defaults to CUDA if available, else CPU.
            dtype: Precision used for both the model weights and inference tensors.
        """
        self.family = family
        self.action_caption_mode = action_caption_mode
        self.conditioning = model_config.conditioning
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.model = CLAPModel(model_config)
        # Inference-only load: always non-strict (missing/extra keys tolerated),
        # no optimizer/step restore — this agent never trains.
        blob = torch.load(ckpt_path, map_location="cpu")
        state_dict = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).to(self.dtype).eval()

    # ------------------------------------------------------------------
    # Video <-> latent conversion
    # ------------------------------------------------------------------

    def encode_video(self, video):
        """(C, T, H, W) uint8 -> (T, 4, h, w) latents, via the model's own per-view VAE encoder."""
        with torch.no_grad():
            batched = video.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
            latents = self.model.encode_video_to_latent(batched)  # (1, T, 4, h, w)
        return latents[0].to(self.dtype)

    def decode_latents(self, latents, decode_chunk_size=7):
        """(T, 4, n_views*h, w) -> (T, 3, n_views*H_view, W) pixels in [-1, 1], decoded one view at a time.

        Views are decoded independently (matching how they were encoded) and
        restacked along pixel-H; `n_views` is inferred from the latent height
        so this works for 3-slot (three_view/two_view/single) and 4-slot
        (four_view, G1 humanoid) layouts alike.
        """
        vae = self.model.vae
        T, _C, H_lat, W_lat = latents.shape
        per_view_h_lat = VIEW_H // 8  # VAE downsamples 8x
        n_views = H_lat // per_view_h_lat

        decoded_views = []
        for v in range(n_views):
            view_latents = latents[:, :, v * per_view_h_lat:(v + 1) * per_view_h_lat, :]
            chunks = []
            for i in range(0, T, decode_chunk_size):
                chunk = view_latents[i:i + decode_chunk_size] / vae.config.scaling_factor
                chunks.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample)
            decoded_views.append(torch.cat(chunks, dim=0))  # (T, 3, H_view, W)
        return torch.cat(decoded_views, dim=2)  # (T, 3, n_views*H_view, W)

    # ------------------------------------------------------------------
    # Episode preprocessing
    # ------------------------------------------------------------------

    def _trim_static_prefix(self, ep, speed_frac=0.15, speed_floor=0.0005, lead=1):
        """Drop leading near-static frames, so replay starts near the first real motion."""
        from clap.data.rollout_loaders.ee_loader import motion_onset

        states = ep.get("states")
        if states is None or len(states) < 6:
            return ep  # too short to estimate an onset; leave unchanged
        motion_cols = states if ep.get("action_mode") == "joint14" else states[:, :3]  # xyz only for cartesian embodiments
        onset = motion_onset(motion_cols, speed_frac, speed_floor)
        start = max(0, onset - lead)
        if start <= 0:
            return ep
        trimmed = dict(ep)
        for key in ("video", "states", "grippers", "actions", "frame_ids"):
            if ep.get(key) is not None:
                trimmed[key] = ep[key][:, start:] if key == "video" else ep[key][start:]
        return trimmed

    def _skip_frames(self, ep, n):
        """Drop the first `n` frames unconditionally (a simpler alternative to static-prefix trimming)."""
        if n <= 0:
            return ep  # nothing to skip
        skipped = dict(ep)  # shallow copy so we don't mutate the caller's episode dict
        for key in ("video", "states", "grippers", "actions", "frame_ids"):
            if ep.get(key) is not None:
                skipped[key] = ep[key][:, n:] if key == "video" else ep[key][n:]  # video is (C, T, H, W), others are (T, ...)
        return skipped

    def _pad_episode(self, ep, gt_latents, num_history):
        """Front-pad `num_history` copies of frame 0, for cold-start conditioning at the start of an episode."""
        pad_latents = gt_latents[0:1].expand(num_history, -1, -1, -1).clone()  # repeat frame 0 latent num_history times
        padded_latents = torch.cat([pad_latents, gt_latents], dim=0)  # (num_history + T, 4, h, w)

        padded = dict(ep)  # shallow copy so we don't mutate the caller's episode dict
        if self.family == "lam":
            actions = ep["actions"]
            pad = np.tile(actions[0:1], (num_history, 1))  # repeat action row 0 num_history times
            padded["actions_padded"] = np.concatenate([pad, actions], axis=0)
        else:
            states = ep["states"]
            pad_states = np.tile(states[0:1], (num_history, 1))  # repeat state row 0 num_history times
            padded["states_padded"] = np.concatenate([pad_states, states], axis=0)
            if ep.get("grippers") is not None:
                grippers = ep["grippers"]
                pad_grip = np.tile(grippers[0:1], (num_history,) + (1,) * (grippers.ndim - 1))  # repeat gripper row 0
                padded["grippers_padded"] = np.concatenate([pad_grip, grippers], axis=0)
        return padded_latents, padded  # (num_history + T latents, episode dict with *_padded fields)

    # ------------------------------------------------------------------
    # Per-chunk conditioning
    # ------------------------------------------------------------------

    def _build_chunk_condition(self, ep_padded, s, T, num_history, history_idx=None):
        """Build this chunk's action/text conditioning, from window [s, s+T) of the padded episode.

        history_idx: If given, sparsely resample the window's HISTORY rows (the first
        num_history of the T total) from ep_padded's full arrays so far, via the same
        offsets `_select_history` applies to the latent buffer -- keeping each pose
        token aligned with the latent frame it actually conditions alongside instead of
        silently staying contiguous while the image history goes sparse. The FUTURE
        rows (num_frames being predicted) always stay the literal contiguous upcoming
        ones. Only meaningful for the ee/language path below -- LAM's transition-based
        convention (row 0 always zero) has no well-defined equivalent, so it's ignored there.
        """
        if self.family == "lam":
            action_dim = ep_padded["actions_padded"].shape[1]
            # Row 0 of every window is always zero: LAM actions encode the transition INTO
            # a frame, and there's no transition into the window's first (anchor) frame.
            action = np.zeros((T, action_dim), dtype=np.float32)
            action[1:] = ep_padded["actions_padded"][s + 1:s + T]
            action = torch.from_numpy(action).unsqueeze(0).to(self.dtype)
            return action, [ep_padded.get("text", "")]

        if history_idx is not None:
            states_hist = _select_history(ep_padded["states_padded"][:s + num_history], history_idx)
            states = np.concatenate([states_hist, ep_padded["states_padded"][s + num_history:s + T]], axis=0)
        else:
            states = ep_padded["states_padded"][s:s + T]

        if ep_padded.get("action_mode") in ("joint14", "joint26"):
            abs_action = states  # joint-space: raw state array is the action, no gripper concat
        else:
            if history_idx is not None:
                grip_hist = _select_history(ep_padded["grippers_padded"][:s + num_history], history_idx)
                grippers = np.concatenate([grip_hist, ep_padded["grippers_padded"][s + num_history:s + T]], axis=0)
            else:
                grippers = ep_padded["grippers_padded"][s:s + T]
            abs_action = np.concatenate([states[:, :6], grippers[:, None]], axis=-1)

        normalizer = BoundNormalizer(ep_padded["stat_path"])
        norm_action = normalizer.normalize(abs_action)

        if self.family == "language":
            if self.action_caption_mode == "relative":
                # Anchor = index num_history within the window ("now", same frame the image condition uses).
                rel = relativize_action_window(norm_action, num_history)
                steps = [format_relative_action_caption(*row) for row in rel]
            else:
                steps = [format_action_caption(*row) for row in norm_action]
            return None, {"action_caption_steps": [steps], "text": [ep_padded.get("text", "")]}

        action = torch.from_numpy(norm_action.astype(np.float32)).unsqueeze(0).to(self.dtype)
        return action, [ep_padded.get("text", "")]

    def _encode_action_hidden(self, action_cond, text_or_batch, num_total, frame_level_cond):
        """Encode this chunk's action/text conditioning into hidden states for the diffusion UNet.

        `conditioning="language"` routes through the model's language branch instead
        (action_cond is unused there — text is the only signal).
        """
        if self.conditioning == "language":
            return self.model._forward_language(text_or_batch, self.device, self.dtype, num_total)
        texts = text_or_batch if text_or_batch and text_or_batch[0] else None  # empty caption string means no text conditioning
        action_cond = action_cond.to(self.device)
        if self.model.action_adapter is not None:
            # Cross-embodiment adapter maps this action's raw dims into the model's expected action_dim.
            adapter_dtype = next(self.model.action_adapter.parameters()).dtype
            action_cond = self.model.action_adapter(action_cond.to(adapter_dtype)).to(self.dtype)
        return self.model.action_encoder(action_cond, texts, self.model.tokenizer, self.model.text_encoder, frame_level_cond)

    def predict_chunk(
        self, image, history, action_cond, text_or_batch, num_frames,
        num_inference_steps=50, guidance_scale=1.0, decode_chunk_size=7,
        fps=7, motion_bucket_id=127, frame_level_cond=True, his_cond_zero=False,
    ):
        """Denoise one chunk of `num_frames` future latents, given an image/history/action condition.

        The single-chunk building block both `autoregressive_replay` (dataset
        episodes) and interactive teleop (live keypresses) call — the only
        difference between those two callers is how `action_cond` gets built.

        Returns:
            (num_frames, 4, h, w) predicted latents.
        """
        action_hidden = self._encode_action_hidden(action_cond, text_or_batch, num_frames + history.shape[1], frame_level_cond)  # encode conditioning into hidden states
        _, latents = CLAPDiffusionPipeline.__call__(
            self.model.pipeline,
            image=image, action_hidden=action_hidden,
            width=image.shape[-1] * 8, height=image.shape[-2] * 8,  # VAE upsamples 8x back to pixel space
            num_frames=num_frames, history=history,
            num_inference_steps=num_inference_steps, decode_chunk_size=decode_chunk_size,
            max_guidance_scale=guidance_scale, fps=fps, motion_bucket_id=motion_bucket_id,
            output_type="latent", return_dict=False,  # keep output as latents, skip pixel decode here
            frame_level_cond=frame_level_cond, his_cond_zero=his_cond_zero,
        )
        return latents[0]  # drop the batch dim

    # ------------------------------------------------------------------
    # Autoregressive replay
    # ------------------------------------------------------------------

    def autoregressive_replay(
        self, ep, num_history, num_frames, num_inference_steps=50, guidance_scale=1.0,
        decode_chunk_size=7, fps=7, motion_bucket_id=127, frame_level_cond=True, his_cond_zero=False,
        max_chunks=0, kept_frames=0, gt_cond=False, history_idx=None,
        trim_static_prefix=False, skip_first_n_frames=0,
    ):
        """Roll out one full episode chunk by chunk, feeding each chunk's own prediction back in as history.

        Returns:
            {"gt_latents_full", "pred_latents", "gt_aligned", "num_chunks"}
        """
        if trim_static_prefix:
            ep = self._trim_static_prefix(ep)
        if skip_first_n_frames > 0:
            ep = self._skip_frames(ep, skip_first_n_frames)

        gt_latents_orig = self.encode_video(ep["video"])
        gt_latents, ep_padded = self._pad_episode(ep, gt_latents_orig, num_history)

        T_total = gt_latents.shape[0] - 1  # -1 keeps parity with LAM's shorter (transition-based) horizon
        T = num_history + num_frames
        # single-step replay (n_keep=2) isn't exposed via config yet; kept_frames
        # lets a caller keep fewer than num_frames per chunk if desired.
        n_keep = kept_frames if kept_frames > 0 else num_frames
        assert 1 < n_keep <= num_frames, f"n_keep={n_keep} must be in (1, {num_frames}]"

        length_key = "actions_padded" if self.family == "lam" else "states_padded"
        max_len = min(T_total, ep_padded[length_key].shape[0])
        if self.family != "lam":
            max_len -= 1  # aligns EE/language's per-frame-state length with LAM's transition-based one
        assert max_len >= T, f"episode too short for one chunk: max_len={max_len} < T={T}"

        chunk_advance = n_keep - 1
        max_chunks_possible = (max_len - T) // chunk_advance + 1
        num_chunks = min(max_chunks_possible, max_chunks) if max_chunks > 0 else max_chunks_possible

        buffer = None if gt_cond else gt_latents[:num_history].clone()  # cold-start: num_history copies of frame 0
        pred_chunks = []

        for c in range(num_chunks):
            s = c * chunk_advance
            if s + T > max_len:
                break

            if gt_cond:
                image = gt_latents[s + num_history].unsqueeze(0).to(self.dtype)
                history = (
                    _select_history(gt_latents[:s + num_history], history_idx) if (history_idx and c > 0)
                    else gt_latents[s:s + num_history]
                ).unsqueeze(0).to(self.dtype)
            else:
                # Conditioning frame is always the model's own last prediction (buffer's last frame),
                # except chunk 0 where the episode's true first frame hasn't been predicted yet.
                image = (gt_latents[num_history].unsqueeze(0) if c == 0 else buffer[-1:]).to(self.dtype)
                history = (
                    _select_history(buffer, history_idx) if (history_idx and c > 0)
                    else buffer[s:s + num_history]
                ).unsqueeze(0).to(self.dtype)

            action_cond, text_or_batch = self._build_chunk_condition(
                ep_padded, s, T, num_history,
                history_idx=history_idx if (history_idx and c > 0) else None,
            )
            pred = self.predict_chunk(
                image, history, action_cond, text_or_batch, num_frames,
                num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                decode_chunk_size=decode_chunk_size, fps=fps, motion_bucket_id=motion_bucket_id,
                frame_level_cond=frame_level_cond, his_cond_zero=his_cond_zero,
            )[:n_keep]
            pred_chunks.append(pred)

            if not gt_cond:
                # Chunk 0 contributes all n_keep frames; later chunks drop their first
                # frame (it duplicates the buffer's last frame / this chunk's image condition).
                buffer = torch.cat([buffer, pred] if c == 0 else [buffer, pred[1:]], dim=0)

        if not pred_chunks:
            raise RuntimeError("autoregressive_replay produced zero chunks — episode too short or max_chunks=0")

        # Stitch chunks: drop the duplicated first frame of every chunk after the first.
        stitched = [pred_chunks[0]] + [p[1:] for p in pred_chunks[1:]]
        pred_full = torch.cat(stitched, dim=0)
        n_pred = pred_full.shape[0]
        gt_aligned = gt_latents[num_history:num_history + n_pred]

        return {"gt_latents_full": gt_latents_orig, "pred_latents": pred_full, "gt_aligned": gt_aligned, "num_chunks": len(pred_chunks)}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self, pred_latents, gt_latents, stacking_mode=None, skip_first=1, decode_chunk_size=7):
        """Latent MSE + decoded-pixel PSNR/SSIM/LPIPS, both overall and per camera view.

        Args:
            skip_first: Frames to exclude from every metric (frame 0 is the
                conditioning image itself, not a real prediction).
        """
        latent_mse = ((pred_latents[skip_first:].float() - gt_latents[skip_first:].float()) ** 2).mean().item()

        pred_dec = self.decode_latents(pred_latents, decode_chunk_size)  # (T, 3, H, W) in [-1, 1]
        gt_dec = self.decode_latents(gt_latents, decode_chunk_size)
        # float32: numpy/skimage don't support bfloat16, and lpips/plain arithmetic are fine in fp32.
        pred01 = (pred_dec / 2 + 0.5).clamp(0, 1).float()
        gt01 = (gt_dec / 2 + 0.5).clamp(0, 1).float()
        T, _, H, W = pred01.shape

        lpips_model = _get_lpips(pred01.device)
        psnr_per_frame, ssim_per_frame = [], []
        for t in range(skip_first, T):
            g = gt01[t].permute(1, 2, 0).cpu().numpy()
            p = pred01[t].permute(1, 2, 0).cpu().numpy()
            psnr_per_frame.append(float(peak_signal_noise_ratio(g, p, data_range=1.0)))
            ssim_per_frame.append(float(structural_similarity(g, p, data_range=1.0, channel_axis=2)))
        with torch.no_grad():
            lpips_per_frame = lpips_model(pred01[skip_first:] * 2 - 1, gt01[skip_first:] * 2 - 1).flatten().tolist()

        psnr_mean_per_view, ssim_mean_per_view, lpips_mean_per_view = {}, {}, {}
        for vname, y0, y1, x0, x1 in view_slices_for_stacking(stacking_mode, H, W):
            g_view = gt01[skip_first:, :, y0:y1, x0:x1]
            p_view = pred01[skip_first:, :, y0:y1, x0:x1]
            v_psnr, v_ssim = [], []
            for t in range(g_view.shape[0]):
                g = g_view[t].permute(1, 2, 0).cpu().numpy()
                p = p_view[t].permute(1, 2, 0).cpu().numpy()
                v_psnr.append(float(peak_signal_noise_ratio(g, p, data_range=1.0)))
                v_ssim.append(float(structural_similarity(g, p, data_range=1.0, channel_axis=2)))
            with torch.no_grad():
                v_lpips = lpips_model(p_view * 2 - 1, g_view * 2 - 1).flatten().mean().item()
            psnr_mean_per_view[vname] = float(np.mean(v_psnr))
            ssim_mean_per_view[vname] = float(np.mean(v_ssim))
            lpips_mean_per_view[vname] = v_lpips

        metrics = {
            "psnr_mean": float(np.mean(psnr_per_frame)), "ssim_mean": float(np.mean(ssim_per_frame)),
            "lpips_mean": float(np.mean(lpips_per_frame)), "latent_mse": latent_mse,
            "psnr_per_frame": psnr_per_frame, "ssim_per_frame": ssim_per_frame, "lpips_per_frame": lpips_per_frame,
            "num_frames": T,
            "psnr_mean_per_view": psnr_mean_per_view, "ssim_mean_per_view": ssim_mean_per_view, "lpips_mean_per_view": lpips_mean_per_view,
        }
        pred_u8 = (pred01.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        gt_u8 = (gt01.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return metrics, pred_u8, gt_u8
