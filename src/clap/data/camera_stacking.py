"""Multi-camera vertical stacking: N views -> one (T, N*192, 320, C) frame.

All embodiments funnel through one of these stacking modes so the model always
sees a fixed-shape input regardless of how many physical cameras an embodiment has.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

VIEW_H, VIEW_W = 192, 320  # per-view size after resize
STACK_H, STACK_W = 576, 320  # three-slot stack size (model input for 3-view embodiments)


def _resize_view(view, h, w):
    """(T, Hi, Wi, C) uint8 -> (T, h, w, C) uint8, per-frame bilinear resize."""
    return np.stack([
        cv2.resize(view[t], (w, h), interpolation=cv2.INTER_LINEAR)
        for t in range(view.shape[0])
    ])


def tile_to_stack(view):
    """Single camera -> 3 identical vertical slots, (T, 576, 320, C).

    Avoids the aspect-ratio distortion a single resize-to-576 would cause,
    and matches the spatial size of the multi-camera stacking modes.
    """
    v = _resize_view(view, VIEW_H, VIEW_W)  # (T, 192, 320, C)
    return np.concatenate([v, v, v], axis=1)  # (T, 576, 320, C)


def stack_vertical(*views):
    """(T, H, W, C) views with matching H/W -> (T, n*H, W, C) via concat, no resize."""
    return np.concatenate(views, axis=1)


def stack_three_view_vertical(right, left, wrist):
    """3 distinct cameras -> (T, 576, 320, C): right top, left middle, wrist bottom."""
    r = _resize_view(right, VIEW_H, VIEW_W)
    l = _resize_view(left, VIEW_H, VIEW_W)
    w = _resize_view(wrist, VIEW_H, VIEW_W)
    return stack_vertical(r, l, w)  # (T, 576, 320, C)


def stack_two_view_tiled(right, left):
    """2 cameras -> (T, 576, 320, C): scene (right) on top, wrist (left) fills middle+bottom.

    Wrist occupies 2 of 3 slots so total height still matches the 3-slot layout.
    """
    r = _resize_view(right, VIEW_H, VIEW_W)  # scene -> top
    l = _resize_view(left, VIEW_H, VIEW_W)  # wrist -> middle + bottom
    return np.concatenate([r, l, l], axis=1)  # (T, 576, 320, C)


def stack_four_view_vertical(right_high, left_high, right_wrist, left_wrist):
    """4 distinct cameras (G1 humanoid) -> (T, 768, 320, C), each view resized to 192x320.

    Slot order top->bottom: right_high, left_high, right_wrist, left_wrist.
    """
    rh = _resize_view(right_high, VIEW_H, VIEW_W)
    lh = _resize_view(left_high, VIEW_H, VIEW_W)
    rw = _resize_view(right_wrist, VIEW_H, VIEW_W)
    lw = _resize_view(left_wrist, VIEW_H, VIEW_W)
    return np.concatenate([rh, lh, rw, lw], axis=1)  # (T, 768, 320, C)


STACKERS = {
    "four_view": stack_four_view_vertical,
    "three_view": stack_three_view_vertical,
    "two_view": stack_two_view_tiled,
    None: tile_to_stack,  # single camera
}


def view_slices_for_stacking(stacking_mode, H, W):
    """Return [(slot_name, h0, h1, w0, w1), ...] for each vertical slot in a stacked frame.

    Used by eval code to crop out one camera's region from a stacked prediction/GT frame.
    """
    if stacking_mode == "four_view":  # G1 humanoid: 4 equal slots
        h = H // 4
        return [
            ("right_high", 0, h, 0, W),
            ("left_high", h, 2 * h, 0, W),
            ("right_wrist", 2 * h, 3 * h, 0, W),
            ("left_wrist", 3 * h, H, 0, W),
        ]
    h = H // 3
    if stacking_mode == "three_view":  # right/left/wrist, one slot each
        return [("right", 0, h, 0, W), ("left", h, 2 * h, 0, W), ("wrist", 2 * h, H, 0, W)]
    if stacking_mode == "two_view":  # wrist duplicated into the bottom two slots
        return [("right", 0, h, 0, W), ("left", h, 2 * h, 0, W), ("left_bot", 2 * h, H, 0, W)]
    # single-camera (tile_to_stack): all three slots are the same view
    return [("slot0", 0, h, 0, W), ("slot1", h, 2 * h, 0, W), ("slot2", 2 * h, H, 0, W)]


def frames_to_video_tensor(frames_np, video_size):
    """(T, H, W, C) uint8 -> (C, T, H', W') uint8, resized to `video_size`.

    Stays uint8 (rather than normalizing here) so the dataset is cheap to
    load; the model normalizes on GPU when it VAE-encodes the batch.
    """
    frames = torch.from_numpy(frames_np.astype(np.uint8)).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    frames = F.interpolate(frames, size=video_size, mode="bilinear", align_corners=False)
    frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
    return frames.permute(1, 0, 2, 3).contiguous()  # (C, T, H', W')
