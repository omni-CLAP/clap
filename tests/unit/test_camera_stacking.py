"""Unit tests for clap.data.camera_stacking: shape/slot correctness for every stacking mode."""

import numpy as np
import pytest

from clap.data.camera_stacking import (
    STACKERS,
    VIEW_H,
    VIEW_W,
    _resize_view,
    frames_to_video_tensor,
    tile_to_stack,
    view_slices_for_stacking,
)


def _fake_view(t=4, h=100, w=200):
    return np.random.randint(0, 255, size=(t, h, w, 3), dtype=np.uint8)


def test_tile_to_stack_shape():
    out = tile_to_stack(_fake_view())
    assert out.shape == (4, 3 * VIEW_H, VIEW_W, 3)


def test_stack_three_view_shape_and_slot_order():
    right, left, wrist = _fake_view(), _fake_view(), _fake_view()
    out = STACKERS["three_view"](right, left, wrist)
    assert out.shape == (4, 3 * VIEW_H, VIEW_W, 3)
    # slot order top->bottom is right, left, wrist.
    assert np.array_equal(out[:, :VIEW_H], _resize_view(right, VIEW_H, VIEW_W))
    assert np.array_equal(out[:, VIEW_H:2 * VIEW_H], _resize_view(left, VIEW_H, VIEW_W))
    assert np.array_equal(out[:, 2 * VIEW_H:], _resize_view(wrist, VIEW_H, VIEW_W))


def test_stack_two_view_wrist_fills_two_slots():
    right, left = _fake_view(), _fake_view()
    out = STACKERS["two_view"](right, left)
    assert out.shape == (4, 3 * VIEW_H, VIEW_W, 3)
    # middle and bottom slots are both the (resized) left/wrist view.
    middle = out[:, VIEW_H:2 * VIEW_H]
    bottom = out[:, 2 * VIEW_H:3 * VIEW_H]
    assert np.array_equal(middle, bottom)


def test_stack_four_view_shape():
    views = [_fake_view() for _ in range(4)]
    out = STACKERS["four_view"](*views)
    assert out.shape == (4, 4 * VIEW_H, VIEW_W, 3)


@pytest.mark.parametrize("stacking_mode,n_slots", [("four_view", 4), ("three_view", 3), ("two_view", 3), (None, 3)])
def test_view_slices_for_stacking_covers_full_height_no_overlap(stacking_mode, n_slots):
    H, W = (n_slots * VIEW_H if stacking_mode != "four_view" else 4 * VIEW_H), VIEW_W
    slices = view_slices_for_stacking(stacking_mode, H, W)
    assert len(slices) == n_slots
    # slots must be contiguous and non-overlapping, covering [0, H).
    covered = 0
    for _name, y0, y1, x0, x1 in slices:
        assert y0 == covered
        assert x0 == 0 and x1 == W
        covered = y1
    assert covered == H


def test_frames_to_video_tensor_shape_and_dtype():
    frames = _fake_view(t=5, h=100, w=200)
    out = frames_to_video_tensor(frames, video_size=(50, 60))
    assert out.shape == (3, 5, 50, 60)  # (C, T, H, W)
    assert out.dtype.is_floating_point is False  # uint8