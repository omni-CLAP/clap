"""Unit tests for clap.data.action_caption: per-frame CLIP caption formatting."""

import numpy as np

from clap.data.action_caption import (
    format_action_caption,
    format_relative_action_caption,
    relativize_action_window,
)


def test_format_action_caption_basic():
    caption = format_action_caption(0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0)
    assert caption == "x=0.100 y=-0.200 z=0.300 roll=-0.400 pitch=0.500 yaw=-0.600 grip=1.000."


def test_relativize_action_window_anchor_row_is_zero_motion():
    norm_action = np.random.uniform(-1, 1, size=(5, 7)).astype(np.float32)
    anchor_idx = 2
    rel = relativize_action_window(norm_action, anchor_idx)
    assert np.allclose(rel[anchor_idx, :6], 0.0, atol=1e-6)


def test_relativize_action_window_gripper_column_untouched():
    norm_action = np.random.uniform(-1, 1, size=(5, 7)).astype(np.float32)
    rel = relativize_action_window(norm_action, anchor_idx=0)
    assert np.array_equal(rel[:, 6], norm_action[:, 6])


def test_relativize_action_window_wraps_rotation_deltas():
    # roll goes from -0.9 (anchor) to 0.9: naive delta is 1.8, should wrap to -0.2.
    norm_action = np.array([[0, 0, 0, -0.9, 0, 0, 0], [0, 0, 0, 0.9, 0, 0, 0]], dtype=np.float32)
    rel = relativize_action_window(norm_action, anchor_idx=0)
    assert np.isclose(rel[1, 3], -0.2, atol=1e-6)


def test_format_relative_action_caption_no_negative_zero():
    # A tiny negative value that rounds to 0 at decimals=0 must format as "0", not "-0".
    caption = format_relative_action_caption(-0.0001, 0, 0, 0, 0, 0, 0)
    assert "x=-0" not in caption
    assert "x=0" in caption


def test_format_relative_action_caption_scales_motion_and_grip_differently():
    caption = format_relative_action_caption(0.001, 0, 0, 0, 0, 0, 0.01)
    # x: 0.001 * 1000 = 1; grip: 0.01 * 100 = 1.
    assert "x=1" in caption
    assert "grip=1" in caption
