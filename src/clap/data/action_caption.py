"""Per-frame action -> CLIP caption formatting, for language conditioning.

Two modes: "absolute" formats the normalized [-1,1] state directly; "relative"
re-baselines each window against its own anchor frame first (better caption
resolution for small motions, at the cost of losing absolute position info).
"""

ACTION_CAPTION_DECIMALS = 3
RELATIVE_MOTION_SCALE = 1000.0  # display scale for relative motion dims (xyz+rpy)
RELATIVE_GRIP_SCALE = 100.0  # separate (smaller) scale: gripper stays an absolute [-1,1] state
RELATIVE_ACTION_CAPTION_DECIMALS = 0


def relativize_action_window(norm_action, anchor_idx):
    """Re-baseline a window of normalized absolute actions against the anchor frame's own action.

    Args:
        norm_action: (T, 7) array, already normalized to [-1,1].
        anchor_idx: Index of the "current"/conditioning frame within the window;
            every other row's motion dims become a delta relative to this one.

    Only the 6 motion dims (xyz+rpy) are relativized; gripper (column 6) stays
    absolute since open/closed state is more informative than its delta.
    Rotation dims are wrapped to (-1, 1] after subtraction, since a raw angle
    crossing the +-pi boundary would otherwise show up as a spurious ~2.0 jump.
    """
    rel = norm_action.copy()
    rel[:, :6] = norm_action[:, :6] - norm_action[anchor_idx, :6]
    rel[:, 3:6] = ((rel[:, 3:6] + 1.0) % 2.0) - 1.0  # wrap rotation deltas to (-1, 1]
    return rel


def format_action_caption(x, y, z, roll, pitch, yaw, grip, decimals=ACTION_CAPTION_DECIMALS):
    """Format one normalized 7-d action row into a per-frame CLIP caption.

    No leading "stepN:" prefix — captions are encoded one per frame independently,
    so the frame index is already carried by tensor position.
    """
    d = decimals
    return (
        f"x={x:.{d}f} y={y:.{d}f} z={z:.{d}f} "
        f"roll={roll:.{d}f} pitch={pitch:.{d}f} yaw={yaw:.{d}f} "
        f"grip={grip:.{d}f}."
    )


def _fmt_no_neg_zero(value, decimals):
    """Round-then-format, clearing the sign on values that round to zero (avoids a literal '-0')."""
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0  # clear sign bit so -0.0 doesn't format as "-0"
    return f"{rounded:.{decimals}f}"


def format_relative_action_caption(
    x, y, z, roll, pitch, yaw, grip,
    motion_scale=RELATIVE_MOTION_SCALE, grip_scale=RELATIVE_GRIP_SCALE,
    decimals=RELATIVE_ACTION_CAPTION_DECIMALS,
):
    """Format one `relativize_action_window` row into a per-frame CLIP caption.

    Uses decimals=0 with a large motion_scale rather than a few decimal places:
    CLIP's tokenizer has no digit merges, so extra decimal digits on small
    deltas are just noise; scaling first keeps small deltas on distinguishable
    integers instead of rounding to 0.
    """
    return (
        f"x={_fmt_no_neg_zero(x * motion_scale, decimals)} "
        f"y={_fmt_no_neg_zero(y * motion_scale, decimals)} "
        f"z={_fmt_no_neg_zero(z * motion_scale, decimals)} "
        f"roll={_fmt_no_neg_zero(roll * motion_scale, decimals)} "
        f"pitch={_fmt_no_neg_zero(pitch * motion_scale, decimals)} "
        f"yaw={_fmt_no_neg_zero(yaw * motion_scale, decimals)} "
        f"grip={_fmt_no_neg_zero(grip * grip_scale, decimals)}."
    )
