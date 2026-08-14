"""Keyboard-to-pose-delta mapping for interactive teleop.

Three key schemes, selected by `pose`'s width:

- Single-arm EE-cartesian tasks (droid/bridge/taco_play, `pose` is (1, 7)):
  the original w/a/s/d/z/x/c/v scheme, plus e/r,o/p,t/y for [roll,pitch,yaw]
  -- `pose`'s [x, y, z, roll, pitch, yaw, grip] convention. No target/dual
  concept at all -- there's only ever the one arm.
- Multi-target poses (bimanual_yam (1, 14), g1_humanoid (1, 26)): a shared
  qwerty(+)/asdfgh(-)/u+j(+/-7th) scheme, one key-pair per raw joint dim of
  whichever target is active. `active_target` (cycled via `SWITCH_TARGET_KEY`)
  picks which named block (see `_TARGETS_BY_WIDTH`) the scheme addresses;
  `dual` (toggled via `TOGGLE_DUAL_KEY`) mirrors every keypress onto that
  target's dual partner too (its left/right counterpart -- see
  `_dual_partner`), e.g. both arms or both hands moving together.
  bimanual_yam's 2 targets are [joint0..5, gripper] each (see that dataset's
  own annotation `state_keys`/`state_format="joint_state"`); g1_humanoid's 4
  are 7-dim arms + 6-dim hands with no gripper dim at all (see its
  `observation.state` feature names) -- these are genuine joint/finger
  angles, not a cartesian pose, so "roll/pitch/yaw" wouldn't mean anything
  here.

Either way there's no IK: every key just nudges one raw pose dimension by a
fixed step. Each keypress is linearly interpolated into a 5-row action chunk
(matching the world model's per-chunk prediction horizon).
"""

import logging

import numpy as np

from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Workspace safety bounds (meters), in the robot's base frame. Only meaningful
# for the single 7-dim cartesian pose (droid/bridge/taco_play) -- skipped
# entirely for a multi-target pose, whose dims are raw joint angles with no
# established safety box here (see module docstring).
# _X_RANGE = (0.3, 0.8)
# _Y_RANGE = (-0.5, 0.5)
# _Z_RANGE = (0.01, 0.5)
_X_RANGE = (0.1, 1.0)
_Y_RANGE = (-0.5, 0.5)
_Z_RANGE = (0.01, 0.5)

_GRIPPER_KEYS = {"c": 0.99, "v": 0.0}  # close / open target values (droid/bridge/taco_play convention: 0=open, ~1=closed)
_BIMANUAL_GRIPPER_KEYS = {"c": 0.0, "v": 0.99}  # bimanual_yam's continuous_gripper_state is the OPPOSITE convention (~1=open, 0=closed) -- same c=close/v=open intent, flipped target values

# Single-arm scheme (droid/bridge/taco_play, 7-dim pose only)
# w/a/s/d/z/x, plus e/r,o/p,t/y for roll/pitch/yaw.
_SINGLE_ARM_KEY_MAP = {
    "w": (0, +1), "s": (0, -1),  # x
    "a": (1, -1), "d": (1, +1),  # y
    "z": (2, +1), "x": (2, -1),  # z
    "e": (3, -1), "r": (3, +1),  # roll
    "o": (4, -1), "p": (4, +1),  # pitch
    "t": (5, -1), "y": (5, +1),  # yaw
}

# Multi-target scheme (bimanual_yam / g1_humanoid) -- full control of whichever
# target `active_target`/`dual` select: qwerty = +, asdfgh = - for that same
# dim, one key-pair per dim, extended with u/j for a 7th dim (g1_humanoid arm
# targets only -- 6-dim targets, bimanual_yam arms and g1_humanoid hands
# alike, simply never reach it).
_MULTI_TARGET_KEY_MAP = {
    "q": (0, +1), "a": (0, -1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "d": (2, -1),
    "r": (3, +1), "f": (3, -1),
    "t": (4, +1), "g": (4, -1),
    "y": (5, +1), "h": (5, -1),
    "u": (6, +1), "j": (6, -1),
}

# pose width -> [(target name, offset, num joint dims, has_gripper, default step size), ...].
# Consecutive pairs are dual-mirror partners (0<->1, 2<->3, ...) -- see
# `_dual_partner`. Order matches each dataset's own state_keys (left before
# right, arms before hands).
#
# Default step sizes started from each dataset's actual per-dim p01/p99 spread
# (dataset_meta_info/<name>/stat.json): bimanual_yam's joints span ~1.6-2.6 rad, so
# 0.05 rad/keypress is a reasonable ~2-3% nudge. g1_humanoid's arm joints only span
# ~1.0-1.8 rad and its hand dims ~0.7-1.0, so the same 0.05 reads as proportionally
# smaller motion there -- bumped per empirical feedback (0.05 barely moved the arms;
# 0.15 still barely moved the hands even though it's a *larger* fraction of the
# hands' narrower range, so raw stat-range doesn't fully predict how much a given
# step visibly moves the model's predicted frame -- these are tuned by feel, not
# purely derived, and may need further adjustment).
_TARGETS_BY_WIDTH = {
    # bimanual_yam: [left_joint_0..5, left_gripper, right_joint_0..5, right_gripper].
    14: [("left arm", 0, 6, True, 0.05), ("right arm", 7, 6, True, 0.05)],
    # g1_humanoid: 2x 7-dim arms (kShoulderPitch/Roll/Yaw, kElbow,
    # kWristRoll/Pitch/Yaw -- no gripper dim) + 2x 6-dim hands (kHandThumb,
    # kHandThumbAux, kHandIndex/Middle/Ring/Pinky -- no gripper dim), per its
    # `observation.state` feature names.
    26: [
        ("left arm", 0, 7, False, 0.15), ("right arm", 7, 7, False, 0.15),
        ("left hand", 14, 6, False, 0.3), ("right hand", 20, 6, False, 0.3),
    ],
}

SWITCH_TARGET_KEY = "\t"  # multi-target poses only: cycle the active target (Tab)
TOGGLE_DUAL_KEY = " "  # multi-target poses only: toggle dual mode for the active target's category (Space)

_SINGLE_ARM_DISTANCE = 0.05  # droid/bridge/taco_play (7-dim cartesian pose) default step, meters

KEY_HELP = (
    "Single-arm (droid/bridge/taco_play): w/s,a/d,z/x (+/- x,y,z), e/r,o/p,t/y (+/- roll,pitch,yaw), "
    "c/v (close/open gripper). "
    "Bimanual (bimanual_yam, 2 targets: left/right arm): q/a,w/s,e/d,r/f,t/g,y/h (+/- that arm's 6 "
    "joint dims), c/v (close/open that arm's gripper). "
    "Humanoid (g1_humanoid, 4 targets: left/right arm, left/right hand): same q/a..y/h (+/- 6 dims), "
    "plus u/j (+/- 7th dim, arms only -- unused for hands, no gripper key either). "
    "Tab = cycle active target; Space = toggle dual mode for the active target's category (mirrors "
    "every keypress onto both arms, or both hands)."
)


def num_targets(pose_width):
    """How many `SWITCH_TARGET_KEY` cycles through for a `pose_width`-dim pose --
    1 for the plain 7-dim single-arm pose (no targets at all), else
    `len(_TARGETS_BY_WIDTH[pose_width])`."""
    targets = _TARGETS_BY_WIDTH.get(pose_width)
    return len(targets) if targets else 1


def target_name(pose_width, target_idx):
    """Display name of target `target_idx` for a `pose_width`-dim pose (e.g. "left arm"),
    or "single arm" for the plain 7-dim pose (which has no named targets)."""
    targets = _TARGETS_BY_WIDTH.get(pose_width)
    return targets[target_idx][0] if targets else "single arm"


def _dual_partner(target_idx):
    """Index of `target_idx`'s dual-mirror partner -- pairs consecutive targets
    (0<->1, 2<->3, ...), e.g. left/right arm, or left/right hand."""
    return target_idx + 1 if target_idx % 2 == 0 else target_idx - 1


def keyboard_control(pose, action, distance=None, active_target=0, dual=False):
    """One keypress -> a (5, D) interpolated pose action chunk, D = pose.shape[1].

    Args:
        pose: (1, 7) single-arm pose, or a (1, 14)/(1, 26) multi-target pose
            (bimanual_yam / g1_humanoid respectively) -- see module docstring
            and `_TARGETS_BY_WIDTH` for what each width's targets/dims mean.
        action: a keypress (see KEY_HELP). `SWITCH_TARGET_KEY`/`TOGGLE_DUAL_KEY`
            are NOT handled here -- they're stateful toggles the caller (e.g.
            `TeleopSession.step`) must intercept before calling this.
        distance: step size (meters for a cartesian dim, raw units for a
            joint-space dim); unused for gripper keys. Default (None) resolves
            to `_SINGLE_ARM_DISTANCE` for a 7-dim pose, else each target's own
            default step from `_TARGETS_BY_WIDTH` (overriding this uniformly
            for every target instead is rarely what you want, since arm vs.
            hand targets can have very different natural step sizes).
        active_target: index into `_TARGETS_BY_WIDTH[pose.shape[1]]` (see
            `num_targets`/`target_name`) this keypress targets when `dual` is
            False. Ignored for a 7-dim pose.
        dual: multi-target pose only -- if True, every keypress is also
            mirrored onto `active_target`'s dual partner (see `_dual_partner`),
            e.g. both arms or both hands move together.
    """
    D = pose.shape[1]
    delta = np.zeros_like(pose)
    handled = False

    if D == 7:
        step = _SINGLE_ARM_DISTANCE if distance is None else distance
        if action in _GRIPPER_KEYS:
            delta[0, 6] = -pose[0, 6] + _GRIPPER_KEYS[action]
            handled = True
        elif action in _SINGLE_ARM_KEY_MAP:
            dim, sign = _SINGLE_ARM_KEY_MAP[action]
            delta[0, dim] = sign * step
            handled = True
    else:
        targets = _TARGETS_BY_WIDTH[D]
        target_idxs = [active_target, _dual_partner(active_target)] if dual else [active_target]
        for idx in target_idxs:
            _, offset, num_dims, has_gripper, default_distance = targets[idx]
            step = default_distance if distance is None else distance
            if has_gripper and action in _BIMANUAL_GRIPPER_KEYS:
                delta[0, offset + num_dims] = -pose[0, offset + num_dims] + _BIMANUAL_GRIPPER_KEYS[action]
                handled = True
            elif action in _MULTI_TARGET_KEY_MAP:
                dim, sign = _MULTI_TARGET_KEY_MAP[action]
                if dim < num_dims:  # e.g. u/j (7th dim) pressed while a 6-dim hand is active -- no-op
                    delta[0, offset + dim] = sign * step
                    handled = True

    if not handled:
        logger.warning(f"⚠️ wrong action key, please use {KEY_HELP}")

    # Linearly interpolate from the current pose to pose+delta over 5 steps
    # (0, 1/4, 2/4, 3/4, 4/4), matching the model's per-chunk prediction length.
    action_chunk = np.concatenate([pose + delta * (i / 4.0) for i in range(5)], axis=0)  # (5, D)

    if D == 7:
        # DROID-frame safety box; only meaningful for the single cartesian
        # pose (droid/bridge/taco_play) -- see module docstring.
        action_chunk[:, 0] = np.clip(action_chunk[:, 0], *_X_RANGE)
        action_chunk[:, 1] = np.clip(action_chunk[:, 1], *_Y_RANGE)
        action_chunk[:, 2] = np.clip(action_chunk[:, 2], *_Z_RANGE)
    return action_chunk
