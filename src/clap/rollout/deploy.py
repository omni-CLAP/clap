"""Real-robot policy-in-the-loop deployment: `clap-rollout-deploy --config <path> --deploy-config <path> --ckpt <path>`.

Each interaction round: a policy (openpi or MolmoAct2) predicts an action chunk
from the current observation; for openpi's joint-velocity output,
`EEVelocityToPositionAdapter` integrates it into future joint positions first
(MolmoAct2 predicts joint positions directly, skipping that step). What happens next depends on
the embodiment (`PolicyInTheLoopAgent.is_joint_space`, from `EmbodimentConfig.action_mode`):
for DROID/Franka (`action_mode="ee7"`), forward kinematics converts the predicted joint
trajectory into cartesian poses the world model conditions on for an "imagined" preview of what
executing the chunk should look like; for bimanual_yam (`action_mode="joint14"`), there's no FK
at all -- the raw predicted dual-arm joint+gripper state is the native representation and feeds
the world model directly. Camera handling is also embodiment-aware: DROID's policy interface
takes 2 cams (right + wrist), bimanual_yam's takes 3 (right/top, left, wrist -- see
`MolmoActPolicy._infer_remote` for the confirmed slot->wire-key mapping). Only openpi and
MolmoAct2's DROID checkpoint are Franka-specific; MolmoAct2's bimanual_yam checkpoint (server
mode only, so far -- see `examples/getting_started/deploy_yam.sh`) is the only policy currently
wired up for a second embodiment.

bimanual_yam is the ONLY joint-space embodiment actually supported here, despite
`is_joint_space` being computed generically from `action_mode.startswith("joint")`.
g1_humanoid (`action_mode="joint26"`) also satisfies that check but is NOT wired up: it uses
4-camera `four_view` stacking, not the 3-camera `three_view` this module hardcodes for
joint-space, and there's no policy server for it. Pointing `deploy_config.yaml` at
`dataset_name: g1_humanoid` will not raise -- it'll silently crop the wrong camera regions (or
crash on an unrecognized stacking mode) instead. Only `droid` and `bimanual_yam` are tested.

Requires the optional `openpi` extra (`pip install clap[openpi]`) or a
`lerobot`-with-MolmoAct2 install, matching `deploy_config.policy_type`.
"""

import argparse
import json
import logging
import os

import cv2
import mediapy
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from clap.data.base import BoundNormalizer
from clap.data.camera_stacking import view_slices_for_stacking
from clap.data.oxe_catalog import get_embodiment_config
from clap.data.rollout_loaders import ROLLOUT_LOADERS
from clap.rollout.agent import CLAPRolloutAgent
from clap.rollout.kinematics import get_fk_solution
from clap.rollout.policies.ee_velocity_to_position_adapter import EEVelocityToPositionAdapter
from clap.rollout.policies.openpi_policy import OpenPIPolicy
from clap.rollout.replay import ep_id_str, select_indices
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class PolicyInTheLoopAgent:
    """Combines a real-robot policy with `CLAPModel` for deployment with a world-model preview.

    Args:
        rollout_agent: A `clap.rollout.agent.CLAPRolloutAgent`, used only for
            its loaded `CLAPModel` (the world-model "imagination" preview).
        deploy_config: `RolloutDeployConfig`. `policy_type` selects the policy:
            "pi05"/"pi0"/"pi0fast" (openpi, joint-velocity output, needs
            `adapter_ckpt`) or "molmoact2" (MolmoAct2, joint-position output,
            no adapter needed).
    """

    def __init__(self, rollout_agent, deploy_config):
        self.rollout_agent = rollout_agent  # already-loaded CLAPModel, reused for the imagination preview
        self.config = deploy_config
        self.policy_type = deploy_config.policy_type
        self.is_molmoact = self.policy_type.startswith("molmoact")
        # Joint-space embodiments (bimanual_yam's action_mode="joint14") condition the world model
        # on the raw predicted joint state directly -- no forward kinematics, no separate cartesian
        # representation, no fixed joint/gripper split (gripper columns are embedded per-arm).
        # NOTE: bimanual_yam is the only joint-space embodiment actually wired up end to end --
        # g1_humanoid ("joint26") also matches this startswith check but is NOT supported (it needs
        # 4-camera four_view stacking, not the 3-camera three_view this module hardcodes below, and
        # has no policy server). See this module's docstring.
        self.is_joint_space = get_embodiment_config(deploy_config.dataset_name).action_mode.startswith("joint")

        if self.is_molmoact:
            from clap.rollout.policies.molmoact_policy import MolmoActPolicy
            self.policy = MolmoActPolicy(
                deploy_config.policy_ckpt or None, norm_tag=deploy_config.molmoact_norm_tag,
                include_wrist=deploy_config.molmoact_include_wrist, server=deploy_config.policy_server,
            )
            self.adapter = None  # MolmoAct2 predicts joint position directly; no velocity-integration needed
        else:
            # Loads the openpi checkpoint for the requested policy_type ("pi05"/"pi0"/"pi0fast"),
            # or connects to a running server -- see RolloutDeployConfig.policy_server.
            self.policy = OpenPIPolicy(self.policy_type, deploy_config.policy_ckpt or None, server=deploy_config.policy_server)
            # action_num=15: the adapter was trained expecting exactly 15 velocity
            # steps in, regardless of how many the policy itself actually predicts.
            self.adapter = EEVelocityToPositionAdapter(action_dim=7, action_num=15, hidden_size=512)
            adapter_state = torch.load(deploy_config.adapter_ckpt, map_location=rollout_agent.device)
            self.adapter.load_state_dict(adapter_state)
            self.adapter.to(rollout_agent.device).eval()  # frozen; never trained during deployment

    def step(self, images, joint_position, gripper_position, text, round_num=0):
        """One policy-in-the-loop interaction round.

        Args:
            images: `[right, wrist]` (DROID) or `[right, left, wrist]` (bimanual_yam) camera frames.
            joint_position: (7,) DROID joint angles, or (14,) bimanual_yam's full dual-arm
                joint+gripper state (gripper columns embedded per-arm, not a trailing column).
            gripper_position: (1,) DROID gripper position, or `None` for bimanual_yam (already
                folded into `joint_position`).
            round_num: This session's current interaction round (0-indexed), from
                `DeploySession.round_num`. Only used when `deploy_config.policy_skip_step_schedule`
                is set -- ignored otherwise (the scalar `policy_skip_step` applies to every round).

        Returns:
            {"joint_pos_chunk": (pred_step, 8 or 14), "action_chunk": (pred_step, 7 or 14),
             "joint_pos": ..., "joint_vel": ..., "action": ...} — the skip-subsampled trajectory
            to execute, plus the full (pre-subsample) trajectory for the world-model preview.
            `action`/`action_chunk` is the FK'd cartesian ee pose for DROID, or (identical to
            `joint_pos`/`joint_pos_chunk`, since no FK ever runs) the raw joint state for
            bimanual_yam -- either way, it's the quantity the world model conditions on.
        """
        raw_action_chunk = self.policy.infer(images, joint_position, gripper_position, text)
        
        if self.is_joint_space:
            # No fixed joint/gripper split: bimanual_yam's 14-dim state bundles gripper columns
            # in per-arm, and MolmoAct2 (the only policy that ever serves joint-space embodiments)
            # already predicts absolute positions directly -- no adapter-integration step either.
            current_joint = joint_position[None, :]  # (1, 14)
            joint_pos_future = raw_action_chunk
            joint_pos = np.concatenate([current_joint, joint_pos_future], axis=0)
            # Informational only (not fed into the world model): no native velocity signal, so
            # approximate one with a finite difference, same as DROID's MolmoAct2 branch below.
            joint_vel = np.diff(joint_pos, axis=0)
            # No FK -- joint_pos already IS the native representation the world model conditions on.
            action = joint_pos
        else:
            current_joint = joint_position[None, :7]
            current_gripper = gripper_position[None, :]

            if self.is_molmoact:
                # MolmoAct2's output is already absolute joint positions — no integration step.
                joint_pos_future = raw_action_chunk[:, :7]
                gripper_future = np.clip(raw_action_chunk[:, 7:], 0, self.config.gripper_max)
            else:
                # pi0/pi0fast predict a shorter horizon (10 steps) than pi05 (15 steps);
                # repeat the last predicted step to still fill the adapter's fixed 15-step window.
                idx = list(range(15)) if "pi05" in self.policy_type else list(range(10)) + [9] * 5
                joint_vel = raw_action_chunk[idx, :7]
                gripper_future = np.clip(raw_action_chunk[idx, 7:], 0, self.config.gripper_max)  # safety clamp
                # Adapter integrates the velocity window into absolute future joint positions.
                with torch.no_grad():
                    joint_pos_future = self.adapter(current_joint, joint_vel, None, training=False)  # (15, 7)

            # Prepend the current joint/gripper as step 0, 
            # then re-truncate back to 15 steps to reduce overhead from FK
            joint_pos = np.concatenate([current_joint, joint_pos_future], axis=0)[:15]
            gripper_pos = np.concatenate([current_gripper, gripper_future], axis=0)[:15]
            if self.is_molmoact:
                # Informational only (not fed into FK/the world model): MolmoAct2 has no native
                # velocity signal the way openpi's raw prediction does, so approximate one with
                # a finite difference of joint_pos instead.
                joint_vel = np.diff(joint_pos, axis=0)
            # else: joint_vel already holds openpi's actual predicted velocities (set above).

            # Forward-kinematics to transform predicted joint position into a cartesian pose
            # row, so the world model (trained on cartesian actions) can condition on it.
            ee_pose = []
            for i in range(joint_pos.shape[0]):
                T = get_fk_solution(joint_pos[i, :7])  # 4x4 world-from-end-effector transform
                xyz = T[:3, 3]  # translation
                euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")  # orientation, as roll/pitch/yaw
                ee_pose.append(np.concatenate([xyz, euler, gripper_pos[i]]))  # (7,): [x,y,z,r,p,y,grip]
            action = np.array(ee_pose)  # (15, 7)

        # Full (pre-subsample) trajectory, truncated to the portion the caller's
        # world-model preview actually spans: skip_step * pred_step steps. schedule[i] holds
        # its last value for every round past the end of the list (e.g. [4, 4, 2]: rounds 0-1
        # skip 4, round 2+ skip 2) -- lets early rounds cover more ground per world-model call
        # before settling into finer-grained subsampling.
        schedule = self.config.policy_skip_step_schedule
        skip = schedule[min(round_num, len(schedule) - 1)] if schedule else self.config.policy_skip_step
        chunk_span = skip * (self.config.pred_step)
        preview = {"joint_pos": joint_pos[:chunk_span], "joint_vel": joint_vel[:chunk_span], "action": action[:chunk_span]}
        
        # Subsample every `skip`-th step down to pred_step actual robot commands.
        action_chunk = action[::skip][:self.config.pred_step]
        joint_pos_chunk = joint_pos[::skip][:self.config.pred_step]
        if not self.is_joint_space:
            joint_pos_chunk = np.concatenate([joint_pos_chunk, action_chunk[:, -1:]], axis=-1)  # append the gripper column
        # else: joint_pos_chunk is already the full 14-dim state -- nothing to append.

        return {"joint_pos_chunk": joint_pos_chunk, "action_chunk": action_chunk, **preview}


def _extract_ee_state(ann, raw_idx, video_length):
    """Cartesian ee state from whichever field `ann` actually carries: the raw-rate
    `observation.state.cartesian_position` (indexed like joint_position, via `raw_idx`) if
    present, else the video-rate `state` field (already downsampled to video_length, the
    only one some cartesian-only annotation copies carry)."""
    if "observation.state.cartesian_position" in ann:
        raw_ee = np.array(ann["observation.state.cartesian_position"], dtype=np.float32)
        return raw_ee[raw_idx]
    return np.array(ann["state"], dtype=np.float32)[:video_length, :6]


def _load_joint_trajectory(oxe_base_path, dataset_name, episode_id, annotation_subdir="annotation", split="val", raw_stride=3):
    """Real joint/gripper/cartesian state at the video frame rate, for policy-in-the-loop seeding.

    Reads joint_position, gripper_position, AND cartesian ee state from the same annotation
    JSON, so the seed pose_buffer entry (see `DeploySession.__init__`) can use the recorded
    cartesian reading directly instead of forward-kinematics from joint_position.

    Tries `annotation_subdir` first (the dataset's configured `EmbodimentConfig.annotation_subdir`,
    which some deployments point at a cartesian-only annotation copy elsewhere via
    `CLAP_<NAME>_ANNOTATION_SUBDIR`). If that copy doesn't carry raw joint-space readings, warns
    and falls back to the raw "annotation" subdir -- and raises if even that subdir has none
    (some datasets, e.g. EE-only ones, never record joint-space readings at all).

    Args:
        raw_stride: Default 3, e.g., for droid dataset with only videos downsampled by 3x
            ($OXE_BASE_PATH/droid/annotation/val/*.json): states[i][:6] exactly matches
            observation.state.cartesian_position[min(i*3, raw_len-1)] at every checked
            index. Not a universal constant — pass the correct value for your specific dataset.

    Returns:
        (joint_position (video_length, 7), gripper_position (video_length, 1), ee_state (video_length, 6)).
    """
    def _read(subdir):
        ann_path = os.path.join(oxe_base_path, dataset_name, subdir, split, f"{episode_id}.json")
        with open(ann_path) as f:
            return json.load(f)

    ann = _read(annotation_subdir)
    if "observation.state.joint_position" not in ann:
        logger.warning(
            f"⚠️ {annotation_subdir}/{split}/{episode_id}.json has no observation.state.joint_position "
            f"(likely a cartesian-only annotation copy) -- falling back to the raw 'annotation' subdir "
            f"for joint_position/ee_state"
        )
        ann = ann if annotation_subdir == "annotation" else _read("annotation")
        if "observation.state.joint_position" not in ann:
            raise KeyError(
                f"Neither '{annotation_subdir}' nor 'annotation' has observation.state.joint_position "
                f"for {dataset_name}/{episode_id} -- this dataset may not record joint-space readings at all."
            )

    video_length = ann["video_length"]  # native (post-video-downsample) frame count, the target index space
    raw_joint = np.array(ann["observation.state.joint_position"], dtype=np.float32)
    raw_gripper = np.array(ann["observation.state.gripper_position"], dtype=np.float32)
    # Map each video-frame index to its raw-sensor index (raw_stride apart), clamped to the raw array's end.
    raw_idx = np.minimum(np.arange(video_length) * raw_stride, len(raw_joint) - 1)
    return raw_joint[raw_idx], raw_gripper[raw_idx][:, None], _extract_ee_state(ann, raw_idx, video_length)


def _overlay_text(video, text, max_chars_per_line=40):
    """Burn `text` into the top-left of every frame of `video` (T, H, W, C uint8), word-wrapped
    to `max_chars_per_line`, on a semi-transparent dark strip so it stays legible over any frame content."""
    if not text:
        return video
    words = text.split()
    lines, cur = [], ""
    for w in words:  # greedy word-wrap, since cv2 has no built-in text layout
        cur = f"{cur} {w}".strip()
        if len(cur) > max_chars_per_line:
            lines.append(cur[:-len(w)].strip())
            cur = w
    lines.append(cur)

    line_h = 18
    strip_h = min(video.shape[1], line_h * len(lines) + 12)
    video = video.copy()
    video[:, :strip_h] = (video[:, :strip_h].astype(np.float32) * 0.35).astype(np.uint8)  # darken a strip for contrast
    for i, line in enumerate(lines):
        y = 16 + i * line_h
        for frame in video:
            cv2.putText(frame, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return video


class DeploySession:
    """One policy-in-the-loop deployment session against a single seed episode.

    Simulation mode by default (self-driven closed loop, no live robot):
    `get_observation` decodes the world model's own last prediction and the
    policy's own last predicted joint state; `execute_action` is a no-op. A
    real-robot integration overrides both — reading live camera frames/joint
    encoders in `get_observation`, sending the chosen pose to the robot
    controller in `execute_action` — without touching `step` itself.

    Args:
        rollout_agent: `CLAPRolloutAgent` (world model, for the imagination preview).
        pil_agent: `PolicyInTheLoopAgent` (policy + adapter + FK).
        ep: Seed episode dict from an `EEEpisodeLoader` (video/text).
        joint_position0 / gripper_position0 / ee_pose0: Real joint/gripper/cartesian state at
            the seed frame, seeding the self-driven joint-state buffer and the initial
            pose_buffer entry. For DROID (`pil_agent.policy.is_joint_space` False), all three
            come from `_load_joint_trajectory` (straight from the annotation JSON). For
            bimanual_yam (joint-space), `joint_position0` is the full 14-dim seed state (from
            `ep["states"]`) and `gripper_position0`/`ee_pose0` are `None` -- there's no separate
            cartesian representation or joint/gripper split to seed.
        num_frames: Must equal `deploy_config.pred_step` — `PolicyInTheLoopAgent.step`'s
            `state_fk_skip` is exactly `pred_step` rows, fed straight into `predict_chunk`.
        history_idx: Optional custom sparse index pattern into the latent buffer
            (see `RolloutDeployConfig.history_idx`); default (None) uses the last
            `num_history` frames directly, same as replay.py/teleop.py's default.
        live_view: Optional `live_viewer.LiveViewServer`, already started -- if given,
            each round's predicted frame (+ round/instruction/episode_id) is pushed to
            it for a live browser preview, same idea as `clap.rollout.teleop`'s, and the
            episode's own first frame is pushed immediately as the seed/initial frame
            (its own slot on the page, not overwritten by later predictions).
        episode_id / total_rounds / dataset_name / ckpt_name: Only used to label
            live_view broadcasts; purely cosmetic.
    """

    def __init__(self, rollout_agent, pil_agent, ep, joint_position0, gripper_position0, ee_pose0, num_history, num_frames,
                 stat_path, text=None, num_inference_steps=50, history_idx=None, live_view=None, episode_id=None,
                 total_rounds=None, dataset_name=None, ckpt_name=None, live_view_fps=4):
        self.agent = rollout_agent
        self.policy = pil_agent
        self.num_history = num_history
        self.num_frames = num_frames
        self.text = text if text is not None else ep.get("text", "")
        self.num_inference_steps = num_inference_steps
        self.history_idx = history_idx
        self.live_view = live_view
        self.live_view_fps = live_view_fps
        self.episode_id = episode_id
        self.total_rounds = total_rounds
        self.round_num = 0
        self.normalizer = BoundNormalizer(stat_path)

        gt_latents = rollout_agent.encode_video(ep["video"])
        # Cold-start: seed the whole history buffer with num_history copies of frame 0.
        self.latent_buffer = gt_latents[0:1].expand(num_history, -1, -1, -1).clone()
        self.joint_position = joint_position0
        self.gripper_position = gripper_position0

        # frame_level_cond conditioning needs one action token per frame of the FULL
        # num_history+num_frames window (matching CLAPRolloutAgent._build_chunk_condition's
        # states_padded[s:s+T] slice, and clap.rollout.teleop.TeleopSession's own pose_buffer)
        # -- not just the num_frames future trajectory PolicyInTheLoopAgent.step returns.
        if pil_agent.is_joint_space:
            # bimanual_yam: joint_position0 IS the native representation already -- no FK, no
            # separate cartesian reading to seed from.
            init_pose = joint_position0.astype(np.float32)[None, :]  # (1, 14)
        else:
            # ee_pose0 is the recorded cartesian [x,y,z,r,p,y] reading for the seed joint state,
            # straight from the annotation JSON (see _load_joint_trajectory) -- no FK needed here.
            init_pose = np.concatenate([ee_pose0, gripper_position0]).astype(np.float32)[None, :]  # (1, 7)
        self.pose_buffer = [init_pose] * num_history

        self.frames = []  # accumulated prediction frames for the output video
        self.rounds = []  # per-round policy_in_out dicts, for the saved info json

        if self.live_view is not None:
            decoded = self.agent.decode_latents(gt_latents[0:1], decode_chunk_size=1)  # (1, 3, H, W) in [-1, 1] -- the episode's real first frame
            seed_frame = ((decoded[0] / 2 + 0.5).clamp(0, 1).float() * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()  # HWC uint8 RGB
            self.live_view.broadcast_frame(
                seed_frame, seed=True, dataset=dataset_name.upper() if dataset_name else dataset_name, ckpt_name=ckpt_name,
                instruction=self.text, episode_id=episode_id, total=total_rounds,
            )
            # Also into the main live-prediction slot (round=0) -- otherwise the previous episode's
            # last predicted frame just sits there while this episode's session gets built, reading
            # as a stall rather than a clean cut to the new episode; reuses the same decoded pixels.
            self.live_view.broadcast_frame(
                seed_frame, round=0, total=total_rounds, instruction=self.text, episode_id=episode_id,
            )

    def get_observation(self):
        """Default: `images` decoded from the world model's own last prediction, plus the
        tracked (self-driven) joint/gripper state. Override for a live robot connection.

        DROID (2-cam policy interface) crops just `["right", "wrist"]`; bimanual_yam (3-cam)
        needs all three three_view slots, `["right", "left", "wrist"]` -- per
        `MolmoActPolicy._infer_remote`'s confirmed camera mapping, right->top_cam,
        left->left_cam, wrist->right_cam on the wire. Only `is_joint_space` (bimanual_yam) vs. not
        is handled here -- a four_view joint-space embodiment (g1_humanoid) would need its own
        branch, not just flipping this same `["right", "left", "wrist"]` list (see
        `PolicyInTheLoopAgent.__init__`'s `is_joint_space` note).
        """
        decoded = self.agent.decode_latents(self.latent_buffer[-1:])  # (1, 3, n_views*H, W)
        pixels = ((decoded / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()
        slots = {name: (y0, y1, x0, x1) for name, y0, y1, x0, x1 in view_slices_for_stacking("three_view", *pixels.shape[:2])}
        slot_names = ["right", "left", "wrist"] if self.policy.is_joint_space else ["right", "wrist"]
        images = []
        for name in slot_names:
            y0, y1, x0, x1 = slots[name]
            images.append(pixels[y0:y1, x0:x1])
        return images, self.joint_position, self.gripper_position

    def execute_action(self, action_chunk):
        """No-op by default (pure simulation/preview). Override to send `action_chunk`
        (the chosen action trajectory, from `step`'s `policy_out["action_chunk"]` -- cartesian
        pose for DROID, raw joint state for bimanual_yam) to a real robot."""

    def step(self):
        """One policy-in-the-loop round: observe, policy-predict, world-model-preview, act."""
        # Current camera frames + tracked joint/gripper state (decoded/self-driven by default, or live if overridden).
        images, joint_position, gripper_position = self.get_observation()
        # Policy predicts a joint trajectory; FK's it to cartesian poses for DROID, or passes it
        # straight through for bimanual_yam (see PolicyInTheLoopAgent.step above).
        policy_out = self.policy.step(images, joint_position, gripper_position, self.text, round_num=self.round_num)

        # Default: the last num_history frames of the running buffer, same as replay.py/teleop.py. A
        # custom history_idx (sparse/non-uniform pattern) can override this if set -- applied
        # identically to both buffers (they grow in lockstep, always the same length), so each
        # pose token stays aligned with the latent frame it's actually conditioning alongside.
        if self.history_idx is not None:
            L = len(self.latent_buffer)
            idx = [min(i, L - 1) if i >= 0 else max(0, L + i) for i in self.history_idx]
            history_latents = self.latent_buffer[idx]
            history_poses = np.concatenate([self.pose_buffer[i] for i in idx], axis=0)  # (num_history, D)
        else:
            history_latents = self.latent_buffer[-self.num_history:]
            history_poses = np.concatenate(self.pose_buffer[-self.num_history:], axis=0)  # (num_history, D)
        history = history_latents.unsqueeze(0)  # (1, num_history, 4, h, w)
        image = self.latent_buffer[-1:]  # (1, 4, h, w) -- the slice already carries the batch dim, matching CLAPRolloutAgent.autoregressive_replay's buffer[-1:] convention
        # frame_level_cond conditioning is one token per frame of the FULL num_history+num_frames
        # window (see __init__'s pose_buffer comment) -- prepend the running history poses to
        # this round's predicted trajectory before normalizing into the world model's training
        # scale (its own p01/p99 stat.json).
        full_action_window = np.concatenate([history_poses, policy_out["action_chunk"]], axis=0)  # (num_history + num_frames, D)
        norm_action = self.normalizer.normalize(full_action_window)
        action_cond = torch.from_numpy(norm_action.astype(np.float32)).unsqueeze(0).to(self.agent.dtype)
        pred = self.agent.predict_chunk(
            image, history, action_cond, [self.text], self.num_frames,
            num_inference_steps=self.num_inference_steps,
        )  # (num_frames, 4, h, w)

        # Round 0 contributes every predicted frame; later rounds drop the first -- it duplicates
        # the conditioning image (this round's `image` = the previous round's last kept frame),
        # same convention as CLAPRolloutAgent.autoregressive_replay's chunk stitching (see its
        # "Chunk 0 contributes all n_keep frames; later chunks drop their first frame" comment).
        # kept_poses tracks kept_latents 1:1 so latent_buffer/pose_buffer keep growing in lockstep.
        kept_latents = pred if self.round_num == 0 else pred[1:]
        kept_poses = policy_out["action_chunk"] if self.round_num == 0 else policy_out["action_chunk"][1:]

        self.latent_buffer = torch.cat([self.latent_buffer, kept_latents], dim=0)  # extend history with every kept frame this round
        self.pose_buffer.extend(kept_poses[:, None, :])  # each appended as its own (1, D) row, matching pose_buffer's existing convention
        # Self-driven state advances to the trajectory's last predicted step, seeding the next
        # round's get_observation() -- joint_pos_chunk is always the policy's own observation-space
        # representation (raw joint+gripper), not action_chunk's world-model-conditioning one.
        if self.policy.is_joint_space:
            self.joint_position = policy_out["joint_pos_chunk"][-1].astype(np.float32)
            self.gripper_position = None
        else:
            self.joint_position = policy_out["joint_pos_chunk"][-1, :7].astype(np.float32)
            self.gripper_position = policy_out["joint_pos_chunk"][-1, 7:].astype(np.float32)
        self.frames.extend(kept_latents)  # accumulated for save_outputs' video -- every kept frame, not just the round's last
        self.rounds.append({k: np.asarray(v).tolist() for k, v in policy_out.items()})  # accumulated for save_outputs' info json
        self.round_num += 1

        if self.live_view is not None:
            if self.live_view.has_clients():
                # Every kept frame, not just the round's last -- now that we're keeping them all for
                # save_outputs anyway, broadcasting only the last one would make the live view
                # choppier than the actual video it's previewing. Worth the extra VAE-decode work
                # (redundant with save_outputs' own later decode) only when someone's actually watching.
                to_broadcast = kept_latents
            else:
                to_broadcast = kept_latents[-1:]  # nobody watching -- cheap single-frame decode, just enough to keep the cache fresh for a later joiner
            decoded = self.agent.decode_latents(to_broadcast, decode_chunk_size=self.num_frames)  # (n, 3, H, W) in [-1, 1]
            frames = [
                ((frame_latent / 2 + 0.5).clamp(0, 1).float() * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()  # HWC uint8 RGB
                for frame_latent in decoded
            ]
            # Paced on the live-view's own background thread (see broadcast_frames), so this
            # never delays the policy loop itself.
            self.live_view.broadcast_frames(
                frames, fps=self.live_view_fps, round=self.round_num, total=self.total_rounds,
                instruction=self.text, episode_id=self.episode_id,
            )

        self.execute_action(policy_out["action_chunk"])  # no-op unless overridden for a real robot
        return policy_out

    def save_outputs(self, video_path, info_path, decode_chunk_size=7, overlay_text=False):
        """Decode every accumulated round's last predicted frame into one video, plus per-round policy_out as JSON."""
        latents = torch.stack(self.frames, dim=0)  # (n_rounds, 4, h, w)
        decoded = self.agent.decode_latents(latents, decode_chunk_size)  # VAE decode, chunked to bound peak memory
        video = ((decoded / 2 + 0.5).clamp(0, 1).float() * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()  # [-1,1] -> uint8 (T, H, W, C)
        if overlay_text:
            video = _overlay_text(video, self.text)  # burn the instruction into every frame, for reviewing the mp4 standalone
        mediapy.write_video(video_path, video, fps=4)
        with open(info_path, "w") as f:
            json.dump({"text": self.text, "rounds": self.rounds}, f, indent=2)


def _resolve_openpi_ckpt(env_var, name):
    """`$<env_var>/<name>` if that env var is set, else "" (falls back to `OpenPIPolicy`'s
    built-in gs://openpi-assets/... default). Backs the `${clap.openpi_ckpt:ENV_VAR,name}`
    resolver, e.g. `${clap.openpi_ckpt:OPENPI_POLICY_BASE_DIR,pi05_droid}` for `policy_ckpt`
    -- evaluated when the deploy-config YAML is loaded, so it reflects whatever the env var
    is set to at that time, not when the YAML was last edited."""
    base = os.environ.get(env_var)
    return f"{base}/{name}" if base else ""


def _load_deploy_config(path):
    """Load a `RolloutDeployConfig` from a YAML file (dataclass defaults + type-checking)."""
    from omegaconf import OmegaConf

    from clap.config.rollout import RolloutDeployConfig

    OmegaConf.register_new_resolver("clap.openpi_ckpt", _resolve_openpi_ckpt, replace=True)
    schema = OmegaConf.structured(RolloutDeployConfig)  # dataclass defaults + type-checking
    merged = OmegaConf.merge(schema, OmegaConf.load(path))
    return OmegaConf.to_object(merged)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TrainingRunConfig YAML (model:/data: sections) the checkpoint was trained with.")
    parser.add_argument("--deploy-config", required=True, help="RolloutDeployConfig-shaped YAML (task_name/val_dataset_dir/dataset_name/episode_ids/...).")
    parser.add_argument("--ckpt", required=True, help="CLAPModel checkpoint path.")
    parser.add_argument("--family", default="ee")
    parser.add_argument("--save-dir", default="deploy_outputs")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--overlay-text", action="store_true",
                         help="Burn the instruction into every frame of the saved video (off by default -- "
                              "it's already printed to the terminal and saved in the info json).")
    parser.add_argument("--no-live-view", action="store_true",
                         help="Disable the live-preview server (on by default) -- otherwise each "
                              "round's predicted frame streams to a browser tab serving "
                              "examples/getting_started/deploy_viewer.html.")
    parser.add_argument("--live-view-ws-port", type=int, default=8765, help="Websocket port (frames).")
    parser.add_argument("--live-view-http-port", type=int, default=8766, help="HTTP port (the viewer page itself).")
    parser.add_argument("--live-view-fps", type=float, default=4, help="Playback rate for multi-frame live-view broadcasts.")
    parser.add_argument("--ckpt-name", default=None, help="Display name for the live-view page (e.g. 'CLAP-EE'); purely cosmetic.")
    parser.add_argument("--policy-type", default=None, choices=["pi05", "pi0", "pi0fast", "molmoact2"],
                         help="Override deploy_config.policy_type, regardless of what the YAML sets.")
    policy_mode = parser.add_mutually_exclusive_group()
    policy_mode.add_argument("--policy-server", default=None,
                              help="Override deploy_config.policy_server to 'host:port', forcing server mode "
                                   "against that server regardless of what the YAML sets.")
    policy_mode.add_argument("--in-process", action="store_true",
                              help="Force in-process mode (clears deploy_config.policy_server), using "
                                   "deploy_config.policy_ckpt -- or OpenPIPolicy's/MolmoActPolicy's built-in "
                                   "default if that's also unset -- regardless of what the YAML sets.")
    return parser.parse_args()


def _resolve_episode_selection(loader, deploy_config):
    """Fill in episode_ids/start_idx/instructions from deploy_config, auto-discovering
    whatever's None (see `RolloutDeployConfig`'s docstring for what each None means)."""
    episode_ids = deploy_config.episode_ids
    if episode_ids is None:
        # Every episode the loader actually found under val_dataset_dir/dataset_name.
        episode_ids = [str(ep.get("episode_id") or ep.get("rel")) for ep in loader.episodes]

    start_idx = deploy_config.start_idx or [0] * len(episode_ids)  # every episode starts at frame 0

    if deploy_config.instructions is not None:
        instructions = deploy_config.instructions
    else:
        # Each episode's own recorded instruction, straight from its annotation JSON
        # (load_text is JSON-only, no video decode -- cheap to call once per episode here).
        instructions = []
        for episode_id in episode_ids:
            indices = select_indices(loader, episode_ids=[episode_id])
            instructions.append(loader.load_text(indices[0]) if indices else "")

    return episode_ids, start_idx, instructions


def main():
    from clap.config import load_config

    args = cli()
    config = load_config(args.config)  # model:/data: sections the checkpoint was trained with
    deploy_config = _load_deploy_config(args.deploy_config)
    if args.policy_type is not None:
        deploy_config.policy_type = args.policy_type  # override, e.g. switch to molmoact2, regardless of the YAML
    if args.policy_server is not None:
        deploy_config.policy_server = args.policy_server  # force server mode against this host, overriding the YAML
    elif args.in_process:
        deploy_config.policy_server = None  # force in-process mode, overriding whatever server the YAML set

    agent = CLAPRolloutAgent(config.model, args.ckpt, family=args.family, action_caption_mode=config.data.action_caption_mode)  # world model
    pil_agent = PolicyInTheLoopAgent(agent, deploy_config)  # wraps agent with the real-robot policy + adapter + FK

    loader = ROLLOUT_LOADERS[args.family](  # per-episode video/state loader for the seed episodes below
        dataset_name=deploy_config.dataset_name, oxe_base_path=deploy_config.val_dataset_dir,
        video_size=config.data.video_size, dataset_meta_info_path=config.data.dataset_meta_info_path,
    )
    episode_ids, start_indices, instructions = _resolve_episode_selection(loader, deploy_config)

    live_view = None
    if not args.no_live_view:
        from clap.rollout.live_viewer import LiveViewServer

        live_view = LiveViewServer(  # streams each round's predicted frame to a browser tab, local or port-forwarded
            ws_port=args.live_view_ws_port, http_port=args.live_view_http_port, viewer_page="deploy_viewer.html",
        )
        live_view.start()

    os.makedirs(args.save_dir, exist_ok=True)
    try:
        for episode_id, start_idx, instruction in zip(episode_ids, start_indices, instructions):
            indices = select_indices(loader, episode_ids=[episode_id])
            if not indices:
                logger.warning(f"⚠️ episode {episode_id} not found, skipping")
                continue
            ep = loader.load(indices[0])

            if pil_agent.is_joint_space:
                # bimanual_yam: ep["states"] (already loaded by loader.load() above) IS the
                # native per-frame joint+gripper representation -- no raw-rate annotation schema
                # to read, no FK-seed needed, so _load_joint_trajectory doesn't apply at all.
                seed_idx = min(start_idx, len(ep["states"]) - 1)
                joint_position0, gripper_position0, ee_pose0 = ep["states"][seed_idx], None, None
            else:
                # Real joint/gripper/cartesian trajectory to seed the self-driven state (see
                # _load_joint_trajectory -- reads from loader.config.annotation_subdir if it carries
                # joint-space readings, else falls back to the raw "annotation" subdir).
                joint_traj, gripper_traj, ee_traj = _load_joint_trajectory(
                    deploy_config.val_dataset_dir, deploy_config.dataset_name, episode_id,
                    annotation_subdir=loader.config.annotation_subdir,
                )
                seed_idx = min(start_idx, len(joint_traj) - 1)  # clamp in case start_idx runs past this episode's length
                joint_position0, gripper_position0, ee_pose0 = joint_traj[seed_idx], gripper_traj[seed_idx], ee_traj[seed_idx]

            session = DeploySession(  # one policy-in-the-loop session for this episode
                agent, pil_agent, ep, joint_position0, gripper_position0, ee_pose0,
                config.model.num_history, config.model.num_frames, ep["stat_path"], text=instruction,
                num_inference_steps=args.num_inference_steps, history_idx=deploy_config.history_idx,
                live_view=live_view, episode_id=episode_id, total_rounds=deploy_config.interact_num,
                dataset_name=deploy_config.dataset_name, ckpt_name=args.ckpt_name, live_view_fps=args.live_view_fps,
            )
            logger.info(f"📝 [{episode_id}] instruction: {session.text!r}")
            for i in range(deploy_config.interact_num):
                session.step()  # one observe/predict/act round
                logger.info(f"🎬 [{episode_id}] round {i + 1}/{deploy_config.interact_num}")

            eid = ep_id_str(ep)
            session.save_outputs(
                os.path.join(args.save_dir, f"{deploy_config.task_name}_{eid}.mp4"),
                os.path.join(args.save_dir, f"{deploy_config.task_name}_{eid}.json"),
                overlay_text=args.overlay_text,
            )
    finally:
        if live_view is not None:
            live_view.stop()


if __name__ == "__main__":
    main()
