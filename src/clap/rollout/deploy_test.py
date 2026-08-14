"""Reference-matching test harness: `clap-rollout-deploy-test --config <path> --deploy-config <path> --ckpt <path>`.

Mirrors Ctrl-World's own `scripts/rollout_interact_pi.py` rollout loop as
literally as possible -- variable-for-variable where practical -- instead of
`clap.rollout.deploy`'s `DeploySession` abstraction, so any behavior
difference between the two can be isolated to a real logic divergence rather
than a refactor artifact. Uses clap's real model/policy classes
(`CLAPRolloutAgent`, `OpenPIPolicy`, `EEVelocityToPositionAdapter`,
`get_fk_solution`) for the actual computation.

Camera identity is resolved via raw cam-id (--cam-ids, default 0,1,2 = 0.mp4/
1.mp4/2.mp4 -> top/middle/bottom stacking slots), NOT clap's EmbodimentConfig
right_view_id/left_view_id/wrist_view_id relabeling -- image1/image2 are
literally stack slot 1/2 (cam_ids[1]/cam_ids[2]), matching the reference's
own `videos[1]`/`videos[2]` indexing exactly and sidestepping the semantic
"right"/"left" ambiguity entirely (see load_video_by_cam_id). Pass e.g.
`--cam-ids 1 0 2` to try a different camera assignment for a given checkpoint.

Matched to the reference (where clap.rollout.deploy currently diverges):
  - his_cond/his_joint/his_eef seeded with num_history*4 copies of frame 0
    (not just num_history: rollout_interact_pi.py's
    `for i in range(Agent.args.num_history*4)`).
  - Each of those three buffers grows by exactly ONE entry per round -- the
    round's LAST predicted frame/pose only (`...[pred_step-1]`) -- never more,
    with no round-0 special case.
  - The saved/broadcast video instead accumulates `pred[:-1]` (every frame
    EXCEPT the last) each round, unconditionally -- the reference's
    `video_to_save.append(videos_cat[:pred_step-1])`. The round's last frame
    isn't saved now because it gets reconstructed again as part of next
    round's output (it's next round's conditioning image).
  - valid_num = skip * (pred_step - 1) for the informational preview fields
    (the reference's `valid_num = int(skip*(self.args.pred_step-1))`; compare
    clap.rollout.deploy.PolicyInTheLoopAgent's current `skip * pred_step`).
  - joint_vel is never recomputed via np.diff for the openpi path -- the
    reference's `joint_vel = joint_vel` is a no-op reassignment of the
    policy/adapter's own velocity, not a finite difference of joint_pos.
"""

import argparse
import json
import logging
import os

import mediapy
import numpy as np
import torch
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

from clap.data.camera_stacking import STACKERS, frames_to_video_tensor, view_slices_for_stacking
from clap.data.rollout_loaders import ROLLOUT_LOADERS
from clap.rollout.agent import CLAPRolloutAgent
from clap.rollout.deploy import _load_joint_trajectory, _load_deploy_config
from clap.rollout.kinematics import get_fk_solution
from clap.rollout.policies.ee_velocity_to_position_adapter import EEVelocityToPositionAdapter
from clap.rollout.policies.openpi_policy import OpenPIPolicy
from clap.rollout.replay import ep_id_str, select_indices
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_video_by_cam_id(loader, episode_id, cam_ids=(0, 1, 2)):
    """Stack an episode's video using RAW cam-id order (0.mp4, 1.mp4, 2.mp4 -> top/middle/
    bottom, literally), bypassing EmbodimentConfig's right_view_id/left_view_id/wrist_view_id
    relabeling that `loader.load()` normally applies -- so testing a checkpoint doesn't depend
    on getting clap's semantic camera labels right, only on which raw cam-id ended up where.

    Mirrors `EEEpisodeLoader.load()`'s own frame-selection exactly (same native_ids
    downsampling), just with `cam_ids` in place of `loader._slot_cams()`.
    """
    ep_dir = loader._resolve_ep_dir(loader.video_root, episode_id)
    with open(os.path.join(loader.ann_dir, f"{episode_id}.json")) as f:
        ann = json.load(f)
    n_state = len(ann.get("state", []))

    readers = [VideoReader(os.path.join(ep_dir, f"{cam}.mp4"), ctx=cpu(0)) for cam in cam_ids]
    n = min(min(len(vr) for vr in readers), n_state)
    native_ids = np.arange(0, n, loader.fps_downsample_ratio)

    views = [reader.get_batch(native_ids).asnumpy() for reader in readers]  # cam_ids order, no BGR/relabel handling
    frames = STACKERS["three_view"](*views).astype(np.uint8)  # positional stack: cam_ids[0]->top, [1]->middle, [2]->bottom
    return frames_to_video_tensor(frames, loader.video_size)  # (C, T, H, W) uint8


def forward_policy(policy, adapter, policy_type, gripper_max, image1, image2, current_pose, current_joint, text):
    """Ctrl-World's `agent.forward_policy`, ported variable-for-variable.

    Args:
        current_pose: (8,) current [x,y,z,r,p,y,grip,?] -- only used for logging in the
            reference; kept for signature parity, not otherwise read here.
        current_joint: (8,) current [7 joint angles, gripper position].

    Returns:
        (joint_pos, joint_vel, state_fk), each the full 15-step trajectory --
        caller applies skip/pred_step subsampling.
    """
    action_chunk = policy.infer(image1, image2, current_joint[:7], current_joint[7:], text)
    current_joint_row = current_joint[None, :7]
    current_gripper_row = current_joint[None, 7:]

    idx = list(range(15)) if "pi05" in policy_type else list(range(10)) + [9] * 5
    joint_vel = action_chunk[:, :7]
    gripper_pos = action_chunk[:, 7:]
    joint_vel = joint_vel[idx]  # (15, 7)
    gripper_pos = gripper_pos[idx]  # (15, 1)
    gripper_pos = np.clip(gripper_pos, 0, gripper_max)

    joint_pos = adapter(current_joint_row, joint_vel, None, training=False)  # (15, 7) -- absolute future joint positions
    joint_pos = np.concatenate([current_joint_row, joint_pos], axis=0)[:15]  # (15, 7)
    gripper_pos = np.concatenate([current_gripper_row, gripper_pos], axis=0)[:15]  # (15, 1)
    # No finite-difference recompute here -- joint_vel stays exactly what the
    # policy/adapter produced (matches the reference's `joint_vel = joint_vel`).

    state_fk = []
    for i in range(joint_pos.shape[0]):
        T = get_fk_solution(joint_pos[i, :7])
        xyz = T[:3, 3]
        euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
        state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
    state_fk = np.array(state_fk)  # (15, 7)

    return joint_pos, joint_vel, state_fk


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TrainingRunConfig YAML (model:/data: sections) the checkpoint was trained with.")
    parser.add_argument("--deploy-config", required=True, help="RolloutDeployConfig-shaped YAML (same shape clap.rollout.deploy uses).")
    parser.add_argument("--ckpt", required=True, help="CLAPModel checkpoint path.")
    parser.add_argument("--family", default="ee")
    parser.add_argument("--save-dir", default="deploy_test_outputs")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cam-ids", type=int, nargs=3, default=[0, 1, 2],
                         help="Which raw camera files (top, middle, bottom stacking slots) to use, e.g. "
                              "'--cam-ids 1 0 2' to swap the first two -- see load_video_by_cam_id.")
    return parser.parse_args()


def main():
    from clap.config import load_config

    args = cli()
    config = load_config(args.config)
    deploy_config = _load_deploy_config(args.deploy_config)

    agent = CLAPRolloutAgent(config.model, args.ckpt, family=args.family, action_caption_mode=config.data.action_caption_mode)  # world model
    device = agent.device

    is_molmoact = deploy_config.policy_type.startswith("molmoact")
    assert not is_molmoact, "deploy_test.py mirrors rollout_interact_pi.py, which is openpi-only -- no MolmoAct2 branch to match"
    policy = OpenPIPolicy(deploy_config.policy_type, deploy_config.policy_ckpt or None, server=deploy_config.policy_server)  # real policy
    adapter = EEVelocityToPositionAdapter(action_dim=7, action_num=15, hidden_size=512)
    adapter.load_state_dict(torch.load(deploy_config.adapter_ckpt, map_location=device))
    adapter.to(device).eval()

    loader = ROLLOUT_LOADERS[args.family](
        dataset_name=deploy_config.dataset_name, oxe_base_path=deploy_config.val_dataset_dir,
        video_size=config.data.video_size, dataset_meta_info_path=config.data.dataset_meta_info_path,
    )
    episode_ids = deploy_config.episode_ids or [str(ep.get("episode_id") or ep.get("rel")) for ep in loader.episodes]
    start_indices = deploy_config.start_idx or [0] * len(episode_ids)
    instructions = deploy_config.instructions or [
        loader.load_text(select_indices(loader, episode_ids=[eid])[0]) for eid in episode_ids
    ]

    num_history = config.model.num_history
    pred_step = deploy_config.pred_step
    skip = deploy_config.policy_skip_step
    history_idx = deploy_config.history_idx
    assert history_idx, "deploy_test.py needs deploy_config.history_idx set -- the reference always indexes his_cond/his_eef via it"

    os.makedirs(args.save_dir, exist_ok=True)
    normalizer_cache = {}

    for episode_id, start_idx, text in zip(episode_ids, start_indices, instructions):
        indices = select_indices(loader, episode_ids=[episode_id])
        if not indices:
            logger.warning(f"⚠️ episode {episode_id} not found, skipping")
            continue
        ep = loader.load(indices[0])

        joint_traj, gripper_traj = _load_joint_trajectory(deploy_config.val_dataset_dir, deploy_config.dataset_name, episode_id)
        seed_idx = min(start_idx, len(joint_traj) - 1)
        joint_position0, gripper_position0 = joint_traj[seed_idx], gripper_traj[seed_idx]

        if ep["stat_path"] not in normalizer_cache:
            from clap.data.base import BoundNormalizer
            normalizer_cache[ep["stat_path"]] = BoundNormalizer(ep["stat_path"])
        normalizer = normalizer_cache[ep["stat_path"]]

        # Raw cam-id stacking (--cam-ids, default 0,1,2), not clap's right/left/wrist relabeling --
        # see load_video_by_cam_id's docstring. encode_video already VAE-encodes each view
        # separately before restacking in latent space (CLAPModel.encode_video_to_latent),
        # matching the reference's own per-camera-encode-then-cat approach.
        video_by_cam_id = load_video_by_cam_id(loader, episode_id, cam_ids=args.cam_ids)
        gt_latents = agent.encode_video(video_by_cam_id)  # (T, 4, h, w) -- only frame 0 (the seed) is actually used below
        first_latent = gt_latents[0:1]  # (1, 4, h, w)

        T = get_fk_solution(joint_position0[:7])
        xyz = T[:3, 3]
        euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
        eef0 = np.concatenate([xyz, euler, gripper_position0]).astype(np.float32)[None, :]  # (1, 7)
        joint0 = np.concatenate([joint_position0[:7], gripper_position0]).astype(np.float32)[None, :]  # (1, 8)

        # Seed with num_history*4 copies of frame 0 (reference's own choice -- see module docstring).
        his_cond = [first_latent] * (num_history * 4)
        his_joint = [joint0] * (num_history * 4)
        his_eef = [eef0] * (num_history * 4)
        video_to_save = []
        info_to_save = []

        logger.info(f"📝 [{episode_id}] instruction: {text!r}")

        for i in range(deploy_config.interact_num):
            # ---- policy forward ----
            current_joint = his_joint[-1][0]  # (8,)
            current_pose = his_eef[-1][0]  # (7,)

            decoded = agent.decode_latents(his_cond[-1])  # (1, 3, n_views*H, W) in [-1, 1]
            pixels = ((decoded[0] / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            # Raw STACK POSITION, not a "right"/"wrist" name lookup -- slot 1 = middle = cam_ids[1],
            # slot 2 = bottom = cam_ids[2], matching the reference's `videos[1]`/`videos[2]` exactly
            # now that the stack itself was built in --cam-ids order by load_video_by_cam_id.
            slots = view_slices_for_stacking("three_view", *pixels.shape[:2])  # [(name, y0,y1,x0,x1), ...], top->bottom
            _, y0, y1, x0, x1 = slots[0]
            image1 = pixels[y0:y1, x0:x1]
            _, y0, y1, x0, x1 = slots[2]
            image2 = pixels[y0:y1, x0:x1]

            joint_pos, joint_vel, state_fk = forward_policy(
                policy, adapter, deploy_config.policy_type, deploy_config.gripper_max,
                image1, image2, current_pose, current_joint, text,
            )
            valid_num = skip * (pred_step - 1)  # matches the reference exactly (clap.rollout.deploy currently omits the -1)
            policy_in_out = {
                "joint_pos": joint_pos[:valid_num], "joint_vel": joint_vel[:valid_num], "state_fk": state_fk[:valid_num],
            }
            state_fk_skip = state_fk[::skip][:pred_step]  # (pred_step, 7)
            joint_pos_skip = joint_pos[::skip][:pred_step]
            joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)  # (pred_step, 8)
            cartesian_pose = state_fk_skip

            logger.info(f"🎬 [{episode_id}] round {i + 1}/{deploy_config.interact_num} cartesian[0]={cartesian_pose[0]} cartesian[-1]={cartesian_pose[-1]}")

            # ---- world model forward ----
            action_cond = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            action_cond = np.concatenate([action_cond, cartesian_pose], axis=0)  # (num_history + pred_step, 7)
            norm_action = normalizer.normalize(action_cond)
            action_cond_t = torch.from_numpy(norm_action.astype(np.float32)).unsqueeze(0).to(agent.dtype)
            his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)  # (1, num_history, 4, h, w)
            current_latent = his_cond[-1]  # (1, 4, h, w)

            pred = agent.predict_chunk(
                current_latent, his_latent, action_cond_t, [text], pred_step,
                num_inference_steps=args.num_inference_steps,
            )  # (pred_step, 4, h, w)

            # ---- record: buffers grow by exactly ONE entry (the round's LAST frame/pose) ----
            his_joint.append(joint_pos_skip[pred_step - 1][None, :])
            his_eef.append(cartesian_pose[pred_step - 1][None, :])
            his_cond.append(pred[pred_step - 1:pred_step])  # (1, 4, h, w)
            # Saved video instead drops the LAST frame (it reappears as next round's reconstruction).
            decoded_pred = agent.decode_latents(pred[:pred_step - 1], decode_chunk_size=pred_step)  # (pred_step-1, 3, H, W) in [-1, 1]
            frames = ((decoded_pred / 2 + 0.5).clamp(0, 1).float() * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            video_to_save.append(frames)
            info_to_save.append(policy_in_out)

        video = np.concatenate(video_to_save, axis=0)
        eid = ep_id_str(ep)
        video_path = os.path.join(args.save_dir, f"{deploy_config.task_name}_{eid}.mp4")
        info_path = os.path.join(args.save_dir, f"{deploy_config.task_name}_{eid}.json")
        mediapy.write_video(video_path, video, fps=4)
        info = {"text": text, "rounds": [{k: np.asarray(v).tolist() for k, v in r.items()} for r in info_to_save]}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"✅ [{episode_id}] wrote {video_path}")


if __name__ == "__main__":
    main()
