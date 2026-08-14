"""Interactive keyboard teleop: `clap-teleop --config <path> --ckpt <path> --dataset <name> --episode <id>`.

Seeds a cold-start history buffer from one recorded episode's first frame,
then repeatedly reads a keypress, turns it into a cartesian action chunk via
`teleop_controls.keyboard_control`, and previews the world model's prediction
for that action before the (real or simulated) robot executes it.
"""

import argparse
import logging
import os
import sys
import termios
import tty

import mediapy
import numpy as np
import torch

from clap.data.base import BoundNormalizer
from clap.data.rollout_loaders import ROLLOUT_LOADERS
from clap.rollout.agent import CLAPRolloutAgent
from clap.rollout.teleop_controls import (
    KEY_HELP,
    SWITCH_TARGET_KEY,
    TOGGLE_DUAL_KEY,
    keyboard_control,
    num_targets,
    target_name,
)
from clap.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TeleopSession:
    """One interactive teleop session against a single seed episode.

    Args:
        rollout_agent: A `CLAPRolloutAgent` (family="ee" for droid/bridge/taco_play alike, see
            `CLAPRolloutAgent`'s docstring) with its checkpoint already loaded.
        history_idx: Optional custom sparse index pattern into the buffer
            (see `CLAPRolloutAgent.autoregressive_replay`); default (None) uses
            the last `num_history` frames directly, same as replay.py's default.
        live_view: Optional `live_viewer.LiveViewServer`, already started -- if given,
            each step's predicted frame is pushed to it for a live browser preview
            (see `_run_interactive`'s docstring for why the final video alone isn't enough),
            and the episode's own first frame is pushed immediately as the seed/initial
            frame (its own slot on the page, not overwritten by later predictions).
        dataset_name / ckpt_name: Only used to label the live_view broadcast; purely cosmetic.
    """

    def __init__(self, rollout_agent, ep, stat_path, num_history, num_frames, history_idx=None,
                 live_view=None, num_inference_steps=50, dataset_name=None, ckpt_name=None, live_view_fps=4):
        self.agent = rollout_agent
        self.num_history = num_history
        self.num_frames = num_frames
        self.history_idx = history_idx
        self.live_view = live_view
        self.live_view_fps = live_view_fps
        self.num_inference_steps = num_inference_steps
        self.normalizer = BoundNormalizer(stat_path)

        gt_latents = rollout_agent.encode_video(ep["video"])
        # Cold-start: seed the whole history buffer with num_history copies of frame 0.
        first_latent = gt_latents[0:1].expand(num_history, -1, -1, -1).clone()
        self.latent_buffer = first_latent
        self.pose_buffer = [ep["states"][0:1]] * num_history  # each (1, D): D=7 [x,y,z,r,p,y,grip] (droid/bridge/taco_play), or 14/26 (bimanual_yam/g1_humanoid, see teleop_controls)
        self.text = ep.get("text", "")
        self.frames = []  # accumulated (GT-seed / prediction) frames for the output video
        self.step_num = 0  # first-step check for step()'s chunk-stitching
        self.active_target = 0  # multi-target poses only -- see teleop_controls.SWITCH_TARGET_KEY
        self.dual = False  # multi-target poses only -- see teleop_controls.TOGGLE_DUAL_KEY

        if self.live_view is not None:
            decoded = self.agent.decode_latents(gt_latents[0:1], decode_chunk_size=1)  # (1, 3, H, W) in [-1, 1] -- the episode's real first frame
            seed_frame = ((decoded[0] / 2 + 0.5).clamp(0, 1).float() * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()  # HWC uint8 RGB
            self.live_view.broadcast_frame(
                seed_frame, seed=True, dataset=dataset_name.upper() if dataset_name else dataset_name,
                ckpt_name=ckpt_name, instruction=self.text, **self._mode_meta(),
            )

    def _mode_meta(self):
        """Extra live-view broadcast fields describing the current active_target/dual state --
        empty for a plain single-arm (7-dim) pose, which has no target/dual concept at all."""
        D = self.pose_buffer[-1].shape[1]
        if D == 7:
            return {}
        return {"active_target": target_name(D, self.active_target), "dual": self.dual}

    def step(self, key):
        """Apply one keypress: build the action chunk, predict, and advance the buffers.

        `SWITCH_TARGET_KEY`/`TOGGLE_DUAL_KEY` (multi-target poses only) are intercepted
        here rather than passed to `keyboard_control` -- they toggle persistent session
        state (which target future keypresses address, and whether they're mirrored)
        with no pose change and no model call, so they return the unchanged current pose.
        """
        current_pose = self.pose_buffer[-1]
        D = current_pose.shape[1]

        if D != 7 and key in (SWITCH_TARGET_KEY, TOGGLE_DUAL_KEY):
            if key == SWITCH_TARGET_KEY:
                self.active_target = (self.active_target + 1) % num_targets(D)
                logger.info(f"🔀 active target -> {target_name(D, self.active_target)}")
            else:
                self.dual = not self.dual
                logger.info(f"🪞 dual mode -> {'ON' if self.dual else 'off'}")
            if self.live_view is not None:
                self.live_view.broadcast_meta(key=key, **self._mode_meta())
            return current_pose[0]  # no motion -- mode toggle only

        action_chunk = keyboard_control(
            current_pose, key, active_target=self.active_target, dual=self.dual,
        )  # (5, D), un-normalized

        # Default: the last num_history frames of the running buffer, same as replay.py. A
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
        image = self.latent_buffer[-1:]  # (1, 4, h, w) -- the slice already carries the batch dim, matching CLAPRolloutAgent.autoregressive_replay's buffer[-1:] convention
        history = history_latents.unsqueeze(0)

        # frame_level_cond conditioning is one token per frame of the FULL num_history+num_frames
        # window (matching CLAPRolloutAgent._build_chunk_condition's states_padded[s:s+T] slice) --
        # not just the num_frames future chunk, so prepend the running history poses.
        full_action_window = np.concatenate([history_poses, action_chunk], axis=0)  # (num_history + num_frames, D)
        norm_action = self.normalizer.normalize(full_action_window)
        action_cond = torch.from_numpy(norm_action.astype(np.float32)).unsqueeze(0).to(self.agent.dtype)

        pred = self.agent.predict_chunk(
            image, history, action_cond, [self.text], self.num_frames,
            num_inference_steps=self.num_inference_steps,
        )  # (num_frames, 4, h, w)

        # Step 0 contributes every predicted frame; later steps drop the first -- it duplicates
        # the conditioning image (this step's `image` = the previous step's last kept frame),
        # same convention as CLAPRolloutAgent.autoregressive_replay's chunk stitching (see its
        # "Chunk 0 contributes all n_keep frames; later chunks drop their first frame" comment).
        # kept_poses tracks kept_latents 1:1 so latent_buffer/pose_buffer keep growing in lockstep.
        kept_latents = pred if self.step_num == 0 else pred[1:]
        kept_poses = action_chunk if self.step_num == 0 else action_chunk[1:]

        self.latent_buffer = torch.cat([self.latent_buffer, kept_latents], dim=0)
        self.pose_buffer.extend(kept_poses[:, None, :])  # each appended as its own (1, D) row, matching pose_buffer's existing convention
        self.frames.extend(kept_latents)  # accumulated for save_video -- every kept frame, not just this step's last
        self.step_num += 1

        if self.live_view is not None:
            if self.live_view.has_clients():
                # Every kept frame, not just this step's last -- now that we're keeping them all for
                # save_video anyway, broadcasting only the last one would make the live view choppier
                # than the actual video it's previewing. Worth the extra VAE-decode work (redundant
                # with save_video's own later decode) only when someone's actually watching.
                to_broadcast = kept_latents
            else:
                to_broadcast = kept_latents[-1:]  # nobody watching -- cheap single-frame decode, just enough to keep the cache fresh for a later joiner
            decoded = self.agent.decode_latents(to_broadcast, decode_chunk_size=self.num_frames)  # (n, 3, H, W) in [-1, 1]
            frames = [
                ((frame_latent / 2 + 0.5).clamp(0, 1).float() * 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()  # HWC uint8 RGB
                for frame_latent in decoded
            ]
            # Paced on the live-view's own background thread (see broadcast_frames), so this
            # never delays the keypress loop itself.
            self.live_view.broadcast_frames(frames, fps=self.live_view_fps, key=key, **self._mode_meta())

        return action_chunk[-1]  # the pose to send to the (real or simulated) robot

    def save_video(self, path, decode_chunk_size=7):
        latents = torch.stack(self.frames, dim=0)  # (T, 4, h, w) stack of all predicted frames
        decoded = self.agent.decode_latents(latents, decode_chunk_size)  # (T, 3, H, W) in [-1, 1]
        video = ((decoded / 2 + 0.5).clamp(0, 1).float() * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()  # to uint8 HWC frames
        mediapy.write_video(path, video, fps=4)


def _read_key():
    """Block for exactly one raw keypress from stdin, no Enter needed (Unix only)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)  # save current terminal mode to restore afterward
    try:
        tty.setraw(fd)  # disable line-buffering/echo so a single keystroke is available immediately
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # always restore, even on an exception


def _run_interactive(session, dataset, episode):
    """Live loop: read one keypress at a time, apply it, print the resulting pose, until Ctrl-C.

    Ctrl-C is the only quit key.

    Predicted frames only land on disk (session.save_video) once this loop exits --
    if session.live_view is set, each step's frame is also pushed there immediately,
    so a connected browser tab shows the model's prediction as you type instead of
    only after quitting.
    """
    print(f"Interactive teleop on {dataset}/{episode}. Keys: {KEY_HELP}, Ctrl-C to quit.")
    try:
        while True:
            key = _read_key()
            if key == "\x03":  # Ctrl-C
                break
            pose = session.step(key)  # apply one keypress, advance buffers
            logger.info(f"📈 key={key} -> pose={pose}")
    except KeyboardInterrupt:
        pass  # Ctrl-C during a step (e.g. mid-inference) exits the loop the same as between steps


def cli():
    """Parse command-line arguments for the teleop CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--family", default="ee")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", required=True, help="Episode id/rel to seed the session from.")
    parser.add_argument("--keys", default=None,
                         help="Sequence of teleop keys to replay non-interactively, e.g. 'wwaaz'. "
                              "Omit for a live interactive session (reads keys from the terminal, Ctrl-C to quit).")
    parser.add_argument("--save-dir", default="teleop_outputs")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--history-idx", type=int, nargs="+", default=None,
                         help="Custom sparse history offsets (e.g. 0 0 -12 -9 -6 -3), same convention as "
                              "RolloutDeployConfig.history_idx / RolloutReplayConfig.history_idx. "
                              "Default (unset) uses the last num_history contiguous frames.")
    parser.add_argument("--no-live-view", action="store_true",
                         help="Disable the live-preview server (on by default) -- otherwise each "
                              "step's predicted frame streams to a browser tab serving "
                              "examples/getting_started/teleop_viewer.html.")
    parser.add_argument("--live-view-ws-port", type=int, default=8765, help="Websocket port (frames).")
    parser.add_argument("--live-view-http-port", type=int, default=8766, help="HTTP port (the viewer page itself).")
    parser.add_argument("--live-view-fps", type=float, default=4, help="Playback rate for multi-frame live-view broadcasts.")
    parser.add_argument("--ckpt-name", default=None, help="Display name for the live-view page (e.g. 'CLAP-EE'); purely cosmetic.")
    return parser.parse_args()


def main():
    from clap.config import load_config

    args = cli()  # parse CLI args
    config = load_config(args.config)  # load model/data config from YAML
    agent = CLAPRolloutAgent(config.model, args.ckpt, family=args.family, action_caption_mode=config.data.action_caption_mode)  # load the checkpoint once

    loader_cls = ROLLOUT_LOADERS[args.family]  # loader class registered for this conditioning family
    loader = loader_cls(
        dataset_name=args.dataset, oxe_base_path=config.data.oxe_base_path, video_size=config.data.video_size,
        dataset_meta_info_path=config.data.dataset_meta_info_path,
    )
    ep = next(loader.load(i) for i in range(len(loader)) if str(loader.episodes[i].get("episode_id", loader.episodes[i].get("rel"))) == args.episode)  # find the requested episode by id

    live_view = None
    if not args.no_live_view:
        from clap.rollout.live_viewer import LiveViewServer

        live_view = LiveViewServer(ws_port=args.live_view_ws_port, http_port=args.live_view_http_port)  # streams each predicted frame to a browser tab, local or port-forwarded
        live_view.start()

    session = TeleopSession(
        agent, ep, ep["stat_path"], config.model.num_history, config.model.num_frames,
        live_view=live_view, num_inference_steps=args.num_inference_steps, history_idx=args.history_idx,
        dataset_name=args.dataset, ckpt_name=args.ckpt_name, live_view_fps=args.live_view_fps,
    )  # seed a teleop session from this episode
    try:
        if args.keys is not None:
            for key in args.keys:  # scripted replay: fixed key sequence, no terminal interaction
                pose = session.step(key)  # apply one keypress, advance buffers
                logger.info(f"📈 key={key} -> pose={pose}")
        else:
            _run_interactive(session, args.dataset, args.episode)  # live loop, reads keys from the terminal
    finally:
        if live_view is not None:
            live_view.stop()

    os.makedirs(args.save_dir, exist_ok=True)
    session.save_video(os.path.join(args.save_dir, f"teleop_{args.dataset}_{args.episode}.mp4"))


if __name__ == "__main__":
    main()