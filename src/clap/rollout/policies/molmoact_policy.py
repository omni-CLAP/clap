"""Wraps a MolmoAct2 policy behind the same `infer(...)` interface as `OpenPIPolicy`.

Isolates the optional `lerobot`-with-MolmoAct2 dependency to this one module —
importing `clap.rollout` doesn't require it; only constructing a `MolmoActPolicy`
with `server=None` does. `server` mode instead talks to one of MolmoAct2's own
official FastAPI inference servers (allenai/molmoact2's `examples/droid/host_server_droid.py`,
2 cams/8-dim state, or `examples/yam/host_server_yam.py`, 3 cams/14-dim state) over HTTP,
needing only the lightweight `json_numpy` + `requests` on this side.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


class MolmoActPolicy:
    """A trained MolmoAct2 policy, predicting a joint-position + gripper action chunk.

    Unlike `OpenPIPolicy` (joint velocity), MolmoAct2 predicts absolute joint positions
    directly. For DROID (`action_mode="ee7"`), the caller (`PolicyInTheLoopAgent.step`) feeds
    that output through forward kinematics before conditioning the world model; for
    bimanual_yam (`action_mode="joint14"`) there's no FK at all -- the caller passes the raw
    prediction straight through, since joint-space embodiments have no separate cartesian
    representation.

    Args:
        checkpoint_path: Path to the MolmoAct2 checkpoint. Only meaningful
            in-process (`server=None`); ignored when `server` is set (the
            server already has its own checkpoint loaded).
        norm_tag: Which normalization statistics the checkpoint expects (e.g.
            "franka_droid"). Only meaningful in-process.
        include_wrist: Whether the checkpoint was trained with a wrist-camera
            input. Only meaningful in-process, and only ever exercised for DROID's
            2-cam interface -- in-process bimanual_yam (3-cam) isn't wired up yet.
        server: `"host:port"` of a running MolmoAct2 inference server --
            `uv run python examples/droid/host_server_droid.py ...` (DROID) or
            `examples/yam/host_server_yam.py ...` (bimanual_yam), both from the
            allenai/molmoact2 repo -- to connect to instead of loading the
            policy in this process. `None` (default) loads in-process here.
    """

    def __init__(self, checkpoint_path=None, norm_tag="franka_droid", device="cuda", include_wrist=True, server=None):
        self.server = server
        if server is not None:
            self._server_url = f"http://{server}/act"  # allenai/molmoact2's host_server_droid.py/host_server_yam.py endpoint
            self._wait_for_server()
            return  # no local model to load -- every infer() call is a remote HTTP request

        from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
        from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

        config = MolmoAct2Config(
            checkpoint_path=checkpoint_path, action_mode="continuous", norm_tag=norm_tag,
            device=device, trust_remote_code=True,
            enable_cuda_graph=False,  # CUDA-graph capture freezes the noise sampling, breaking inference
        )
        self.policy = MolmoAct2Policy(config)
        self.policy.eval()
        self.norm_tag = norm_tag
        self.include_wrist = include_wrist

    def _wait_for_server(self, poll_interval=5):
        """Blocks until the MolmoAct2 server's health check (`GET /act` -- see
        `host_server_droid.py`'s docstring) responds, retrying indefinitely on connection/HTTP
        errors instead of failing fast. Matches the default server mode, which waits
        because the DROID/YAM checkpoints can take minutes to load, and
        `clap-rollout-deploy` is often started before that finishes."""
        import requests

        logger.info(f"Waiting for MolmoAct2 server at {self._server_url}...")
        while True:
            try:
                requests.get(self._server_url, timeout=5).raise_for_status()
                return
            except requests.exceptions.RequestException:
                logger.info("Still waiting for MolmoAct2 server...")
                time.sleep(poll_interval)

    def infer(self, images, joint_position, gripper_position, text):
        """Predict one action chunk.

        Args:
            images: `[right, wrist]` (2 elements, DROID) or `[right, left, wrist]` (3 elements,
                bimanual_yam) — each (192, 320, 3) uint8 camera frame, in `EmbodimentConfig`
                three_view slot order. `_infer_remote` dispatches on `len(images)` to pick the
                right server wire format.
            joint_position: (7,) DROID joint angles, or (14,) bimanual_yam's full dual-arm
                joint+gripper state (gripper columns already embedded per-arm -- bimanual_yam has
                no separate cartesian representation, so this is never an ee/cartesian pose).
            gripper_position: (1,) DROID gripper position, or `None` for bimanual_yam (already
                folded into `joint_position` above).
            text: Task instruction string.

        Returns:
            (num_steps, D) array: D=8 for DROID (columns 0:7 joint position, column 7 gripper),
            D=14 for bimanual_yam (raw joint+gripper positions, no fixed joint/gripper split).
        """
        if gripper_position is not None:
            state = np.concatenate([np.asarray(joint_position, dtype=np.float32), np.asarray(gripper_position, dtype=np.float32)])
        else:
            state = np.asarray(joint_position, dtype=np.float32)
        if self.server is not None:
            return self._infer_remote(images, state, text)
        return self._infer_local(images, state, text)

    def _infer_local(self, images, state, text):
        self.policy.reset()  # drain any stale queued timesteps, forcing a fresh chunk this call
        batch = {"observation.images.cam0": images[0], "observation.state": state, "task": text}
        if self.include_wrist:
            batch["observation.images.cam1"] = images[1]

        # select_action returns only the first timestep and queues the rest internally;
        # drain that queue to recover the policy's full predicted chunk.
        first_action = self.policy.select_action(batch, norm_tag=self.norm_tag)
        first = first_action[0].detach().to("cpu")
        queue = self.policy._action_queues[0]
        rest = []
        while queue:
            rest.append(queue.popleft().detach().to("cpu"))

        import torch
        return torch.stack([first, *rest], dim=0).to(torch.float32).numpy()

    def _infer_remote(self, images, state, text):
        """POST to the DROID/YAM server's /act endpoint (images/instruction/state -> actions).

        Wire format depends on `len(images)`: DROID's `host_server_droid.py` wants 2 cams under
        `external_cam`/`wrist_cam`; bimanual_yam's `host_server_yam.py` wants 3 under
        `top_cam`/`left_cam`/`right_cam`. Camera-slot mapping for the 3-cam case (confirmed):
        `EmbodimentConfig` three_view slot "right" -> top_cam, "left" -> left_cam, "wrist" ->
        right_cam -- i.e. `images` must already be ordered `[right, left, wrist]` by the caller.
        """
        import json_numpy
        import requests

        json_numpy.patch()  # lets json_numpy.dumps/loads handle numpy arrays transparently
        if len(images) == 3:
            cams = {"top_cam": images[0], "left_cam": images[1], "right_cam": images[2]}
        else:
            cams = {"external_cam": images[0], "wrist_cam": images[1]}
        payload = json_numpy.dumps({**cams, "instruction": text, "state": state})
        response = requests.post(self._server_url, data=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = json_numpy.loads(response.text)
        return np.asarray(result["actions"], dtype=np.float32)
