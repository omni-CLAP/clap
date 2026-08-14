"""Wraps an openpi (pi0/pi05/pi0fast) policy behind a simple `act(observation)` interface.

Isolates the optional `openpi`/`openpi_client` dependency to this one module —
importing `clap.rollout` doesn't require them; only constructing an
`OpenPIPolicy` does (install with `pip install clap[openpi]`).
"""

import numpy as np
import torch
import torch.nn.functional as F


_DEFAULT_POLICY_CKPT = "gs://openpi-assets/checkpoints/pi05_droid"  # openpi's published DROID pi05 checkpoint


class OpenPIPolicy:
    """A trained openpi policy (pi0 / pi05 / pi0fast), predicting a joint-velocity + gripper action chunk.

    Args:
        policy_type: "pi05" | "pi0" | "pi0fast" — selects which openpi training
            config (`<policy_type>_droid`) the checkpoint was trained under.
            Only meaningful in-process (`server=None`) — a running server was
            already started against a specific config.
        policy_ckpt: Path to the openpi checkpoint directory, or a `gs://...`
            URI (openpi downloads/caches it on first use). Defaults to
            openpi's published DROID pi05 checkpoint. Only meaningful
            in-process; ignored when `server` is set (the server already has
            its own checkpoint loaded).
        server: `"host:port"` of a running `scripts/serve_policy.py` instance
            (from the openpi repo) to connect to instead of loading the
            policy in this process — lets openpi run in its own environment,
            avoiding any dependency clash with clap's. `None` (default) loads
            the policy in-process here instead.
    """

    def __init__(self, policy_type, policy_ckpt=None, server=None):
        self.policy_type = policy_type
        if server is not None:
            # Server already has its own checkpoint/config loaded; this client only needs
            # openpi_client (numpy + msgpack + websockets), not the full openpi/JAX stack.
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            host, port = server.split(":")
            self.policy = WebsocketClientPolicy(host=host, port=int(port))
        else:
            from openpi.policies import policy_config
            from openpi.training import config as openpi_config

            config_name = f"{policy_type}_droid"  # openpi training configs follow this naming convention
            ckpt = policy_ckpt or _DEFAULT_POLICY_CKPT
            self.policy = policy_config.create_trained_policy(openpi_config.get_config(config_name), ckpt)

    def infer(self, images, joint_position, gripper_position, text):
        """Predict one action chunk.

        Args:
            images: `[right, wrist]` (192, 320, 3) uint8 exterior/wrist camera frames -- openpi
                has no bimanual_yam checkpoint upstream, so this is always exactly 2 images.
            joint_position: (7,) current joint angles.
            gripper_position: (1,) current gripper position.
            text: Task instruction string.

        Returns:
            (num_steps, 8) array: columns 0:7 are joint velocity, column 7 is
            target gripper position.
        """
        from openpi_client import image_tools

        # openpi was trained on 180x320 frames, resized+padded to a square 224x224 input.
        image1 = self._resize_180x320(images[0])
        image2 = self._resize_180x320(images[1])
        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position": joint_position,
            "observation/gripper_position": gripper_position,
            "prompt": text,
        }
        return self.policy.infer(example)["actions"]

    @staticmethod
    def _resize_180x320(image):
        """(192, 320, 3) uint8 -> (180, 320, 3) uint8, the frame size openpi's Franka policies expect."""
        assert image.shape == (192, 320, 3), f"expected (192, 320, 3), got {image.shape}"
        t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, 192, 320)
        t = F.interpolate(t, size=(180, 320), mode="bilinear", align_corners=False)
        return t.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy()