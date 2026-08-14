"""Converts an openpi policy's predicted joint velocities into joint positions.

`CLAPModel` is trained on cartesian/joint *position* actions, but the openpi
pi0/pi05 policies predict joint *velocities* — this adapter bridges the two by
predicting a joint-position delta from a window of (position, velocity) pairs.
"""

import einops
import numpy as np
import torch
import torch.nn as nn


class EEVelocityToPositionAdapter(nn.Module):
    """Predicts a per-step joint-position delta from the current joint position + a window of joint velocities.

    Args:
        action_dim: Joint-space dimensionality (7 for a Franka arm).
        action_num: Number of future velocity steps in one prediction window.
    """

    # p01/p99 normalization bounds, fit once on real DROID joint-velocity/joint-delta data.
    JOINT_VEL_P01 = np.array([-0.4077107, -0.79047304, -0.47850373, -0.8666644, -0.6729502, -0.5602032, -0.692411])[None, :]
    JOINT_VEL_P99 = np.array([0.4900636, 0.7259861, 0.45910007, 0.79220384, 0.69864315, 0.648198, 0.810115])[None, :]
    JOINT_DELTA_P01 = np.array([-0.2801219, -0.397792, -0.22935797, -0.3351759, -0.42025003, -0.36825255, -0.450706])[None, :]
    JOINT_DELTA_P99 = np.array([0.2827909, 0.42184818, 0.33529875, 0.35958457, 0.375613, 0.44463825, 0.4697690])[None, :]

    def __init__(self, action_dim, action_num, hidden_size=512):
        """Build the joint/velocity -> position-delta MLP.

        Args:
            hidden_size: Width of the two hidden layers.
        """
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        # Input: current joint position (1 step) + the velocity window (action_num steps), each action_dim-wide.
        input_dim = action_dim * (action_num + 1)
        output_dim = action_num * action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, output_dim),
        )

    @staticmethod
    def _normalize(data, data_min, data_max, eps=1e-8):
        """Min-max normalize `data` from [data_min, data_max] to [-1, 1]."""
        return 2 * (data - data_min) / (data_max - data_min + eps) - 1

    @staticmethod
    def _denormalize(data, data_min, data_max, eps=1e-8):
        """Invert `_normalize`: map [-1, 1] back to [data_min, data_max]."""
        return (data + 1) / 2 * (data_max - data_min + eps) + data_min

    def forward(self, joint, joint_vel, joint_delta=None, training=True):
        """Predict future joint positions from a current joint + velocity window.

        Args:
            joint: (B, 1, action_dim) current joint position.
            joint_vel: (B, action_num, action_dim) predicted velocity window.
            joint_delta: (B, action_num, action_dim) ground-truth position delta;
                required when training=True (used as the regression target).
            training: If True, returns the MSE loss; if False, returns the
                predicted future joint positions instead (numpy, batch dim dropped).
        """
        device = next(self.parameters()).device
        if joint.ndim == 2:
            joint = joint[None, :]
        if joint_vel.ndim == 2:
            joint_vel = joint_vel[None, :]
        assert joint.shape[1:] == (1, self.action_dim), f"expected (B, 1, {self.action_dim}), got {joint.shape}"
        assert joint_vel.shape[1:] == (self.action_num, self.action_dim), \
            f"expected (B, {self.action_num}, {self.action_dim}), got {joint_vel.shape}"

        joint_t = torch.as_tensor(joint, dtype=torch.float32, device=device)
        joint_vel_norm = self._normalize(joint_vel, self.JOINT_VEL_P01, self.JOINT_VEL_P99)
        joint_vel_t = torch.as_tensor(joint_vel_norm, dtype=torch.float32, device=device)

        B = joint_t.shape[0]
        net_input = torch.cat([joint_t.reshape(B, -1), joint_vel_t.reshape(B, -1)], dim=1)  # (B, action_dim*(action_num+1))
        pred = self.net(net_input)  # (B, action_num*action_dim)
        pred = einops.rearrange(pred, "b (t d) -> b t d", t=self.action_num, d=self.action_dim)  # (B, action_num, action_dim)

        if training:
            target = self._normalize(joint_delta, self.JOINT_DELTA_P01, self.JOINT_DELTA_P99)
            target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
            return nn.functional.mse_loss(pred, target_t)

        # Inference: denormalize the predicted delta and add it to the current joint position.
        pred_delta = self._denormalize(pred.detach().cpu().numpy(), self.JOINT_DELTA_P01, self.JOINT_DELTA_P99)
        joint_future = joint_t.detach().cpu().numpy() + pred_delta
        return joint_future[0]  # (action_num, action_dim), batch dim dropped (inference is always batch_size=1)
