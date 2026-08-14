"""Franka Panda forward kinematics — converts joint angles to an end-effector pose.

Used by the openpi policy-in-the-loop deployment to turn a predicted joint
trajectory into cartesian pose rows for world-model conditioning. Scoped to
Franka/DROID only: joint-space embodiments (bimanual YAM, G1 humanoid) act
directly in joint space and never need this conversion.
"""

import numpy as np

# Modified-DH parameters for the 7 Franka joints + fixed flange offset. DROID's
# recorded cartesian_position/state convention stops at this flange frame --
# verified against real recorded (joint_angles, cartesian_pose) pairs from DROID
# episodes, to ~1e-8 xyz/rpy error.
_DH_FIXED_ROWS = [
    [0, 0.107, 0, 0],
    # The 2 rows below chain a further -pi/4 tool-frame rotation + a 0.1034m offset
    # (a gripper-TCP frame, past the flange). Left here commented out rather than
    # deleted: it's a real, standard Franka/panda_hand_tcp offset, just not the one
    # DROID's recorded convention uses -- enabling it reintroduces a ~0.1m xyz /
    # ~45deg orientation error against real DROID ground truth (verified above).
    # [0, 0, 0, -np.pi / 4],
    # [0.0, 0.1034, 0, 0],
]


def _dh_transform(a, d, alpha, theta):
    """One modified-DH link's 4x4 homogeneous transform."""
    q = theta
    return np.array([
        [np.cos(q), -np.sin(q), 0, a],
        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
        [0, 0, 0, 1],
    ])


def get_fk_solution(joint_angles):
    """7 joint angles -> 4x4 world-from-end-effector homogeneous transform.

    Callers typically extract `T[:3, 3]` (xyz) and
    `scipy.spatial.transform.Rotation.from_matrix(T[:3, :3]).as_euler('xyz')`
    (roll/pitch/yaw) to build a cartesian pose row.
    """
    dh_params = [  # modified-DH [a, d, alpha, theta] rows, one per joint
        [0, 0.333, 0, joint_angles[0]],
        [0, 0, -np.pi / 2, joint_angles[1]],
        [0, 0.316, np.pi / 2, joint_angles[2]],
        [0.0825, 0, np.pi / 2, joint_angles[3]],
        [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
        [0, 0, np.pi / 2, joint_angles[5]],
        [0.088, 0, np.pi / 2, joint_angles[6]],
        *_DH_FIXED_ROWS,  # append the fixed flange-offset rows
    ]
    T = np.eye(4)  # start from identity (base frame)
    for row in dh_params:  # chain all 7 joint transforms + 2 fixed flange offsets
        T = T @ _dh_transform(*row)
    return T  # world-from-end-effector transform
