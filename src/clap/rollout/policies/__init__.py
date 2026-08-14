"""Real-robot policies.

`openpi_policy`/`molmoact_policy` each require their own optional heavy
dependency (`openpi`/`lerobot`-with-MolmoAct2) — neither is imported here, so
`clap.rollout.policies` stays importable without either installed; only
constructing `OpenPIPolicy`/`MolmoActPolicy` requires their own package.
"""

from clap.rollout.policies.ee_velocity_to_position_adapter import EEVelocityToPositionAdapter

__all__ = [
    "EEVelocityToPositionAdapter",
]
