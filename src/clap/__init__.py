"""CLAP: cross-embodiment action-conditioned video world model."""

from clap.config.model import CLAPModelConfig
from clap.models.model import CLAPModel

__version__ = "0.1.0"

__all__ = [
    "CLAPModel",
    "CLAPModelConfig",
    "__version__",
]
