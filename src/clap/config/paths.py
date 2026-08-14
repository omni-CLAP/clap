import os
from dataclasses import dataclass, field


def _env(name, default=None):
    """Read an env var at PathConfig-construction time (not import time), so
    tests/scripts can set os.environ before building a PathConfig."""
    return os.environ.get(name, default)


def _require_env(name):
    value = os.environ.get(name)
    if value is None:
        raise EnvironmentError(f"Required environment variable {name} is not set.")  # no sane default exists
    return value


@dataclass
class PathConfig:
    """Cluster/environment-specific paths, resolved from `CLAP_*` env vars.

    Every path a training/eval/rollout run needs comes from here, set once in
    a sourced env file, never hardcoded in code.

    Attributes:
        checkpoint_root: Base dir `finetune_ckpt`/output_dir paths resolve
            against when given as a relative path (an absolute path overrides
            this). Supports symlinks for cross-referencing paths.
        oxe_lam_root: Only needed when training with conditioning="lam"; may
            differ from oxe_base_path if LAM latent-action extractions live in
            a separate tree. The actual subdir name under it
            (`DataConfig.oxe_lam_subdir`, default "latent_actions") is a
            per-run/config choice, not a path — see `configs/data/lam.yaml`
            and `CLAP_OXE_LAM_SUBDIR` for how to point it at a specific
            extraction run without editing code.
    """

    # HF repo id or local path; falls back to downloading the public HF model if unset.
    svd_model_path: str = field(default_factory=lambda: _env("CLAP_SVD_MODEL_PATH", "stabilityai/stable-video-diffusion-img2vid"))
    clip_model_path: str = field(default_factory=lambda: _env("CLAP_CLIP_MODEL_PATH", "openai/clip-vit-base-patch32"))

    oxe_base_path: str = field(default_factory=lambda: _require_env("CLAP_OXE_BASE_PATH"))  # required: no sane default
    oxe_lam_root: str = field(default_factory=lambda: _env("CLAP_OXE_LAM_ROOT"))
    dataset_meta_info_root: str = field(default_factory=lambda: _env("CLAP_META_INFO_ROOT", "dataset_meta_info"))

    checkpoint_root: str = field(default_factory=lambda: _env("CLAP_CHECKPOINT_ROOT", "model_ckpt"))

    def resolve_checkpoint(self, path: str) -> str:
        """Join a relative checkpoint path against `checkpoint_root`; absolute paths pass through unchanged."""
        if path is None or os.path.isabs(path):
            return path
        return os.path.join(self.checkpoint_root, path)