from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from clap.config.data import DataConfig
from clap.config.model import CLAPModelConfig
from clap.config.training import TrainingConfig


@dataclass
class TrainingRunConfig:
    """Everything one `clap-train` invocation needs: model + data + training config."""

    model: CLAPModelConfig
    data: DataConfig
    training: TrainingConfig


def load_config(path: str, overrides: dict = None) -> TrainingRunConfig:
    """Load a `TrainingRunConfig` from a YAML file, resolving `defaults:` composition and CLI overrides.

    Args:
        path: `configs/experiment/<name>.yaml` (or any file matching TrainingRunConfig's
            model:/data:/training: shape). May start with a Hydra-style `defaults:` list,
            e.g. `defaults: [{model: base}, {data: ee}]`, which merges in
            `<configs_root>/model/base.yaml` under `model:` and `<configs_root>/data/ee.yaml`
            under `data:` (in list order) before this file's own body is applied on top —
            `<configs_root>` is `path`'s grandparent directory (`configs/experiment/x.yaml` -> `configs/`).
        overrides: e.g. `{"training.finetune_ckpt": "..."}` from `--override key=value`
            CLI flags, applied last (highest priority, over both defaults and the file body).
    """
    file_cfg = OmegaConf.load(path)
    defaults = file_cfg.pop("defaults", None) or []

    schema = OmegaConf.structured(TrainingRunConfig)  # dataclass defaults + type-checking
    configs_root = Path(path).resolve().parent.parent
    merged = schema
    for entry in defaults:
        for group, name in entry.items():
            default_cfg = OmegaConf.load(configs_root / group / f"{name}.yaml")
            merged = OmegaConf.merge(merged, OmegaConf.create({group: default_cfg}))
    merged = OmegaConf.merge(merged, file_cfg)  # the file's own body overrides its defaults

    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]))
    return OmegaConf.to_object(merged)  # back to a plain (nested) dataclass instance
