"""Unit tests for clap.config: dataclass defaults, env-var resolution, YAML composition/overrides."""

import os

import pytest

from clap.config.data import DataConfig, EmbodimentConfig
from clap.config.model import CLAPModelConfig
from clap.config.paths import PathConfig
from clap.config.run import load_config
from clap.config.training import TrainingConfig


def test_model_config_num_total_frames():
    cfg = CLAPModelConfig(svd_model_path="x", clip_model_path="y", num_history=6, num_frames=5)
    assert cfg.num_total_frames == 11


def test_data_config_video_size():
    cfg = DataConfig(height=576, width=320)
    assert cfg.video_size == (576, 320)


def test_data_config_single_dataset_default_none():
    cfg = DataConfig(conditioning="lam")
    assert cfg.single_dataset is None


def test_embodiment_config_defaults():
    cfg = EmbodimentConfig(name="bridge", cam_id="rgb")
    assert cfg.action_mode == "ee7"
    assert cfg.stacking_mode is None


def test_training_config_requires_output_dir_and_tag():
    with pytest.raises(TypeError):
        TrainingConfig()  # output_dir/tag have no default


def test_path_config_missing_required_env_raises(monkeypatch):
    monkeypatch.delenv("CLAP_OXE_BASE_PATH", raising=False)
    with pytest.raises(EnvironmentError):
        PathConfig()


def test_path_config_resolve_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAP_OXE_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("CLAP_CHECKPOINT_ROOT", "model_ckpt")
    paths = PathConfig()
    assert paths.resolve_checkpoint("foo/last.pt") == os.path.join("model_ckpt", "foo/last.pt")
    assert paths.resolve_checkpoint("/abs/last.pt") == "/abs/last.pt"
    assert paths.resolve_checkpoint(None) is None


def test_load_config_defaults_composition(tmp_path):
    """A `defaults:` list pulls in configs/<group>/<name>.yaml under the matching top-level key."""
    configs_root = tmp_path / "configs"
    (configs_root / "model").mkdir(parents=True)
    (configs_root / "data").mkdir(parents=True)
    (configs_root / "experiment").mkdir(parents=True)

    (configs_root / "model" / "base.yaml").write_text(
        "svd_model_path: fake-svd\nclip_model_path: fake-clip\nnum_history: 6\nnum_frames: 12\n"
    )
    (configs_root / "data" / "ee.yaml").write_text("conditioning: ee\noxe_base_path: /fake/oxe\n")
    (configs_root / "experiment" / "test_exp.yaml").write_text(
        "defaults:\n  - model: base\n  - data: ee\n\n"
        "training:\n  tag: test_exp\n  output_dir: model_ckpt/test_exp\n  train_batch_size: 2\n"
    )

    cfg = load_config(str(configs_root / "experiment" / "test_exp.yaml"))
    assert cfg.model.svd_model_path == "fake-svd"
    assert cfg.model.num_frames == 12  # from the model:base default
    assert cfg.data.conditioning == "ee"  # from the data:ee default
    assert cfg.data.oxe_base_path == "/fake/oxe"
    assert cfg.training.tag == "test_exp"
    assert cfg.training.train_batch_size == 2


def test_load_config_overrides_take_precedence(tmp_path):
    configs_root = tmp_path / "configs"
    (configs_root / "model").mkdir(parents=True)
    (configs_root / "data").mkdir(parents=True)
    (configs_root / "experiment").mkdir(parents=True)
    (configs_root / "model" / "base.yaml").write_text("svd_model_path: fake-svd\nclip_model_path: fake-clip\n")
    (configs_root / "data" / "ee.yaml").write_text("conditioning: ee\noxe_base_path: /fake/oxe\n")
    (configs_root / "experiment" / "test_exp.yaml").write_text(
        "defaults:\n  - model: base\n  - data: ee\n\ntraining:\n  tag: test_exp\n  output_dir: out\n"
    )

    cfg = load_config(
        str(configs_root / "experiment" / "test_exp.yaml"),
        overrides={"training.train_batch_size": 4, "data.conditioning": "droid"},
    )
    assert cfg.training.train_batch_size == 4
    assert cfg.data.conditioning == "droid"


def test_load_config_without_defaults_is_self_contained(tmp_path):
    """A file with no `defaults:` key must still load (all its own model:/data:/training: sections)."""
    path = tmp_path / "self_contained.yaml"
    path.write_text(
        "model:\n  svd_model_path: fake-svd\n  clip_model_path: fake-clip\n"
        "data:\n  conditioning: droid\n  oxe_base_path: /fake/oxe\n"
        "training:\n  tag: t\n  output_dir: out\n"
    )
    cfg = load_config(str(path))
    assert cfg.data.conditioning == "droid"
