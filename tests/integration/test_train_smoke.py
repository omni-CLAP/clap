"""Real end-to-end training smoke test: a few real steps on a tiny debug slice, on GPU.

Not a correctness check on the learned weights — just confirms the full
dataset -> dataloader -> model -> optimizer -> checkpoint path runs without
crashing, on real data and real SVD/CLIP weights.
"""

import os

import pytest

from tests.conftest import requires_gpu, requires_oxe_data


def _resolvable_env_path(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@requires_gpu
@requires_oxe_data
def test_train_a_few_steps_on_debug_slice(oxe_base_path, tmp_path, monkeypatch):
    svd_model_path = _resolvable_env_path("CLAP_SVD_MODEL_PATH", "SVD_MODEL_PATH")
    clip_model_path = _resolvable_env_path("CLAP_CLIP_MODEL_PATH", "CLIP_MODEL_PATH")
    if not svd_model_path or not clip_model_path:
        pytest.skip("CLAP_SVD_MODEL_PATH/CLAP_CLIP_MODEL_PATH (or SVD_MODEL_PATH/CLIP_MODEL_PATH) not set")
    meta_info_path = _resolvable_env_path("CLAP_META_INFO_ROOT")
    if not meta_info_path or not os.path.isdir(meta_info_path):
        pytest.skip("CLAP_META_INFO_ROOT not set to an existing directory")
    monkeypatch.setenv("WANDB_MODE", "disabled")  # a smoke test shouldn't create a real cloud run

    from clap.config import CLAPModelConfig, DataConfig, PathConfig, TrainingConfig, TrainingRunConfig
    from clap.training.train import main

    config = TrainingRunConfig(
        model=CLAPModelConfig(
            svd_model_path=svd_model_path, clip_model_path=clip_model_path,
            conditioning="ee", action_dim=7, num_history=6, num_frames=5,
        ),
        data=DataConfig(
            conditioning="ee", oxe_base_path=oxe_base_path, dataset_meta_info_path=meta_info_path, debug_dataset=True,
        ),
        training=TrainingConfig(
            output_dir=str(tmp_path / "smoke_run"), tag="smoke_test",
            train_batch_size=1, gradient_accumulation_steps=1, mixed_precision="bf16",
            max_train_steps=2, checkpointing_steps=2, validation_steps=1000,
        ),
    )
    paths = PathConfig()

    main(config, paths)

    assert os.path.isfile(os.path.join(config.training.output_dir, "last.pt"))
