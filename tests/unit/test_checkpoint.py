"""Unit tests for clap.training.checkpoint: save/load round-trip, using a tiny dummy
model instead of a real CLAPModel so these run fast without GPU/SVD weights."""

import os

import torch
import torch.nn as nn

from clap.training.checkpoint import load_checkpoint_for_resume, save_checkpoint


class _TinyModel(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()
        self.action_encoder = nn.Linear(action_dim, 4)
        self.backbone = nn.Linear(4, 4)


def _make_model_and_optimizer(action_dim=7):
    model = _TinyModel(action_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_checkpoint_writes_step_file_and_last_pt_hardlink(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    save_path, last_path = save_checkpoint(model, optimizer, global_step=5, output_dir=str(tmp_path))

    assert os.path.isfile(save_path)
    assert os.path.basename(save_path) == "checkpoint-5.pt"
    assert os.path.isfile(last_path)
    # last.pt is a hard link to the same inode, not a copy.
    assert os.stat(save_path).st_ino == os.stat(last_path).st_ino


def test_save_then_load_round_trips_weights_optimizer_and_step(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    with torch.no_grad():
        model.action_encoder.weight.fill_(0.5)
    save_checkpoint(model, optimizer, global_step=42, output_dir=str(tmp_path))

    fresh_model, _ = _make_model_and_optimizer()
    optim_state, step = load_checkpoint_for_resume(fresh_model, os.path.join(tmp_path, "last.pt"))

    assert step == 42
    assert optim_state is not None
    assert torch.allclose(fresh_model.action_encoder.weight, torch.full_like(fresh_model.action_encoder.weight, 0.5))


def test_load_checkpoint_finetune_mode_ignores_training_state(tmp_path):
    model, optimizer = _make_model_and_optimizer()
    save_checkpoint(model, optimizer, global_step=100, output_dir=str(tmp_path))

    fresh_model, _ = _make_model_and_optimizer()
    optim_state, step = load_checkpoint_for_resume(
        fresh_model, os.path.join(tmp_path, "last.pt"), load_training_state=False,
    )
    assert optim_state is None
    assert step == 0


def test_load_checkpoint_non_strict_drops_shape_mismatched_keys(tmp_path):
    """Cross-action_dim finetune: source has action_dim=7, target model has action_dim=14."""
    source_model, optimizer = _make_model_and_optimizer(action_dim=7)
    save_checkpoint(source_model, optimizer, global_step=1, output_dir=str(tmp_path))

    target_model = _TinyModel(action_dim=14)
    original_weight = target_model.action_encoder.weight.clone()
    load_checkpoint_for_resume(target_model, os.path.join(tmp_path, "last.pt"), strict_resume=False)

    # action_encoder shape mismatch (7 vs 14) -> kept its fresh init, unchanged.
    assert torch.equal(target_model.action_encoder.weight, original_weight)
    # backbone shape matches (4x4 either way) -> transferred from the checkpoint.
    assert torch.equal(target_model.backbone.weight, source_model.backbone.weight)


def test_load_checkpoint_reset_action_encoder_drops_it_even_when_shape_matches(tmp_path):
    source_model, optimizer = _make_model_and_optimizer(action_dim=7)
    with torch.no_grad():
        source_model.action_encoder.weight.fill_(9.0)
    save_checkpoint(source_model, optimizer, global_step=1, output_dir=str(tmp_path))

    target_model, _ = _make_model_and_optimizer(action_dim=7)  # same shape, so it WOULD transfer without the flag
    original_weight = target_model.action_encoder.weight.clone()
    load_checkpoint_for_resume(
        target_model, os.path.join(tmp_path, "last.pt"), strict_resume=True, reset_action_encoder=True,
    )

    assert torch.equal(target_model.action_encoder.weight, original_weight)  # NOT the source's filled 9.0
    assert torch.equal(target_model.backbone.weight, source_model.backbone.weight)  # backbone still transfers


def test_load_checkpoint_legacy_flat_state_dict_parses_step_from_filename(tmp_path):
    """A bare state_dict (no {"model": ...} wrapper) is a legacy/foreign checkpoint format."""
    model, _ = _make_model_and_optimizer()
    ckpt_path = tmp_path / "checkpoint-77.pt"
    torch.save(model.state_dict(), ckpt_path)

    fresh_model, _ = _make_model_and_optimizer()
    optim_state, step = load_checkpoint_for_resume(fresh_model, str(ckpt_path))

    assert optim_state is None
    assert step == 77
