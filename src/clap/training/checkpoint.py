"""Checkpoint save/load, including cross-checkpoint finetune loading.

Handles the structured checkpoint format ({"model", "optimizer", "global_step"})
this package writes, plus resuming step-counting from a bare filename when the
optimizer/step aren't otherwise available.
"""

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)

_CKPT_NAME_RE = re.compile(r"checkpoint-(\d+)\.pt$")


def _parse_step_from_filename(path):
    """Extract N from a 'checkpoint-N.pt' filename; None if it doesn't match."""
    m = _CKPT_NAME_RE.match(os.path.basename(path))
    return int(m.group(1)) if m else None


def _resolve_step_for_last_pt(last_path):
    """`last.pt` is a hard link to some 'checkpoint-N.pt'; find N by matching inodes."""
    if not os.path.exists(last_path):
        return None  # no last.pt at all
    try:
        target_inode = os.stat(last_path).st_ino  # inode identifies which checkpoint-N.pt last.pt links to
    except OSError:
        return None
    parent = os.path.dirname(last_path) or "."
    try:
        files = os.listdir(parent)
    except OSError:
        return None
    for fname in files:
        m = _CKPT_NAME_RE.match(fname)
        if not m:
            continue  # not a checkpoint-N.pt file
        try:
            if os.stat(os.path.join(parent, fname)).st_ino == target_inode:
                return int(m.group(1))  # found the matching hard-linked file
        except OSError:
            continue
    return None  # no checkpoint-N.pt shares last.pt's inode


def _filter_shape_mismatched(model, state_dict):
    """Drop checkpoint keys whose tensor shape doesn't match the live model.

    Used for finetunes across a changed action_dim (e.g. LAM 32-d ckpt loaded
    into an EE 7-d model): the mismatched layer keeps its fresh random init;
    every shape-compatible layer still transfers.
    """
    model_sd = model.state_dict()
    kept, dropped = {}, []
    for k, v in state_dict.items():
        if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape):
            dropped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))  # record ckpt shape vs. model shape
            continue
        kept[k] = v  # shape-compatible (or key not present in model_sd at all): keep as-is
    if dropped:
        logger.info(f"🔀 dropped {len(dropped)} shape-mismatched key(s) (those layers keep their fresh init):")
        for k, ck, mk in dropped:
            logger.info(f"    {k}: ckpt {ck} != model {mk}")
    return kept


def _drop_module_keys(state_dict, prefix):
    """Drop every checkpoint key starting with `prefix`, regardless of shape.

    Used when a whole submodule should start from random init even where its
    shape happens to match — e.g. resetting `action_encoder.*` on a
    cross-action_dim finetune, since partially transferring only the
    shape-matched layers otherwise anchors the MLP to the source embodiment's
    feature basin and hurts downstream learning.
    """
    kept, dropped = {}, []
    for k, v in state_dict.items():
        if k.startswith(prefix):
            dropped.append(k)  # under the target prefix: drop regardless of shape
            continue
        kept[k] = v
    if dropped:
        logger.info(f"🔀 dropped {len(dropped)} key(s) under prefix '{prefix}' (those params keep their fresh init):")
        for k in dropped:
            logger.info(f"    {k}")
    return kept


def load_checkpoint_for_resume(model, ckpt_path, strict_resume=True, load_training_state=True, reset_action_encoder=False):
    """Load a checkpoint into `model`.

    Args:
        strict_resume: If False, drop checkpoint keys whose shape doesn't
            match the live model (a finetune loading a different action_dim).
        load_training_state: If False, ignore the checkpoint's optimizer state
            and step count — used for finetune loads, which start a fresh
            optimizer at step 0 regardless of what the source checkpoint carries.
        reset_action_encoder: Drop every `action_encoder.*` key (not just
            shape-mismatched ones); see `_drop_module_keys`.

    Returns:
        (optimizer_state_dict_or_None, global_step)
    """
    logger.info(f"📂 Loading checkpoint from {ckpt_path} "
                f"(strict_resume={strict_resume}, load_training_state={load_training_state}, "
                f"reset_action_encoder={reset_action_encoder})")
    blob = torch.load(ckpt_path, map_location="cpu")

    # Structured checkpoints (this package's save_checkpoint) nest the weights
    # under "model"; a bare state_dict (a legacy/foreign checkpoint) is used as-is.
    sd = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if not strict_resume:
        sd = _filter_shape_mismatched(model, sd)  # e.g. finetuning into a different action_dim
    if reset_action_encoder:
        sd = _drop_module_keys(sd, "action_encoder.")  # force the whole MLP to retrain from scratch
    # strict=False: any key filtered out above must still be tolerated as "missing".
    model.load_state_dict(sd, strict=False)

    if not load_training_state:
        # Finetune loads: weights only, start optimizer/step fresh.
        logger.info("🌱 load_training_state=False -> fresh optimizer + step=0")
        return None, 0

    if isinstance(blob, dict) and "model" in blob:
        # Structured checkpoint: optimizer state and step were saved alongside the weights.
        optim_state = blob.get("optimizer")
        step = int(blob.get("global_step", 0))
        logger.info(f"✅ global_step={step}, optimizer_state={'yes' if optim_state else 'no'}")
        return optim_state, step

    # Legacy flat state_dict with no recorded step: recover it from the filename instead.
    step = _resolve_step_for_last_pt(ckpt_path) if os.path.basename(ckpt_path) == "last.pt" else _parse_step_from_filename(ckpt_path)
    if step is None:
        logger.warning("⚠️ could not parse step from filename — starting at step 0 (no optimizer state either)")
        return None, 0
    logger.info(f"✅ parsed global_step={step} from filename (no optimizer state in this checkpoint)")
    return None, step


def save_checkpoint(unwrapped_model, optimizer, global_step, output_dir):
    """Write a structured checkpoint, then atomically refresh `last.pt` as a hard link to it."""
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
    torch.save({
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
    }, save_path)

    # Hard-link through a temp name + rename so a reader never sees a half-written last.pt.
    last_path = os.path.join(output_dir, "last.pt")
    tmp_path = last_path + ".tmp"
    if os.path.lexists(tmp_path):
        os.remove(tmp_path)
    os.link(save_path, tmp_path)
    os.replace(tmp_path, last_path)
    return save_path, last_path
