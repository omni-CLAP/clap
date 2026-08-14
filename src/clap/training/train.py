"""Training entrypoint: `clap-train --config <path> [--override key=value ...]`.

Covers pretraining, post-training, and novel-embodiment adaptation alike —
they're the same loop, distinguished only by `TrainingConfig.finetune_ckpt`
and `DataConfig.conditioning`, not separate code paths.
"""

import argparse
import datetime
import math
import os
import signal
import sys

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm

from clap.config import PathConfig, load_config
from clap.data import build_dataset
from clap.utils import setup_logging
from clap.models import CLAPModel
from clap.training.checkpoint import load_checkpoint_for_resume, save_checkpoint
from clap.training.dataloader import build_dataloader
from clap.training.validation import validate_video_generation

try:
    import wandb
except ImportError:
    wandb = None

setup_logging()  # rich console handler; without this, logger.info(...) below is silently dropped
logger = get_logger(__name__, log_level="INFO")


def _resolve_resume_source(training_config, paths: PathConfig):
    """Decide which checkpoint (if any) to load, and under what loading rules.

    Auto-resume always wins: if this run's own output_dir already has a
    last.pt, we're continuing it — full optimizer/step restore, strict
    shapes, no action_encoder reset (the run's own checkpoint is by
    definition shape-compatible, and resetting mid-run would erase progress).
    Otherwise, an explicit finetune_ckpt starts a fresh optimizer/step at
    whatever weights transfer (dropping shape-mismatched keys).

    Returns:
        (ckpt_path_or_None, strict_resume, load_training_state, reset_action_encoder)
    """
    auto_last = os.path.join(training_config.output_dir, "last.pt")
    if os.path.exists(auto_last):
        # Continuing this exact run: strict shapes, full optimizer/step restore.
        logger.info(f"🔄 Auto-resuming from {auto_last}")
        return auto_last, True, True, False

    if training_config.finetune_ckpt:
        # Starting a new run from someone else's checkpoint: tolerate shape
        # mismatches and always start the optimizer/step fresh.
        ckpt = paths.resolve_checkpoint(training_config.finetune_ckpt)
        logger.info(f"🌱 Finetuning from {ckpt} (fresh optimizer, step=0)")
        return ckpt, False, False, training_config.reset_action_encoder

    return None, True, True, False  # no checkpoint configured; train from scratch


def _named_submodules(model):
    """(name, submodule) pairs for the per-component parameter-count log at startup."""
    mods = [
        ("unet", model.unet), ("vae", model.vae), ("image_encoder", model.image_encoder),
        ("text_encoder", model.text_encoder), ("action_encoder", model.action_encoder),
    ]
    if getattr(model, "action_adapter", None) is not None:
        mods.append(("action_adapter", model.action_adapter))  # only present in adapter-conditioning runs
    return mods


def _end_training_with_timeout(accelerator, timeout=1200):
    """accelerator.end_training() can hang indefinitely if wandb keeps retrying failed uploads
    (e.g. no outbound route to storage.googleapis.com from this compute node) -- even though
    training itself already finished successfully. Give it `timeout` seconds, then give up
    rather than block the job from exiting; doesn't touch wandb's online/offline mode or
    anything during the run itself, only this final flush.
    """
    def _on_timeout(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(timeout)
    try:
        accelerator.end_training()
    except TimeoutError:
        logger.warning(f"⚠️ accelerator.end_training() didn't finish within {timeout}s (likely wandb "
                        f"stuck retrying uploads) -- giving up on it; training itself already completed")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def main(config, paths: PathConfig):
    """Run one training job end-to-end: build/resume the model, then loop until max_train_steps.

    Args:
        config: Full `TrainingRunConfig` (model/data/training sections).
        paths: Resolved `CLAP_*` path config, used to resolve `finetune_ckpt`.
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb" if wandb is not None else None,
        project_dir=config.training.output_dir,
    )

    model = CLAPModel(config.model)

    # Load a checkpoint (if any) before moving to device, so state_dict shapes
    # are compared/loaded on CPU tensors first.
    ckpt_path, strict_resume, load_training_state, reset_action_encoder = _resolve_resume_source(config.training, paths)
    resumed_optim_state, resumed_step = None, 0
    if ckpt_path is not None:
        if os.path.exists(ckpt_path):
            resumed_optim_state, resumed_step = load_checkpoint_for_resume(
                model, ckpt_path, strict_resume=strict_resume,
                load_training_state=load_training_state, reset_action_encoder=reset_action_encoder,
            )
        else:
            logger.warning(f"⚠️ ckpt_path={ckpt_path} does not exist; training from scratch")

    model.to(accelerator.device)
    model.train()

    # Adapter-only training freezes everything else (CLAPModel.__init__ already
    # set requires_grad accordingly); filter here so the optimizer's state and
    # LR schedule only ever see the params actually being trained.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.model.train_action_adapter and accelerator.is_main_process:
        n_train = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(f"🧩 [adapter mode] trainable params: {n_train / 1e6:.2f}M / total: {n_total / 1e6:.2f}M")

    # bitsandbytes' 8-bit optimizer trades a little precision for much less
    # optimizer-state memory — useful for the larger 4-camera G1 humanoid input.
    optimizer_cls = torch.optim.AdamW
    if config.training.use_8bit_optimizer:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    optimizer = optimizer_cls(trainable_params, lr=config.training.learning_rate)
    if resumed_optim_state is not None:
        try:
            optimizer.load_state_dict(resumed_optim_state)
            logger.info("✅ optimizer state restored")
        except Exception as e:
            # Shape/param-group mismatches (e.g. resuming after an architecture
            # change) shouldn't crash the run — just start the optimizer fresh.
            logger.warning(f"⚠️ optimizer restore failed ({e}); continuing with a fresh optimizer")

    if accelerator.is_main_process:
        run_name = config.training.wandb_run_name or f"train_{datetime.datetime.now():%Y-%m-%dT%H-%M-%S}_{config.training.tag}"
        if wandb is not None:
            accelerator.init_trackers(config.training.wandb_project_name, config={}, init_kwargs={"wandb": {"name": run_name}})
        os.makedirs(config.training.output_dir, exist_ok=True)
        for name, mod in _named_submodules(model):
            n = sum(p.numel() for p in mod.parameters())
            logger.info(f"📊 parameters in {name}: {n / 1e6:.2f}M")

    train_dataset = build_dataset(config.data, config.model, mode="train", egodex_lam_subdir=config.data.egodex_lam_subdir)
    val_dataset = build_dataset(config.data, config.model, mode="val", egodex_lam_subdir=config.data.egodex_lam_subdir)
    if len(train_dataset) == 0:
        raise RuntimeError(
            f"train dataset is empty for conditioning={config.data.conditioning}. "
            f"Check oxe_base_path={config.data.oxe_base_path!r} (and oxe_lam_root for conditioning='lam')."
        )
    train_dataloader = build_dataloader(train_dataset, config.training, config.data, mode="train")
    val_dataloader = build_dataloader(val_dataset, config.training, config.data, mode="val")

    # accelerate wraps the model for DDP, shards the dataloaders across ranks,
    # and moves the optimizer's state to the right device.
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    total_batch_size = config.training.train_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    # Dataloaders don't have a fixed epoch count in terms of desired steps, so
    # derive how many epochs are needed to reach max_train_steps.
    num_train_epochs = math.ceil(
        config.training.max_train_steps * config.training.gradient_accumulation_steps * total_batch_size / max(1, len(train_dataloader))
    )
    logger.info("🚀 Running training")
    logger.info(f"  num examples = {len(train_dataset)}")
    logger.info(f"  total train batch size = {total_batch_size}")
    logger.info(f"  total optimization steps = {config.training.max_train_steps}")

    global_step = resumed_step
    train_loss = 0.0
    progress_bar = tqdm(initial=global_step, total=config.training.max_train_steps, disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    done = False
    for _epoch in range(num_train_epochs):
        if done:
            break
        for batch in train_dataloader:
            if global_step >= config.training.max_train_steps:
                done = True
                break

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, _ = model(batch)
                # Gather each rank's loss before averaging — a single rank's
                # loss only reflects its own local batch otherwise.
                avg_loss = accelerator.gather(loss.repeat(config.training.train_batch_size)).mean()
                train_loss += avg_loss.item() / config.training.gradient_accumulation_steps
                accelerator.backward(loss)
                if accelerator.sync_gradients:  # true once per optimizer step, not per micro-batch
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 100 == 0:
                    progress_bar.set_postfix({"loss": train_loss})
                    accelerator.log({"train_loss": train_loss / 100}, step=global_step)
                    train_loss = 0.0

                if global_step % config.training.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path, last_path = save_checkpoint(accelerator.unwrap_model(model), optimizer, global_step, config.training.output_dir)
                    logger.info(f"💾 saved checkpoint to {save_path} (refreshed {last_path})")

                if global_step % config.training.validation_steps == 0 and accelerator.is_main_process:
                    model.eval()
                    with accelerator.autocast():
                        # Each preview call renders 2 clips; cover video_num clips
                        # total (or all configured val_sample_ids, if given).
                        num_previews = math.ceil(len(config.data.val_sample_ids) / 2) if config.data.val_sample_ids else config.training.video_num
                        for preview_id in range(num_previews):
                            try:
                                validate_video_generation(model, val_dataset, config, global_step, config.training.output_dir, preview_id, accelerator)
                            except Exception as e:
                                # A rendering failure (e.g. a transient decode
                                # error) shouldn't kill an otherwise-healthy run.
                                logger.warning(f"⚠️ validation preview {preview_id} failed: {e}")
                    model.train()
    # Without this, an active wandb tracker (accelerator.init_trackers above, whenever wandb
    # is installed) never gets told to finish -- its background logging service keeps the
    # process alive after the loop exits, hanging indefinitely instead of returning control.
    # Timeout-guarded: see _end_training_with_timeout's docstring.
    _end_training_with_timeout(accelerator)


def _parse_overrides(pairs):
    """Parse `--override a.b=1 c.d=x` CLI args into {"a.b": "1", "c.d": "x"}."""
    overrides = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")  # split on the first "="; value stays a raw string
        overrides[key] = value
    return overrides


def cli():
    """Parse command-line arguments for `clap-train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a TrainingRunConfig YAML file.")
    # action="extend": a repeated --override MERGES into the same list instead of replacing
    # the earlier occurrence (argparse's default for a repeated flag) -- needed because
    # an "--override data.conditioning=... ... "$@"" pattern means
    # a caller appending their own "--override key=val ..." would otherwise silently wipe out
    # the script's own overrides (e.g. data.conditioning).
    parser.add_argument("--override", nargs="*", default=[], action="extend",
                         help="Dotted-key overrides, e.g. training.learning_rate=1e-4")
    return parser.parse_args()  # values are parsed/typed later by load_config, not here


def main_cli():
    """Entry point installed as the `clap-train` console script."""
    args = cli()
    config = load_config(args.config, overrides=_parse_overrides(args.override))
    paths = PathConfig()  # resolved from CLAP_* env vars, see clap.config.paths
    try:
        main(config, paths)
    except KeyboardInterrupt:
        # accelerate launch spawns one process per GPU under this same process group; killing
        # just this one leaves the rest running (dataloader workers, other ranks' training
        # loops, wandb's service process) instead of actually stopping the job.
        logger.info("⏹️  interrupted -- killing process group")
        os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
