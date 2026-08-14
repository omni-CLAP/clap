from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Training-loop configuration: optimization, checkpointing, and periodic validation.

    Attributes:
        finetune_ckpt: Stage-2 finetune source (e.g.adaptation to a new embodiment). When set, only weights load
            (shape-mismatched keys dropped) and training starts at step 0 —
            distinct from auto-resume, which restores full training state
            from `output_dir`'s own last checkpoint.
        reset_action_encoder: Drop the entire `action_encoder.*` substate before
            a finetune load. Needed for cross-action_dim transfers (e.g. EE 7-d to EE 14-d) — partially transferring only shape-matched layers
            leaves the encoder stuck in the source embodiment's feature basin.
        use_8bit_optimizer: Use `bitsandbytes.optim.AdamW8bit` instead of
            `torch.optim.AdamW`, trading a little precision for lower optimizer-
            state memory (useful for larger inputs, e.g. 4-camera G1 humanoid).
        video_num / num_inference_steps / decode_chunk_size / guidance_scale:
            Control the periodic video-generation preview logged during
            training, not the training loss itself.
    """

    output_dir: str  # checkpoints/logs for this run are written here
    tag: str  # short run name, used to derive output_dir/wandb_run_name by convention
    learning_rate: float = 1e-5
    train_batch_size: int = 24  # per-device batch size
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"  # "no" | "fp16" | "bf16", passed to accelerate
    shuffle: bool = True  # shuffle the training dataloader each epoch
    max_train_steps: int = 500_000
    checkpointing_steps: int = 20_000  # save a checkpoint every N steps
    validation_steps: int = 2_500  # run the video-generation preview every N steps
    max_grad_norm: float = 1.0  # gradient-clipping threshold
    use_8bit_optimizer: bool = False

    finetune_ckpt: Optional[str] = None
    reset_action_encoder: bool = False
    strict_resume: bool = True  # error (vs. warn) on any resume-time state_dict mismatch
    load_training_state: bool = True  # restore optimizer/scheduler/step count, not just weights

    video_num: int = 4  # number of validation clips rendered per preview
    num_inference_steps: int = 50  # denoising steps for the preview rollout
    decode_chunk_size: int = 7  # VAE-decode this many frames at a time during preview
    guidance_scale: float = 1.0  # classifier-free guidance scale for the preview rollout

    wandb_project_name: str = "clap"
    wandb_run_name: Optional[str] = None  # defaults to `tag` if left unset
