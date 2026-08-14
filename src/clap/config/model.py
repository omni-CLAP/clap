from dataclasses import dataclass
from typing import Optional


@dataclass
class CLAPModelConfig:
    """Architecture and conditioning configuration for `CLAPModel`.

    Attributes:
        conditioning: Selects both the dataset family (via `clap.data.EMBODIMENTS`)
            and the model's forward branch. `"language"` routes through CLIP-encoded
            per-step captions instead of `action_encoder`; every other value is a
            numeric action (EE-cartesian, LAM-latent, joint-space, ...).
        action_dim: Dimensionality of the raw action fed to `action_encoder`
            (e.g. 7 for EE, 32 for LAM latents, 26 for the G1 humanoid).
        num_history / num_frames: History and future context length; the model
            always expects `num_history + num_frames` latent frames per sample.
        use_action_adapter: Reconstructs `action_adapter` so a checkpoint trained
            with an embodiment-action -> LAM-latent adapter loads correctly
            (inference/eval path only).
        train_action_adapter: Freezes the UNet and trains only `action_adapter`
            against a frozen, LAM-pretrained backbone (cross-embodiment adaptation
            path distinct from the usual finetune_ckpt + reset_action_encoder flow).
        adapter_input_dim: Embodiment action dim the adapter maps *from*; only
            meaningful when one of the two adapter flags above is set.
    """

    svd_model_path: str  # HF repo id or local path for the base SVD pipeline weights
    clip_model_path: str  # HF repo id or local path for the CLIP text/tokenizer
    conditioning: str = "ee"
    action_dim: int = 7
    num_history: int = 6  # number of history frames
    num_frames: int = 5  # number of predicted frames (including the current frame in the first slot)
    text_cond: bool = True  # add a task-description CLIP embedding on top of the action embedding
    frame_level_cond: bool = True  # one conditioning token per frame vs. one token for the whole clip
    his_cond_zero: bool = False  # ablation: zero the image condition on history frames
    deep_action_encoder: bool = False  # 4 hidden layers instead of 1 in ActionEncoder's MLP
    motion_bucket_id: int = 127  # SVD micro-conditioning: higher = more motion
    fps: int = 7  # SVD micro-conditioning: frames-per-second the model was trained at

    use_action_adapter: bool = False
    train_action_adapter: bool = False
    adapter_arch: str = "mlp"  # "mlp" or "transformer", see clap.models.action_adapter
    adapter_input_dim: Optional[int] = None
    adapter_hidden_dim: int = 512
    adapter_num_layers: int = 3
    adapter_num_heads: int = 8  # only used when adapter_arch == "transformer"
    adapter_dropout: float = 0.0

    @property
    def num_total_frames(self) -> int:
        """Total latent frames per sample (history + future); models are built expecting this length."""
        return self.num_history + self.num_frames
