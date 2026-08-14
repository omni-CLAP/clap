from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RolloutReplayConfig:
    """Offline autoregressive replay against recorded episodes, for eval (FVD/FID/per-view metrics).

    Attributes:
        family: Which embodiment's replay loader to use — matches a key in
            `clap.data.EMBODIMENTS` (e.g. "ee", "lam", "language", "g1_humanoid").
        history_idx: Optional fixed history-frame offsets (e.g. `[0,0,-20,-16,-12,-8]`)
            relative to the anchor, overriding the dataset's usual random sampling
            so eval runs are reproducible frame-for-frame.
        max_chunks: Cap on how many autoregressive chunks to roll out per
            episode; None replays the full episode.
        gt_cond: If True, condition each chunk on ground-truth frames instead of
            the model's own previous prediction (open-loop vs. closed-loop eval).
    """

    family: str
    ckpt_path: str  # checkpoint to load CLAPModel from
    num_inference_steps: int = 50  # denoising steps per chunk
    guidance_scale: float = 1.0  # classifier-free guidance scale
    decode_chunk_size: int = 7  # VAE-decode this many frames at a time
    max_chunks: Optional[int] = None
    history_idx: Optional[List[int]] = None
    gt_cond: bool = False


@dataclass
class RolloutDeployConfig:
    """Real-robot policy-in-the-loop deployment configuration, one instance per task.

    Loaded from `configs/experiment/deploy_<task>.yaml` rather than hardcoded
    per-task branches — episode_ids/instructions/start_idx are task-specific
    data, not something that belongs in Python control flow.

    Attributes:
        val_dataset_dir: OXE-catalog-style base path (an `EEEpisodeLoader`'s
            `oxe_base_path`) holding `dataset_name`'s recorded episodes, used
            as the visual/joint-state starting point for each rollout — the
            robot doesn't need to replay these; only the seed frame +
            instruction condition the first interaction round.
        dataset_name: Which `clap.data.oxe_catalog` entry under `val_dataset_dir`
            to seed from (e.g. "droid").
        episode_ids: One entry per rollout episode. None auto-discovers every
            episode `EEEpisodeLoader` finds under val_dataset_dir/dataset_name
            (see `clap.rollout.deploy._resolve_episode_selection`) — convenient
            for a small sample dataset, but pin this explicitly for a real task
            so the episode set doesn't silently change if the data does.
        start_idx: Starting frame within the seed episode, aligned with
            episode_ids. None defaults every episode to 0.
        instructions: Language instruction per episode, aligned with
            episode_ids. None reads each episode's own recorded instruction
            (its annotation JSON's "texts" field) instead of a task-specific override.
        adapter_ckpt: Checkpoint for `EEVelocityToPositionAdapter`, mapping the
            openpi policy's joint-velocity output into joint positions.
        policy_server: `"host:port"` of a running policy server to connect to
            instead of loading the policy in this process (openpi's
            `scripts/serve_policy.py`, or MolmoAct2's `host_server_droid.py`/
            `host_server_yam.py`) — keeps that policy's dependencies out of
            clap's own environment. None (default) loads the policy in-process.
            `policy_ckpt` is ignored when this is set (the server already has
            its own checkpoint loaded).
        molmoact_norm_tag: Only meaningful for `policy_type="molmoact2"` +
            in-process mode — which normalization statistics the checkpoint
            expects (e.g. "franka_droid" for the DROID checkpoint). Server
            mode doesn't need this; the server already has its own norm_tag baked in.
        molmoact_include_wrist: Only meaningful for `policy_type="molmoact2"`
            + in-process mode — whether the checkpoint was trained with a
            wrist-camera input.
        pred_step: Actions predicted per policy call.
        policy_skip_step: Spacing between executed actions within a prediction
            (horizon = (pred_step - 1) * policy_skip_step). Used for every round unless
            policy_skip_step_schedule is set.
        policy_skip_step_schedule: Optional per-round override of policy_skip_step --
            round `i` (0-indexed) uses `policy_skip_step_schedule[min(i, len(schedule) - 1)]`,
            so a shorter schedule than `interact_num` just holds its last value for every
            remaining round (e.g. `[4, 4, 2]`: rounds 0-1 skip 4, round 2+ skip 2). None
            (default) uses the scalar `policy_skip_step` for every round.
        interact_num: Number of policy-call rounds per episode.
        history_idx: Optional fixed history-frame offsets (e.g. `[0,0,-12,-9,-6,-3]`)
            into the running latent buffer, overriding the default last-num_history
            contiguous window — same convention as `RolloutReplayConfig.history_idx`
            / `clap.rollout.teleop.TeleopSession`. None uses the contiguous window.
    """

    task_name: str
    val_dataset_dir: str
    dataset_name: str
    episode_ids: Optional[List[str]] = None  # None -> auto-discover, see class docstring
    start_idx: Optional[List[int]] = None  # None -> every episode starts at frame 0
    instructions: Optional[List[str]] = None  # None -> use each episode's own recorded instruction
    policy_type: str = "pi05"  # "pi05" | "pi0" | "pi0fast" — which openpi policy to load
    policy_ckpt: str = ""  # openpi/MolmoAct2 policy checkpoint path (in-process mode only)
    policy_server: Optional[str] = None  # "host:port" -> connect to a running server instead; see class docstring
    molmoact_norm_tag: str = "franka_droid"  # MolmoAct2 in-process only; see class docstring
    molmoact_include_wrist: bool = True  # MolmoAct2 in-process only; see class docstring
    adapter_ckpt: str = ""  # EEVelocityToPositionAdapter checkpoint path
    gripper_max: float = 0.75  # per-task gripper-open clamp
    z_min: float = 0.18  # per-task minimum end-effector height (safety clamp)
    pred_step: int = 5  # number of predicted frames
    policy_skip_step: int = 2
    policy_skip_step_schedule: Optional[List[int]] = None  # None -> policy_skip_step every round; see class docstring
    interact_num: int = 24
    history_idx: Optional[List[int]] = None  # None -> last num_history contiguous frames; see class docstring
