import einops
import torch
import torch.nn as nn


def clip_encode_strings(strings, tokenizer, text_encoder):
    """Encode a batch of strings with a frozen CLIP text tower.

    Returns:
        (B, projection_dim) text embeddings, e.g. used for both the per-frame
        language-caption path and the task-description conditioning.
    """
    with torch.no_grad():  # text_encoder is always frozen
        inputs = tokenizer(strings, padding="max_length", return_tensors="pt", truncation=True).to(text_encoder.device)
        outputs = text_encoder(**inputs)
        return outputs.text_embeds  # pooled CLIP projection, not per-token hidden states


class ActionEncoder(nn.Module):
    """Per-frame MLP mapping a numeric action to the UNet's cross-attention hidden dim.

    Args:
        action_dim: Dimensionality of one frame's raw action.
        action_num: Number of frames in a sample (history + future); only used
            when `frame_level_cond=False`, where all frames are flattened into
            one token instead of encoded per-frame.
        text_cond: If True and `texts` is passed to `forward`, adds a CLIP task
            embedding to every frame's action embedding.
        deep: Widens the MLP to 4 hidden layers instead of 1.
    """

    def __init__(self, action_dim, action_num, hidden_size, text_cond=True, deep=False):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size
        self.text_cond = text_cond

        n_hidden = 4 if deep else 1
        # input projection -> n_hidden x (Linear+SiLU) -> final linear (no activation)
        layers = [nn.Linear(action_dim, hidden_size), nn.SiLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_size, hidden_size), nn.SiLU()]
        layers.append(nn.Linear(hidden_size, hidden_size))
        self.action_encode = nn.Sequential(*layers)
        # Kaiming init (not the framework default) since every layer feeds a SiLU.
        for layer in self.action_encode:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, action, texts=None, tokenizer=None, text_encoder=None, frame_level_cond=True):
        """Encode raw actions into cross-attention conditioning.

        Args:
            texts: Optional task-description strings; only used if `self.text_cond`
                is True (requires `tokenizer`/`text_encoder` to also be passed).
            frame_level_cond: If False, all T frames are flattened into a single
                token instead of one token per frame (see `action_num`).

        Returns:
            (B, T, hidden_size) if `frame_level_cond`, else (B, 1, hidden_size).
        """
        # action: (B, T, action_dim)
        if not frame_level_cond:
            # Collapse all frames into one token instead of encoding each frame separately.
            action = einops.rearrange(action, "b t d -> b 1 (t d)")  # (B, 1, T*action_dim)
        action = self.action_encode(action)  # (B, T, hidden_size) or (B, 1, hidden_size)

        if texts is not None and self.text_cond:
            # Add a shared task-description embedding on top of the per-frame action embedding.
            task_emb = clip_encode_strings(texts, tokenizer, text_encoder)  # (B, clip_dim)
            task_emb = einops.repeat(task_emb, "b c -> b 1 (n c)", n=2)  # (B, 1, hidden_size)
            action = action + task_emb
        return action
