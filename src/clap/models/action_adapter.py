"""Maps an embodiment's own action representation into the frozen LAM-latent space."""

import torch
import torch.nn as nn


def _zero_init_(module):
    """Zero a Linear layer's weight/bias so the adapter starts by outputting all zeros."""
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ActionAdapterMLP(nn.Module):
    """Per-frame MLP; final layer is zero-initialized to start at the unconditional direction.

    Args:
        in_dim: Dimensionality of the embodiment's own raw action representation.
        out_dim: Dimensionality of the target (LAM-latent) action space.
        num_layers: Total Linear layers including the zero-initialized head;
            must be >= 2 (at least one hidden block plus the head).
        dropout: Applied only inside hidden blocks, not on the head input.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=512, num_layers=3, dropout=0.0):
        super().__init__()
        assert num_layers >= 2, "need at least input + output layer"
        # Build num_layers-1 hidden Linear+SiLU blocks; the final projection (head)
        # is separate so it can be zero-initialized independently of the trunk.
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 2):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, out_dim)
        _zero_init_(self.head)

    def forward(self, action):
        # action: (B, T, in_dim) -> (B, T, out_dim), applied independently per frame.
        h = self.trunk(action)
        return self.head(h)


class ActionAdapterTransformer(nn.Module):
    """Temporal transformer adapter for when a single-frame delta lacks enough context.

    Args:
        max_seq_len: Upper bound on clip length; the learned positional
            embedding is sliced to the actual sequence length at forward time.
        num_heads: Attention heads per `TransformerEncoderLayer`.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=64,
        dropout=0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # Learned (not sinusoidal) positional embedding, sliced to the actual seq
        # len at forward time; max_seq_len just bounds how long a clip can be.
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)
        _zero_init_(self.head)

    def forward(self, action):
        # action: (B, T, in_dim) -> (B, T, out_dim); full self-attention across
        # T lets each frame's output see the whole window, unlike the MLP adapter.
        B, T, _ = action.shape
        assert T <= self.pos_emb.shape[1], (
            f"seq len {T} > adapter max_seq_len {self.pos_emb.shape[1]}"
        )
        h = self.in_proj(action) + self.pos_emb[:, :T]  # (B, T, hidden_dim), + sliced positional embedding
        h = self.encoder(h)  # full self-attention across T
        h = self.norm(h)
        return self.head(h)  # (B, T, out_dim), zero-initialized so it starts at the unconditional direction


def build_action_adapter(
    arch,
    in_dim,
    out_dim,
    hidden_dim=512,
    num_layers=3,
    num_heads=8,
    max_seq_len=64,
    dropout=0.0,
):
    """Construct an `ActionAdapterMLP` or `ActionAdapterTransformer` by name (`arch`).

    Args:
        arch: `"mlp"` or `"transformer"`; selects which adapter class is built.
        max_seq_len: Only meaningful for `"transformer"` (ignored by the MLP).
    """
    if arch == "mlp":
        # Per-frame MLP: no cross-frame context, cheaper.
        return ActionAdapterMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if arch == "transformer":
        # Temporal self-attention adapter: each frame's output can see the whole window.
        return ActionAdapterTransformer(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    raise ValueError(f"Unknown adapter arch: {arch}")
