import torch
import torch.nn as nn

class MHAPooler(nn.Module):
    """A minimal Transformer block: pre-norm MultiheadAttention + MLP with residuals."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x, mask=None):
        """
        x:    (B, T, D)
        mask: (B, T) with 1 for valid tokens, 0 for padding
        """
        B = x.size(0)
        q = self.query.expand(B, 1, -1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0

        out, attn_weights = self.attn(
            q, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )

        # out: (B, 1, D)
        return out.squeeze(1), attn_weights