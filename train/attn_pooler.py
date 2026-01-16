import torch
import torch.nn as nn
import torch.nn.functional as F


class MHAPooler(nn.Module):
    """MHA-based attentive pooling."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.query)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        """Attentive pooling over a sequence.

        Args:
            x:    Tensor of shape (B, T, D)
            mask: Optional bool/int tensor of shape (B, T) where 1/True means
                  valid tokens and 0/False means padding.
        Returns:
            pooled: (B, D) pooled representation
            attn_weights: attention weights from MultiheadAttention
        """
        B, T, D = x.shape
        x = F.layer_norm(x, (D,))
        
        q = self.query.expand(B, 1, -1)
        q = F.layer_norm(q, (D,))

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0

        out, attn_weights = self.attn(
            q, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )

        pooled = out.squeeze(1)  # (B, D)

        return pooled, attn_weights