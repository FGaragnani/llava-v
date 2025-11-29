import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

class TransformerBlock(nn.Module):
    """A minimal Transformer block: pre-norm MultiheadAttention + MLP with residuals."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [N, S, D]
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        # Feed-forward
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class PatchEmbedder(nn.Module):
    """
        Embed image patches using a pretrained model and aggregate or return tokens.
        Intended to serve as a lightweight, frozen teacher for VRA.

        agg_mode:
          - "cls": return CLS token [B, D]
          - "mean": mean over patch tokens [B, D]
          - "max": max over patch tokens [B, D]
          - "attn": learn a small attention block, return first token [B, D]
          - "tokens": return patch tokens [B, P, D] (no aggregation)
    """
    def __init__(self, model_name="facebook/dinov2-base", agg_mode="mean", device="cuda"):
        super().__init__()
        """
        model_name: name of the pretrained model from HuggingFace
        agg_mode: aggregation mode for patch embeddings. One of ["cls", "mean", "max", "attn"]
        device: device to run the model on
        """
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            model_name = "/work/tesi_fgaragnani/checkpoints/facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(device)
        self.agg_mode = agg_mode
        self.dim = self.model.config.hidden_size
        if self.agg_mode == "attn":
            self.attn_block = TransformerBlock(self.dim, num_heads=8, mlp_ratio=4.0)
        else:
            self.attn_block = None
        # Move model to specified device
        self.to_device(self.device)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.attn_block is not None:
            for param in self.attn_block.parameters():
                param.requires_grad = False

    def to_device(self, device):
        self.model.to(device)
        if self.attn_block is not None:
            self.attn_block.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, patches):
        """
        patches: [N, 3, H, W] tensor or list of PIL images
        returns: [N, D] aggregated patch embeddings
        """
        # Preprocess and move to device
        original_device = patches.device if isinstance(patches, torch.Tensor) else None
        inputs = self.processor(images=patches, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            outputs = self.model(**inputs, output_hidden_states=True)
        except RuntimeError as e:
            print("RuntimeError:", e)
            print("Input tensor shape:", inputs['pixel_values'].shape)
            print("Input dtype: ", inputs['pixel_values'].dtype)
            print("Model device: ", next(self.model.parameters()).device)
            print("Sharded?", any(hasattr(p, "ds_id") for p in self.model.parameters()))

        # Hidden states -> last layer embeddings
        last_hidden = outputs.last_hidden_state  # [N, num_tokens, D]
        cls_token = last_hidden[:, 0, :]         # [N, D]
        patch_tokens = last_hidden[:, 1:, :]     # [N, num_patches, D]

        if self.agg_mode == "cls":
            agg = cls_token
        elif self.agg_mode == "mean":
            agg = patch_tokens.mean(dim=1)
        elif self.agg_mode == "max":
            agg = patch_tokens.max(dim=1).values
        elif self.agg_mode == "attn":
            # tokens: [N, 1 + num_patches, D]
            tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
            if self.attn_block is None:
                raise RuntimeError("attn_block is not initialized")
            with torch.enable_grad():
                out_tokens = self.attn_block(tokens)
            agg = out_tokens[:, 0, :]
        else:
            raise ValueError(f"Unknown agg_mode: {self.agg_mode}")

        if original_device is not None:
            agg = agg.to(original_device)
        return agg # [N, D]
    
    def _get_model_image_size(self):
        try:
            return self.model.config.image_size
        except AttributeError:
            return None