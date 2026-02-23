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
        self.model_name = model_name
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            if "clip" in model_name.lower():
                from transformers import CLIPVisionModel
                print(f"[PatchEmbedder] Loading CLIP vision encoder only (skipping text encoder)")
                self.vision_model = CLIPVisionModel.from_pretrained(model_name)
                self.model = self.vision_model  # Reference for config access
            elif "siglip" in model_name.lower():
                from transformers import SiglipVisionModel
                print(f"[PatchEmbedder] Loading SigLip vision encoder only (skipping text encoder)")
                self.vision_model = SiglipVisionModel.from_pretrained(model_name)
                self.model = self.vision_model  # Reference for config access
            else:
                # DINOv2 and other vision-only models - load normally
                self.model = AutoModel.from_pretrained(model_name)
                self.vision_model = self.model
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            raise e
        
        self.model.eval()
        self.vision_model.eval()
        self.device = torch.device(device)
        self.agg_mode = agg_mode
        
        # Handle different model config structures
        if hasattr(self.model.config, 'hidden_size'):
            self.dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'vision_config') and hasattr(self.model.config.vision_config, 'hidden_size'):
            self.dim = self.model.config.vision_config.hidden_size
        elif hasattr(self.model.config, 'projection_dim'):
            self.dim = self.model.config.projection_dim
        else:
            raise ValueError(f"Cannot determine hidden dimension from model config. Available attributes: {dir(self.model.config)}")
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
        if hasattr(self, 'vision_model') and self.vision_model is not self.model:
            self.vision_model.to(device)
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
            outputs = self.vision_model(**inputs, output_hidden_states=True)
        except RuntimeError as e:
            print("RuntimeError:", e)
            print("Input tensor shape:", inputs["pixel_values"].shape)
            print("Input dtype: ", inputs["pixel_values"].dtype)
            print("Model device: ", next(self.vision_model.parameters()).device)
            print("Sharded?", any(hasattr(p, "ds_id") for p in self.vision_model.parameters()))
            raise

        # Hidden states -> last layer embeddings
        last_hidden = self._get_last_tokens(outputs)  # [N, num_tokens, D]
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
    
    @torch.no_grad()
    def forward_tokens(self, imgs):
        """
        imgs: [N, 3, H, W] tensor or list of PIL images
        returns:
            patch_tokens: [N, P, D]
            patch_grid: (H_p, W_p) patch grid inferred from resize + patch size
            patch_size: int patch stride (pixels)
        """
        original_device = imgs.device if isinstance(imgs, torch.Tensor) else None
        inputs = self.processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.vision_model(**inputs, output_hidden_states=True)

        last_hidden = self._get_last_tokens(outputs)  # [N, num_tokens, D]
        patch_tokens = last_hidden[:, 1:, :]     # [N, num_patches, D]

        patch_size = self._get_model_patch_size()
        h_px, w_px = inputs["pixel_values"].shape[-2:]
        if patch_size is not None:
            grid_h = max(1, h_px // patch_size)
            grid_w = max(1, w_px // patch_size)
        else:
            raise ValueError(f"Cannot infer patch grid without patch size from model [{self.model}] config.")
        patch_grid = (grid_h, grid_w)

        if original_device is not None:
            patch_tokens = patch_tokens.to(original_device)
        return patch_tokens, patch_grid, patch_size
    
    # given the set of output embeddings from forward_tokens,
    # decide which to aggregate into a single embedding given crop location
    @torch.no_grad()
    def aggregated_embeddings_from_crop(self, patch_tokens, patch_grid, patch_size, bboxes, orig_size):
        """
        patch_tokens: Tensor [P, D] or [1, P, D] from forward_tokens (single image).
        patch_grid: (H_p, W_p) grid returned by forward_tokens.
        patch_size: int patch stride (pixels) used by the vision backbone.
        bboxes: list of (l, t, r, b) in original pixel coords.
        orig_size: (H_orig, W_orig) of the original image.
        returns: Tensor [K, D] mean embedding per bbox (order follows bboxes).
        """
        token_sets = PatchEmbedder.token_sets_from_bboxes(
            patch_tokens=patch_tokens,
            patch_grid=patch_grid,
            patch_size=patch_size,
            bboxes=bboxes,
            orig_size=orig_size,
        )

        if not token_sets:
            return torch.empty(0, patch_tokens.shape[-1], device=patch_tokens.device)

        agg_list = []
        for token_set in token_sets:
            if token_set.numel() == 0:
                agg_list.append(torch.zeros(patch_tokens.shape[-1], device=patch_tokens.device))
            else:
                agg_list.append(token_set.mean(dim=0))

        return torch.stack(agg_list, dim=0)

    @staticmethod
    @torch.no_grad()
    def token_sets_from_bboxes(patch_tokens, patch_grid, patch_size, bboxes, orig_size):
        """
        patch_tokens: Tensor [P, D] or [1, P, D] from forward_tokens (single image).
        patch_grid: (H_p, W_p) grid returned by forward_tokens.
        patch_size: int patch stride (pixels) used by the vision backbone.
        bboxes: list of (l, t, r, b) in original pixel coords.
        orig_size: (H_orig, W_orig) of the original image.
        returns: list of Tensors [N_i, D] per bbox (order follows bboxes).
        """
        if patch_tokens.dim() == 3 and patch_tokens.size(0) == 1:
            patch_tokens = patch_tokens.squeeze(0)
        if patch_tokens.dim() != 2:
            raise ValueError(f"Expected patch_tokens shape [P, D] or [1, P, D], got {patch_tokens.shape}")

        grid_h, grid_w = patch_grid
        num_patches = patch_tokens.shape[0]
        if num_patches != grid_h * grid_w:
            raise ValueError(f"Patch/token mismatch: tokens={num_patches}, grid={grid_h}x{grid_w}")

        # Reshape tokens into grid
        tokens_grid = patch_tokens.view(grid_h, grid_w, -1)
        H_orig, W_orig = orig_size

        token_sets = []
        for (l, t, r, b) in bboxes:
            # Scale bbox from original coords to resized pixel coords used by the backbone.
            x0 = max(0.0, float(l) * grid_w * patch_size / max(1.0, W_orig))
            y0 = max(0.0, float(t) * grid_h * patch_size / max(1.0, H_orig))
            x1 = max(0.0, float(r) * grid_w * patch_size / max(1.0, W_orig))
            y1 = max(0.0, float(b) * grid_h * patch_size / max(1.0, H_orig))

            c0 = int(math.floor(x0 / patch_size))
            c1 = int(math.ceil(x1 / patch_size))
            r0 = int(math.floor(y0 / patch_size))
            r1 = int(math.ceil(y1 / patch_size))

            c0 = max(0, min(grid_w, c0))
            c1 = max(0, min(grid_w, c1))
            r0 = max(0, min(grid_h, r0))
            r1 = max(0, min(grid_h, r1))

            if c1 <= c0 or r1 <= r0:
                token_sets.append(torch.empty(0, patch_tokens.shape[-1], device=patch_tokens.device))
                continue

            region = tokens_grid[r0:r1, c0:c1, :]
            token_sets.append(region.reshape(-1, region.shape[-1]))

        return token_sets

    def _get_model_patch_size(self):
        if hasattr(self.model.config, "patch_size"):
            ps = self.model.config.patch_size
            return ps
        return None
    
    def _get_model_image_size(self):
        try:
            return self.model.config.image_size
        except AttributeError:
            return None
        
    def _get_last_tokens(self, outputs):
        if "clip" in self.model_name.lower():
            return outputs.hidden_states[-2]
        elif "siglip" in self.model_name.lower():
            return outputs.last_hidden_state
        else:
            return outputs.last_hidden_state