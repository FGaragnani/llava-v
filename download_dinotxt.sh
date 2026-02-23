import os

# Torch cache
os.environ["TORCH_HOME"] = "/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"

# Hugging Face cache
os.environ["HF_HOME"] = "/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"
os.environ["HF_HUB_CACHE"] = "/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"
os.environ["TRANSFORMERS_CACHE"] = "/leonardo_scratch/large/userexternal/fgaragna/models/lmsys"

import torch

dinov2_vitl14_reg4_dinotxt_tet1280d20h24l = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
)