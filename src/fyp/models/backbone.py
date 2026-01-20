import torch
import torch.nn as nn
from transformers import AutoModel
from ..config import BACKBONE, PATCH_SIZE

class DINOv3FeatureExtractor(nn.Module):
    """Extracts dense features from DINOv3 backbone"""

    MODEL_REGISTRY = {
        "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "vitsp16": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "vithp16": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    }

    FEATURE_DIMS = {
        "vits16": 384,
        "vitsp16": 384,
        "vitb16": 768,
        "vitl16": 1024,
        "vithp16": 1280,
    }

    def __init__(
        self,
        model_name: str = BACKBONE,
        patch_size: int = PATCH_SIZE,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.patch_size = patch_size
        self.feature_dim = self.FEATURE_DIMS[model_name]

        # Load pre-trained model
        self.backbone = AutoModel.from_pretrained(self.MODEL_REGISTRY[model_name])

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        H_patches, W_patches = H // self.patch_size, W // self.patch_size

        with torch.set_grad_enabled(not self.training):
            outputs = self.backbone(
                x,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract spatial patch tokens (skip CLS + register tokens)
        last_layer = outputs.hidden_states[-1]
        num_patch_tokens = H_patches * W_patches
        patch_tokens = last_layer[:, 1:1+num_patch_tokens, :]

        # Reshape to spatial grid
        features = patch_tokens.reshape(B, H_patches, W_patches, self.feature_dim)
        return features.permute(0, 3, 1, 2)

    def get_feature_dim(self) -> int:
        return self.feature_dim
