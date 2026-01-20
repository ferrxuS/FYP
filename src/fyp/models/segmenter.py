import torch
import torch.nn as nn
from .backbone import DINOv3FeatureExtractor
from .decoder import SegmentationHead
from ..config import BACKBONE, NUM_CLASSES, DECODER_HIDDEN_DIM

class ContainerDefectSegmenter(nn.Module):
    """Complete segmentation model: DINOv3 + Decoder"""

    def __init__(
        self,
        backbone_name: str = BACKBONE,
        num_classes: int = NUM_CLASSES,
        freeze_backbone: bool = True,
        decoder_hidden_dim: int = DECODER_HIDDEN_DIM,
    ):
        super().__init__()

        self.backbone = DINOv3FeatureExtractor(
            model_name=backbone_name,
            freeze=freeze_backbone
        )

        # Segmentation head (decoder)
        self.head = SegmentationHead(
            self.backbone.get_feature_dim(),
            num_classes,
            decoder_hidden_dim
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)
