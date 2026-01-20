import torch
import torch.nn as nn
from ..config import NUM_CLASSES, DECODER_HIDDEN_DIM, DROPOUT

class SegmentationHead(nn.Module):
    """Upsamples features to pixel-wise predictions"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = DECODER_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        # 48→96→192→384→768 progressive upsampling
        self.layer1 = self._make_layer(in_channels, hidden_dim, dropout)
        self.layer2 = self._make_layer(hidden_dim, hidden_dim // 2, dropout)
        self.layer3 = self._make_layer(hidden_dim // 2, hidden_dim // 4, dropout)
        self.layer4 = self._make_layer(hidden_dim // 4, hidden_dim // 8, dropout)
        self.classifier = nn.Conv2d(hidden_dim // 8, num_classes, kernel_size=1)

    def _make_layer(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.layer1(features)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.classifier(x)
