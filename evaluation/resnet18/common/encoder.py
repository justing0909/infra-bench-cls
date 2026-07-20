"""
ResNet-18/50 encoder backbone for the ResNet-18 baselines.

Used by both the supervised ResNet-18 and the Random Features (frozen
random-init) baselines. Wraps torchvision's ResNet with two adaptations
for multimodal Infra-Bench tiles:

- First-conv expansion from 3 to N input channels (default 9 for
  Sentinel-1 VV/VH + Sentinel-2 [B02, B03, B04, B08, B8A, B11, B12]).
- Final FC replaced with Identity so the encoder outputs feature
  vectors of `feature_dim` (512 for ResNet-18, 2048 for ResNet-50);
  the classification head is separate (`LinearClassifier` below).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, resnet50


class EncoderBackbone(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        if backbone_name == "resnet18":
            net = resnet18(weights="DEFAULT" if pretrained else None)
            self.feature_dim = net.fc.in_features
        elif backbone_name == "resnet50":
            net = resnet50(weights="DEFAULT" if pretrained else None)
            self.feature_dim = net.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False)

        net.fc = nn.Identity()   # match pretraining/model.py exactly
        self.encoder = net       # store as self.encoder, not Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)



class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
