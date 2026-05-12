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


class SimCLRModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet18", pretrained_backbone: bool = False,
                 projection_dim: int = 128, in_channels: int = 3):
        super().__init__()
        if backbone_name == "resnet18":
            net = resnet18(weights="DEFAULT" if pretrained_backbone else None)
            feat_dim = net.fc.in_features
        elif backbone_name == "resnet50":
            net = resnet50(weights="DEFAULT" if pretrained_backbone else None)
            feat_dim = net.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        if in_channels != 3:
            old = net.conv1
            net.conv1 = nn.Conv2d(in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False)

        net.fc = nn.Identity()
        self.backbone = net          # directly, no EncoderBackbone wrapper
        self.feature_dim = feat_dim  # expose for LinearClassifier
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        z = self.projector(features)
        z = nn.functional.normalize(z, dim=1)
        return features, z


class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
