from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = False) -> None:
        super().__init__()

        if pretrained:
            weights = tv_models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        backbone = tv_models.resnet18(weights=weights)

        # All children: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        children = list(backbone.children())
        # We drop the final FC layer (children[-1])
        # We can also merge them back into a single sequential block:
        self.encoder = nn.Sequential(*children[:-1])
        self.output_dim = backbone.fc.in_features  # 512

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Standard forward
        feat = self.encoder(images)
        return feat.flatten(1)
