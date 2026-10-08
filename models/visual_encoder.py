"""Visual encoders for manipulation policies."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18VisualEncoder(nn.Module):
    """ImageNet-normalized ResNet18 with a 512-to-feature projection head."""

    def __init__(
        self,
        output_dim=256,
        pretrained=True,
        freeze_backbone=True,
        dropout=0.0,
    ):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.output_dim = int(output_dim)
        self.pretrained = bool(pretrained)
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self._backbone_frozen = False
        self._layer4_trainable = False
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self._backbone_frozen = True
        self._layer4_trainable = False
        self.backbone.eval()
        return self

    def unfreeze_layer4(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad_(True)
        self._backbone_frozen = False
        self._layer4_trainable = True
        if self.training:
            self.backbone.eval()
            self.backbone.layer4.train()
        return self

    def unfreeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(True)
        self._backbone_frozen = False
        self._layer4_trainable = False
        self.backbone.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        elif self._layer4_trainable:
            self.backbone.eval()
            if mode:
                self.backbone.layer4.train()
        return self

    def forward(self, image):
        spatial = self.forward_spatial(image)
        backbone_feature = torch.flatten(
            self.backbone.avgpool(spatial), start_dim=1
        )
        return self.projection(backbone_feature)

    def forward_spatial(self, image):
        """Return the final ResNet feature map before global pooling."""

        return self.backbone.layer4(self.forward_layer3(image))

    def forward_layer3(self, image):
        """Return the stride-16 ResNet feature map for spatial policies."""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if not image.is_floating_point():
            raise TypeError("image must use a floating-point dtype")
        normalized = (image - self.image_mean) / self.image_std
        feature = self.backbone.conv1(normalized)
        feature = self.backbone.bn1(feature)
        feature = self.backbone.relu(feature)
        feature = self.backbone.maxpool(feature)
        feature = self.backbone.layer1(feature)
        feature = self.backbone.layer2(feature)
        return self.backbone.layer3(feature)
