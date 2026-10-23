"""Train-from-scratch CNN or ResNet18 with configurable spatial pooling."""

import math

import torch
from torch import nn


# Each convolution is (output channels, kernel size, stride, padding).
CONV_LAYERS = ((32, 5, 2, 2), (64, 3, 2, 1), (128, 3, 2, 1), (128, 3, 2, 1))
POOL_SIZE = (4, 4)
FEATURE_DIM = 256


class SpatialSoftmax(nn.Module):
    """Return expected (x, y) coordinates in [-1, 1] for each feature channel."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Spatial-softmax temperature must be finite and positive")

    def forward(self, features):
        height, width = features.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=features.device, dtype=features.dtype),
            torch.linspace(-1, 1, width, device=features.device, dtype=features.dtype),
            indexing="ij",
        )
        attention = (features.flatten(2) / self.temperature).softmax(dim=-1)
        expected_x = (attention * x.flatten()).sum(dim=-1)
        expected_y = (attention * y.flatten()).sum(dim=-1)
        return torch.stack((expected_x, expected_y), dim=-1).flatten(1)


class VisualEncoder(nn.Module):
    """Encode RGB in [0, 1] directly, without ImageNet preprocessing or weights."""

    def __init__(self, *, conv_layers=None, pool_size=None, feature_dim=None,
                 pooling="avg", spatial_temperature=1.0, backbone="cnn"):
        super().__init__()
        self.conv_layers = tuple(tuple(layer) for layer in (
            CONV_LAYERS if conv_layers is None else conv_layers
        ))
        self.pool_size = tuple(POOL_SIZE if pool_size is None else pool_size)
        self.output_dim = FEATURE_DIM if feature_dim is None else int(feature_dim)
        self.backbone = backbone
        if self.backbone not in ("cnn", "resnet18"):
            raise ValueError("Backbone must be cnn or resnet18")
        self.pooling = pooling
        self.spatial_temperature = float(spatial_temperature)
        if self.pooling not in ("avg", "spatial_softmax"):
            raise ValueError("Pooling must be avg or spatial_softmax")
        if (not self.conv_layers or any(len(layer) != 4 for layer in self.conv_layers)
                or any(out <= 0 or kernel <= 0 or stride <= 0 or pad < 0
                       for out, kernel, stride, pad in self.conv_layers)):
            raise ValueError("Invalid CNN convolution specification")
        if len(self.pool_size) != 2 or min(self.pool_size) <= 0 or self.output_dim <= 0:
            raise ValueError("Pool and feature dimensions must be positive")
        if self.backbone == "resnet18":
            # Import only when selected so existing CNN checkpoints stay lightweight.
            try:
                from torchvision.models import resnet18
            except ImportError as error:
                raise ImportError("ResNet18 requires torchvision compatible with the installed torch build") from error
            network = resnet18(weights=None)
            channels = network.fc.in_features
            self.convolutions = nn.Sequential(*list(network.children())[:-2])
        else:
            layers = []
            channels = 3
            for output, kernel, stride, padding in self.conv_layers:
                layers.extend((nn.Conv2d(channels, output, kernel, stride, padding), nn.ReLU()))
                channels = output
            self.convolutions = nn.Sequential(*layers)
        if self.pooling == "spatial_softmax":
            pool = SpatialSoftmax(self.spatial_temperature)
            pooled_dim = 2 * channels
        else:
            pool = nn.AdaptiveAvgPool2d(self.pool_size)
            pooled_dim = channels * self.pool_size[0] * self.pool_size[1]
        self.projection = nn.Sequential(
            pool, nn.Flatten(),
            nn.Linear(pooled_dim, self.output_dim),
            nn.ReLU(),
        )

    def get_config(self):
        config = {"conv_layers": [list(layer) for layer in self.conv_layers],
                  "pool_size": list(self.pool_size), "feature_dim": self.output_dim}
        if self.backbone != "cnn":
            config.pop("conv_layers")
            config["backbone"] = self.backbone
        # Keep the legacy average-pooling config and state-dict layout unchanged.
        if self.pooling == "spatial_softmax":
            config.update(pooling=self.pooling, spatial_temperature=self.spatial_temperature)
        return config

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if not torch.is_floating_point(image):
            raise ValueError("image must be floating-point RGB in [0, 1]")
        return self.projection(self.convolutions(image))
