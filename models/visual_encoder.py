"""Train-from-scratch visual encoder; architecture is configured here, not by CLI."""

import torch
from torch import nn


# Each convolution is (output channels, kernel size, stride, padding).
CONV_LAYERS = ((32, 5, 2, 2), (64, 3, 2, 1), (128, 3, 2, 1), (128, 3, 2, 1))
POOL_SIZE = (4, 4)
FEATURE_DIM = 256


class VisualEncoder(nn.Module):
    """Encode RGB in [0, 1] directly, without ImageNet preprocessing or weights."""

    def __init__(self, *, conv_layers=None, pool_size=None, feature_dim=None):
        super().__init__()
        self.conv_layers = tuple(tuple(layer) for layer in (
            CONV_LAYERS if conv_layers is None else conv_layers
        ))
        self.pool_size = tuple(POOL_SIZE if pool_size is None else pool_size)
        self.output_dim = FEATURE_DIM if feature_dim is None else int(feature_dim)
        if (not self.conv_layers or any(len(layer) != 4 for layer in self.conv_layers)
                or any(out <= 0 or kernel <= 0 or stride <= 0 or pad < 0
                       for out, kernel, stride, pad in self.conv_layers)):
            raise ValueError("Invalid CNN convolution specification")
        if len(self.pool_size) != 2 or min(self.pool_size) <= 0 or self.output_dim <= 0:
            raise ValueError("Pool and feature dimensions must be positive")
        layers = []
        channels = 3
        for output, kernel, stride, padding in self.conv_layers:
            layers.extend((nn.Conv2d(channels, output, kernel, stride, padding), nn.ReLU()))
            channels = output
        self.convolutions = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(self.pool_size), nn.Flatten(),
            nn.Linear(channels * self.pool_size[0] * self.pool_size[1], self.output_dim),
            nn.ReLU(),
        )

    def get_config(self):
        return {"conv_layers": [list(layer) for layer in self.conv_layers],
                "pool_size": list(self.pool_size), "feature_dim": self.output_dim}

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        if not torch.is_floating_point(image):
            raise ValueError("image must be floating-point RGB in [0, 1]")
        return self.projection(self.convolutions(image))
