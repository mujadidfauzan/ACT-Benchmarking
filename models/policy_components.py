"""Shared neural-network components for manipulation policies."""

from torch import nn


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=64):
        super().__init__()
        if min(input_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Proprio encoder dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, proprio):
        if proprio.ndim != 2 or proprio.shape[-1] != self.input_dim:
            raise ValueError(
                f"proprio must have shape [B, {self.input_dim}], "
                f"got {tuple(proprio.shape)}"
            )
        return self.network(proprio)
