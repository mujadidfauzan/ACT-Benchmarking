"""Shared neural-network components for manipulation policies."""

import torch
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


class TemporalHistoryEncoder(nn.Module):
    """Encode a fixed window of normalized proprioception and past actions."""

    def __init__(
        self,
        history_size,
        proprio_dim,
        action_dim,
        hidden_dim=256,
        output_dim=128,
        dropout=0.1,
    ):
        super().__init__()
        if history_size <= 1:
            raise ValueError("history_size must be greater than one")
        if min(proprio_dim, action_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("Temporal encoder dimensions must be positive")
        self.history_size = int(history_size)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        input_dim = (
            history_size * proprio_dim
            + (history_size - 1) * action_dim
            + history_size
            + history_size
            - 1
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        proprio_history,
        action_history,
        proprio_history_mask,
        action_history_mask,
    ):
        batch = proprio_history.shape[0]
        expected_proprio = (batch, self.history_size, self.proprio_dim)
        expected_action = (batch, self.history_size - 1, self.action_dim)
        if tuple(proprio_history.shape) != expected_proprio:
            raise ValueError(
                f"proprio_history must have shape {expected_proprio}, "
                f"got {tuple(proprio_history.shape)}"
            )
        if tuple(action_history.shape) != expected_action:
            raise ValueError(
                f"action_history must have shape {expected_action}, "
                f"got {tuple(action_history.shape)}"
            )
        if tuple(proprio_history_mask.shape) != (batch, self.history_size):
            raise ValueError("Invalid proprio_history_mask shape")
        if tuple(action_history_mask.shape) != (batch, self.history_size - 1):
            raise ValueError("Invalid action_history_mask shape")

        proprio_mask = proprio_history_mask.to(proprio_history.dtype)
        action_mask = action_history_mask.to(action_history.dtype)
        proprio_history = proprio_history * proprio_mask.unsqueeze(-1)
        action_history = action_history * action_mask.unsqueeze(-1)
        flattened = torch.cat(
            (
                proprio_history.flatten(1),
                action_history.flatten(1),
                proprio_mask,
                action_mask,
            ),
            dim=-1,
        )
        return self.network(flattened)
