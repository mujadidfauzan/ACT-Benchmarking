"""Normalization utilities shared by training and policy inference."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


class PolicyNormalizer(nn.Module):
    """Z-score proprioception and actions using train-split statistics."""

    def __init__(
        self,
        proprio_mean,
        proprio_std,
        action_mean,
        action_std,
        *,
        proprio_keys=None,
        source_path=None,
        split_sha256=None,
        epsilon=1e-6,
    ):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        proprio_mean = self._as_vector(proprio_mean, "proprio_mean")
        proprio_std = self._as_vector(proprio_std, "proprio_std")
        action_mean = self._as_vector(action_mean, "action_mean")
        action_std = self._as_vector(action_std, "action_std")
        if proprio_mean.shape != proprio_std.shape:
            raise ValueError("Proprio mean and std dimensions do not match")
        if action_mean.shape != action_std.shape:
            raise ValueError("Action mean and std dimensions do not match")
        if torch.any(proprio_std <= 0):
            raise ValueError("Proprio std must be strictly positive")
        if torch.any(action_std <= 0):
            raise ValueError("Action std must be strictly positive")

        self.register_buffer("proprio_mean", proprio_mean)
        self.register_buffer("proprio_std", proprio_std.clamp_min(epsilon))
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std.clamp_min(epsilon))
        self.source_path = str(source_path) if source_path is not None else None
        self.proprio_keys = tuple(proprio_keys or ())
        if self.proprio_keys and len(self.proprio_keys) > self.proprio_dim:
            raise ValueError("More proprio keys than scalar proprio dimensions")
        self.split_sha256 = split_sha256
        self.epsilon = float(epsilon)

    @staticmethod
    def _as_vector(values, name):
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.numel() == 0:
            raise ValueError(f"{name} must be a non-empty rank-1 vector")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
        return tensor

    @classmethod
    def from_json(cls, path, epsilon=1e-6):
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            statistics = json.load(handle)
        try:
            proprio = statistics["task_balanced_proprio"]
            action = statistics["task_balanced_action"]
            proprio_mean = proprio["mean"]
            proprio_std = proprio["std"]
            action_mean = action["mean"]
            action_std = action["std"]
            proprio_keys = proprio["keys"]
        except KeyError as error:
            raise ValueError(
                f"{path} does not contain task-balanced normalization statistics"
            ) from error
        return cls(
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            action_mean=action_mean,
            action_std=action_std,
            proprio_keys=proprio_keys,
            source_path=path.resolve(),
            split_sha256=statistics.get("split_sha256"),
            epsilon=epsilon,
        )

    @property
    def proprio_dim(self):
        return int(self.proprio_mean.numel())

    @property
    def action_dim(self):
        return int(self.action_mean.numel())

    @staticmethod
    def _validate_input(value, feature_dim, name):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if value.ndim == 0 or value.shape[-1] != feature_dim:
            raise ValueError(
                f"{name} must end with dimension {feature_dim}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")

    def normalize_proprio(self, proprio):
        self._validate_input(proprio, self.proprio_dim, "proprio")
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_action(self, action):
        self._validate_input(action, self.action_dim, "action")
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, normalized_action):
        self._validate_input(
            normalized_action, self.action_dim, "normalized_action"
        )
        return normalized_action * self.action_std + self.action_mean

    def validate_split(self, split_manifest):
        """Raise when a split file differs from the statistics source split."""

        if self.split_sha256 is None:
            raise ValueError("Normalization file does not record split_sha256")
        split_manifest = Path(split_manifest)
        digest = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
        if digest != self.split_sha256:
            raise ValueError(
                "Split manifest does not match the split used for normalization: "
                f"{split_manifest}"
            )
        return True
