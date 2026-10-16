"""Current-frame visual BC with one shared, fully trainable Small CNN."""

import torch
from torch import nn

from models.small_cnn_visual_encoder import SmallCNNVisualEncoder
from models.visual_bc import POLICY_CAMERAS

# Change width/depth here for new experiments. Checkpoints retain their architecture.
CONTROL_HIDDEN_DIMS = (512, 256)


class SmallCNNVisualBCPolicy(nn.Module):
    policy_type = "small_cnn_visual_bc"
    visual_fusion = "small_cnn_concat"

    def __init__(
        self,
        camera_names=("agentview",),
        action_dim=7,
        proprio_dim=9,
        dropout=0.1,
        *,
        encoder_config=None,
        hidden_dims=None,
    ):
        super().__init__()
        self.camera_names = tuple(camera_names)
        if (
            not self.camera_names
            or self.camera_names[0] != "agentview"
            or len(set(self.camera_names)) != len(self.camera_names)
            or set(self.camera_names) - set(POLICY_CAMERAS)
        ):
            raise ValueError("Use unique supported cameras, starting with agentview")
        self.use_eye_in_hand = "robot0_eye_in_hand" in self.camera_names
        self.action_dim, self.proprio_dim = int(action_dim), int(proprio_dim)
        self.dropout = float(dropout)
        self.hidden_dims = tuple(
            CONTROL_HIDDEN_DIMS if hidden_dims is None else hidden_dims
        )
        if (
            min(self.action_dim, self.proprio_dim) <= 0
            or not self.hidden_dims
            or min(self.hidden_dims) <= 0
        ):
            raise ValueError("Policy dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.visual_encoder = SmallCNNVisualEncoder(**(encoder_config or {}))
        self.fusion_input_dim = (
            len(self.camera_names) * self.visual_encoder.output_dim + self.proprio_dim
        )
        layers = []
        previous = self.fusion_input_dim
        for width in self.hidden_dims:
            layers.extend(
                (
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                )
            )
            previous = width
        self.control_head = nn.Sequential(*layers)
        self.action_head = nn.Linear(previous, self.action_dim)

    def get_config(self):
        return {
            "policy_type": self.policy_type,
            "visual_fusion": self.visual_fusion,
            "visual_encoder": "small_cnn",
            "camera_names": list(self.camera_names),
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "dropout": self.dropout,
            "hidden_dims": list(self.hidden_dims),
            "encoder_config": self.visual_encoder.get_config(),
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            **{
                key: config[key]
                for key in (
                    "camera_names",
                    "action_dim",
                    "proprio_dim",
                    "dropout",
                    "hidden_dims",
                    "encoder_config",
                )
            }
        )

    def forward(self, image, proprio, *, eye_in_hand_image=None):
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must have shape [B, {self.proprio_dim}]")
        if self.use_eye_in_hand != (eye_in_hand_image is not None):
            raise ValueError("eye_in_hand_image must match the configured cameras")
        agent_feature = self.visual_encoder(image)
        eye_feature = (
            self.visual_encoder(eye_in_hand_image) if self.use_eye_in_hand else None
        )
        features = (
            [agent_feature]
            + ([eye_feature] if self.use_eye_in_hand else [])
            + [proprio]
        )
        if len({feature.shape[0] for feature in features}) != 1:
            raise ValueError("Vision and proprio batch sizes must match")
        fused = self.control_head(torch.cat(features, dim=-1))
        return {
            "action": self.action_head(fused),
            "fusion_feature": fused,
            "agentview_feature": agent_feature,
            "eye_in_hand_feature": eye_feature,
        }
