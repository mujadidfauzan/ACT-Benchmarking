"""Spatial-softmax Visual-BC policy for fixed-target pick."""

import torch
from torch import nn

from models.visual_encoder import ResNet18VisualEncoder
from models.visual_bc import POLICY_CAMERAS


class SpatialSoftmax(nn.Module):
    """Convert channel-wise spatial activations into expected XY coordinates."""

    def forward(self, feature_map):
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape [B, C, H, W]")
        batch, channels, height, width = feature_map.shape
        weights = torch.softmax(feature_map.flatten(2), dim=-1)
        x = torch.linspace(
            -1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype
        )
        y = torch.linspace(
            -1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        expected_x = torch.sum(weights * grid_x.reshape(1, 1, -1), dim=-1)
        expected_y = torch.sum(weights * grid_y.reshape(1, 1, -1), dim=-1)
        coordinates = torch.stack((expected_x, expected_y), dim=-1)
        if coordinates.shape != (batch, channels, 2):
            raise AssertionError("Unexpected spatial-softmax coordinate shape")
        return coordinates.flatten(1), weights.view(batch, channels, height, width)


class CameraSpatialSoftmaxEncoder(nn.Module):
    """Learn camera-specific keypoint maps from frozen ResNet layer3 features."""

    def __init__(self, input_channels=256, spatial_channels=64):
        super().__init__()
        if min(input_channels, spatial_channels) <= 0:
            raise ValueError("Spatial encoder dimensions must be positive")
        self.spatial_channels = int(spatial_channels)
        self.adapter = nn.Sequential(
            nn.Conv2d(input_channels, spatial_channels, kernel_size=1),
            nn.GroupNorm(8, spatial_channels),
            nn.ReLU(),
        )
        self.spatial_softmax = SpatialSoftmax()

    @property
    def output_dim(self):
        return 2 * self.spatial_channels

    def forward(self, feature_map):
        adapted = self.adapter(feature_map)
        return self.spatial_softmax(adapted)


class SpatialSoftmaxVisualBCPolicy(nn.Module):
    """Two-view visual BC with layer3 spatial-softmax features and compact proprio."""

    policy_type = "spatial_softmax_visual_bc"
    visual_fusion = "spatial_softmax_layer3"

    def __init__(
        self,
        camera_names=("agentview",),
        action_dim=7,
        proprio_dim=9,
        spatial_channels=64,
        hidden_dim=1024,
        dropout=0.1,
        pretrained_visual=True,
        freeze_visual_backbone=True,
    ):
        super().__init__()
        camera_names = tuple(camera_names)
        if not camera_names or camera_names[0] != "agentview":
            raise ValueError("camera_names must start with agentview")
        if len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names cannot contain duplicates")
        unknown_cameras = set(camera_names) - set(POLICY_CAMERAS)
        if unknown_cameras:
            raise ValueError(f"Unsupported policy cameras: {sorted(unknown_cameras)}")
        if min(action_dim, proprio_dim, spatial_channels, hidden_dim) <= 0:
            raise ValueError("Policy dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.camera_names = camera_names
        self.use_eye_in_hand = "robot0_eye_in_hand" in camera_names
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.spatial_channels = int(spatial_channels)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.visual_encoder = ResNet18VisualEncoder(
            output_dim=256,
            pretrained=pretrained_visual,
            freeze_backbone=freeze_visual_backbone,
        )
        # This policy reads layer3 directly, so the regular global projection is unused.
        for parameter in self.visual_encoder.projection.parameters():
            parameter.requires_grad_(False)

        self.agentview_encoder = CameraSpatialSoftmaxEncoder(
            input_channels=256,
            spatial_channels=spatial_channels,
        )
        self.eye_in_hand_encoder = (
            CameraSpatialSoftmaxEncoder(
                input_channels=256,
                spatial_channels=spatial_channels,
            )
            if self.use_eye_in_hand
            else None
        )
        visual_feature_count = 2 if self.use_eye_in_hand else 1
        self.fusion_input_dim = (
            visual_feature_count * self.agentview_encoder.output_dim + proprio_dim
        )
        self.control_head = nn.Sequential(
            nn.Linear(self.fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def _encode_view(self, image, camera_encoder):
        return camera_encoder(self.visual_encoder.forward_layer3(image))

    def forward(self, image, proprio, *, eye_in_hand_image=None):
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"proprio must have shape [B, {self.proprio_dim}], "
                f"got {tuple(proprio.shape)}"
            )
        if self.use_eye_in_hand and eye_in_hand_image is None:
            raise ValueError("Dual-camera policy requires eye_in_hand_image")
        if not self.use_eye_in_hand and eye_in_hand_image is not None:
            raise ValueError("Single-camera policy does not accept eye_in_hand_image")

        agent_feature, agent_attention = self._encode_view(
            image, self.agentview_encoder
        )
        features = [agent_feature]
        if self.use_eye_in_hand:
            eye_feature, eye_attention = self._encode_view(
                eye_in_hand_image, self.eye_in_hand_encoder
            )
            features.append(eye_feature)
        else:
            eye_feature = eye_attention = None
        features.append(proprio)
        if len({feature.shape[0] for feature in features}) != 1:
            raise ValueError("Vision and proprio batch sizes must match")
        fusion_input = torch.cat(features, dim=-1)
        fusion_feature = self.control_head(fusion_input)
        return {
            "action": self.action_head(fusion_feature),
            "agentview_spatial_feature": agent_feature,
            "agentview_spatial_weights": agent_attention,
            "eye_in_hand_spatial_feature": eye_feature,
            "eye_in_hand_spatial_weights": eye_attention,
            "fusion_feature": fusion_feature,
        }
