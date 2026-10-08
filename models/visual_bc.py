"""Vision-and-proprioception behavioral cloning policy without language."""

import math

import torch
from torch import nn

from models.policy_components import ProprioEncoder, TemporalHistoryEncoder
from models.visual_encoder import ResNet18VisualEncoder


VISUAL_BC_FUSION_MODES = ("global", "spatial_attention")
POLICY_CAMERAS = ("agentview", "robot0_eye_in_hand")


class LearnedSpatialAttention(nn.Module):
    """Attend over visual cells using a task-level learned query."""

    def __init__(self, input_channels=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.output_dim = int(output_dim)
        self.visual_projection = nn.Sequential(
            nn.Conv2d(input_channels, output_dim, kernel_size=1),
            nn.GroupNorm(1, output_dim),
            nn.GELU(),
        )
        self.query = nn.Parameter(torch.empty(output_dim))
        nn.init.normal_(self.query, std=output_dim**-0.5)
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim + 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, spatial_feature):
        if spatial_feature.ndim != 4:
            raise ValueError("spatial_feature must have shape [B, C, H, W]")
        visual = self.visual_projection(spatial_feature)
        logits = torch.einsum("bchw,c->bhw", visual, self.query)
        logits = logits / math.sqrt(self.output_dim)
        batch, height, width = logits.shape
        attention = torch.softmax(logits.flatten(1), dim=-1).view(
            batch, height, width
        )
        attended = torch.einsum("bchw,bhw->bc", visual, attention)

        y = torch.linspace(-1.0, 1.0, height, device=visual.device, dtype=visual.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=visual.device, dtype=visual.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack(
            (
                torch.sum(attention * grid_x.unsqueeze(0), dim=(1, 2)),
                torch.sum(attention * grid_y.unsqueeze(0), dim=(1, 2)),
            ),
            dim=-1,
        )
        feature = self.output_projection(torch.cat((attended, coordinates), dim=-1))
        return feature, attention, coordinates


class VisualBCPolicy(nn.Module):
    """Predict one normalized action from camera images and robot state."""

    def __init__(
        self,
        visual_fusion="spatial_attention",
        camera_names=("agentview",),
        action_dim=7,
        proprio_dim=23,
        visual_dim=256,
        proprio_feature_dim=64,
        fusion_hidden_dims=(256, 128),
        pretrained_visual=True,
        freeze_visual_backbone=True,
        dropout=0.1,
    ):
        super().__init__()
        if visual_fusion not in VISUAL_BC_FUSION_MODES:
            raise ValueError(
                f"visual_fusion must be one of {VISUAL_BC_FUSION_MODES}, "
                f"got {visual_fusion!r}"
            )
        camera_names = tuple(camera_names)
        if not camera_names or camera_names[0] != "agentview":
            raise ValueError("camera_names must start with agentview")
        if len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names cannot contain duplicates")
        unknown_cameras = set(camera_names) - set(POLICY_CAMERAS)
        if unknown_cameras:
            raise ValueError(f"Unsupported policy cameras: {sorted(unknown_cameras)}")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        self.visual_fusion = visual_fusion
        self.camera_names = camera_names
        self.use_eye_in_hand = "robot0_eye_in_hand" in camera_names
        self.action_dim = int(action_dim)
        self.visual_encoder = ResNet18VisualEncoder(
            output_dim=visual_dim,
            pretrained=pretrained_visual,
            freeze_backbone=freeze_visual_backbone,
            dropout=dropout,
        )
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim,
            hidden_dim=proprio_feature_dim,
            output_dim=proprio_feature_dim,
        )

        if visual_fusion == "spatial_attention":
            for parameter in self.visual_encoder.projection.parameters():
                parameter.requires_grad_(False)
            self.spatial_attention = LearnedSpatialAttention(
                input_channels=512,
                output_dim=visual_dim,
                dropout=dropout,
            )
            self.eye_in_hand_spatial_attention = (
                LearnedSpatialAttention(
                    input_channels=512,
                    output_dim=visual_dim,
                    dropout=dropout,
                )
                if self.use_eye_in_hand
                else None
            )
        else:
            self.spatial_attention = None
            self.eye_in_hand_spatial_attention = None

        visual_feature_count = 2 if self.use_eye_in_hand else 1
        fusion_input_dim = visual_dim * visual_feature_count + proprio_feature_dim
        hidden_1, hidden_2 = fusion_hidden_dims
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(hidden_2, action_dim)

    def _encode_view(self, image, attention_module):
        if self.visual_fusion == "global":
            return self.visual_encoder(image), None, None
        return attention_module(self.visual_encoder.forward_spatial(image))

    def forward(self, image, proprio, *, eye_in_hand_image=None):
        if self.use_eye_in_hand and eye_in_hand_image is None:
            raise ValueError("Dual-camera policy requires eye_in_hand_image")
        if not self.use_eye_in_hand and eye_in_hand_image is not None:
            raise ValueError("Single-camera policy does not accept eye_in_hand_image")

        visual_feature, attention, coordinates = self._encode_view(
            image, self.spatial_attention
        )
        if self.use_eye_in_hand:
            eye_feature, eye_attention, eye_coordinates = self._encode_view(
                eye_in_hand_image, self.eye_in_hand_spatial_attention
            )
        else:
            eye_feature = eye_attention = eye_coordinates = None
        proprio_feature = self.proprio_encoder(proprio)

        features = [visual_feature]
        if eye_feature is not None:
            features.append(eye_feature)
        features.append(proprio_feature)
        if len({feature.shape[0] for feature in features}) != 1:
            raise ValueError("Vision and proprio batch sizes must match")
        fusion_feature = self.fusion(torch.cat(features, dim=-1))
        return {
            "action": self.action_head(fusion_feature),
            "visual_feature": visual_feature,
            "proprio_feature": proprio_feature,
            "fusion_feature": fusion_feature,
            "spatial_attention_weights": attention,
            "spatial_attention_coordinates": coordinates,
            "eye_in_hand_visual_feature": eye_feature,
            "eye_in_hand_attention_weights": eye_attention,
            "eye_in_hand_attention_coordinates": eye_coordinates,
        }


class HistoryVisualBCPolicy(VisualBCPolicy):
    """Visual-BC with a fixed proprioception and executed-action history."""

    def __init__(
        self,
        history_size=8,
        temporal_feature_dim=128,
        *args,
        **kwargs,
    ):
        if history_size <= 1:
            raise ValueError("history_size must be greater than one")
        action_dim = int(kwargs.get("action_dim", 7))
        proprio_dim = int(kwargs.get("proprio_dim", 23))
        proprio_feature_dim = int(kwargs.get("proprio_feature_dim", 64))
        super().__init__(*args, **kwargs)
        self.history_size = int(history_size)
        self.temporal_encoder = TemporalHistoryEncoder(
            history_size=history_size,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            output_dim=temporal_feature_dim,
            dropout=float(kwargs.get("dropout", 0.1)),
        )
        self.proprio_encoder = None

        visual_dim = self.visual_encoder.output_dim
        visual_feature_count = 2 if self.use_eye_in_hand else 1
        fusion_hidden_dims = kwargs.get("fusion_hidden_dims", (256, 128))
        hidden_1, hidden_2 = fusion_hidden_dims
        self.fusion = nn.Sequential(
            nn.Linear(
                visual_dim * visual_feature_count + temporal_feature_dim,
                hidden_1,
            ),
            nn.LayerNorm(hidden_1),
            nn.GELU(),
            nn.Dropout(float(kwargs.get("dropout", 0.1))),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),
            nn.Dropout(float(kwargs.get("dropout", 0.1))),
        )

    def forward(
        self,
        image,
        proprio_history,
        action_history,
        proprio_history_mask,
        action_history_mask,
        *,
        eye_in_hand_image=None,
    ):
        if self.use_eye_in_hand and eye_in_hand_image is None:
            raise ValueError("Dual-camera policy requires eye_in_hand_image")
        if not self.use_eye_in_hand and eye_in_hand_image is not None:
            raise ValueError("Single-camera policy does not accept eye_in_hand_image")

        visual_feature, attention, coordinates = self._encode_view(
            image, self.spatial_attention
        )
        if self.use_eye_in_hand:
            eye_feature, eye_attention, eye_coordinates = self._encode_view(
                eye_in_hand_image, self.eye_in_hand_spatial_attention
            )
        else:
            eye_feature = eye_attention = eye_coordinates = None
        temporal_feature = self.temporal_encoder(
            proprio_history,
            action_history,
            proprio_history_mask,
            action_history_mask,
        )
        features = [visual_feature]
        if eye_feature is not None:
            features.append(eye_feature)
        features.append(temporal_feature)
        if len({feature.shape[0] for feature in features}) != 1:
            raise ValueError("Vision and temporal history batch sizes must match")
        fusion_feature = self.fusion(torch.cat(features, dim=-1))
        return {
            "action": self.action_head(fusion_feature),
            "visual_feature": visual_feature,
            "temporal_feature": temporal_feature,
            "fusion_feature": fusion_feature,
            "spatial_attention_weights": attention,
            "spatial_attention_coordinates": coordinates,
            "eye_in_hand_visual_feature": eye_feature,
            "eye_in_hand_attention_weights": eye_attention,
            "eye_in_hand_attention_coordinates": eye_coordinates,
        }
