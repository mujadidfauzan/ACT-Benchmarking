"""Language-conditioned single-action behavioral cloning policy."""

import math

import torch
from torch import nn

from language.frozen_minilm import TrainableAttentionPooling
from models.visual_encoder import ResNet18VisualEncoder


LANGUAGE_MODES = ("pooled", "token_attention")
VISUAL_FUSION_MODES = ("global", "spatial_attention")
POLICY_CAMERAS = ("agentview", "robot0_eye_in_hand")


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


class PooledLanguageHead(nn.Module):
    def __init__(self, input_dim=384, output_dim=128, dropout=0.1):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Language head dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, language_embedding):
        if language_embedding.ndim != 2 or language_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"language_embedding must have shape [B, {self.input_dim}], "
                f"got {tuple(language_embedding.shape)}"
            )
        return {
            "feature": self.projection(language_embedding),
            "attention_weights": None,
        }


class LanguageConditionedSpatialAttention(nn.Module):
    """Attend over ResNet cells using language and retain target coordinates."""

    def __init__(
        self,
        input_channels=512,
        language_dim=128,
        output_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.visual_projection = nn.Sequential(
            nn.Conv2d(input_channels, output_dim, kernel_size=1),
            nn.GroupNorm(1, output_dim),
            nn.GELU(),
        )
        self.language_query = nn.Sequential(
            nn.Linear(language_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim + 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, spatial_feature, language_feature):
        if spatial_feature.ndim != 4:
            raise ValueError("spatial_feature must have shape [B, C, H, W]")
        if language_feature.ndim != 2:
            raise ValueError("language_feature must have shape [B, D]")
        if spatial_feature.shape[0] != language_feature.shape[0]:
            raise ValueError("Visual and language batch sizes must match")

        visual = self.visual_projection(spatial_feature)
        query = self.language_query(language_feature)
        logits = torch.einsum("bchw,bc->bhw", visual, query)
        logits = logits / math.sqrt(self.output_dim)
        batch, height, width = logits.shape
        attention = torch.softmax(logits.flatten(1), dim=-1).view(
            batch, height, width
        )
        attended = torch.einsum("bchw,bhw->bc", visual, attention)

        y = torch.linspace(-1.0, 1.0, height, device=visual.device, dtype=visual.dtype)
        x = torch.linspace(-1.0, 1.0, width, device=visual.device, dtype=visual.dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        expected_x = torch.sum(attention * grid_x.unsqueeze(0), dim=(1, 2))
        expected_y = torch.sum(attention * grid_y.unsqueeze(0), dim=(1, 2))
        coordinates = torch.stack((expected_x, expected_y), dim=-1)
        feature = self.output_projection(
            torch.cat((attended, coordinates), dim=-1)
        )
        return feature, attention, coordinates


class LanguageBCPolicy(nn.Module):
    """Predict one normalized 7D action from vision, state, and language."""

    def __init__(
        self,
        language_mode="pooled",
        visual_fusion="spatial_attention",
        camera_names=("agentview",),
        action_dim=7,
        proprio_dim=16,
        visual_dim=256,
        proprio_feature_dim=64,
        language_input_dim=384,
        language_feature_dim=128,
        fusion_hidden_dims=(256, 128),
        pretrained_visual=True,
        freeze_visual_backbone=True,
        dropout=0.1,
    ):
        super().__init__()
        if language_mode not in LANGUAGE_MODES:
            raise ValueError(
                f"language_mode must be one of {LANGUAGE_MODES}, "
                f"got {language_mode!r}"
            )
        if visual_fusion not in VISUAL_FUSION_MODES:
            raise ValueError(
                f"visual_fusion must be one of {VISUAL_FUSION_MODES}, "
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
        if len(fusion_hidden_dims) != 2 or min(fusion_hidden_dims) <= 0:
            raise ValueError("fusion_hidden_dims must contain two positive values")

        self.language_mode = language_mode
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
        if language_mode == "pooled":
            self.language_head = PooledLanguageHead(
                input_dim=language_input_dim,
                output_dim=language_feature_dim,
                dropout=dropout,
            )
        else:
            self.language_head = TrainableAttentionPooling(
                input_dim=language_input_dim,
                output_dim=language_feature_dim,
                dropout=dropout,
            )

        if visual_fusion == "spatial_attention":
            for parameter in self.visual_encoder.projection.parameters():
                parameter.requires_grad_(False)
            self.spatial_attention = LanguageConditionedSpatialAttention(
                input_channels=512,
                language_dim=language_feature_dim,
                output_dim=visual_dim,
                dropout=dropout,
            )
            self.eye_in_hand_spatial_attention = (
                LanguageConditionedSpatialAttention(
                    input_channels=512,
                    language_dim=language_feature_dim,
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
        fusion_input_dim = (
            visual_dim * visual_feature_count
            + proprio_feature_dim
            + language_feature_dim
        )
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

    def _encode_language(
        self,
        language_embedding=None,
        token_embeddings=None,
        attention_mask=None,
    ):
        if self.language_mode == "pooled":
            if language_embedding is None:
                raise ValueError("pooled mode requires language_embedding")
            if token_embeddings is not None or attention_mask is not None:
                raise ValueError("pooled mode does not accept token language inputs")
            return self.language_head(language_embedding)

        if token_embeddings is None or attention_mask is None:
            raise ValueError(
                "token_attention mode requires token_embeddings and attention_mask"
            )
        if language_embedding is not None:
            raise ValueError(
                "token_attention mode does not accept language_embedding"
            )
        return self.language_head(token_embeddings, attention_mask)

    def forward(
        self,
        image,
        proprio,
        *,
        eye_in_hand_image=None,
        language_embedding=None,
        token_embeddings=None,
        attention_mask=None,
    ):
        language_output = self._encode_language(
            language_embedding=language_embedding,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        language_feature = language_output["feature"]
        if self.use_eye_in_hand and eye_in_hand_image is None:
            raise ValueError("Dual-camera policy requires eye_in_hand_image")
        if not self.use_eye_in_hand and eye_in_hand_image is not None:
            raise ValueError("Single-camera policy does not accept eye_in_hand_image")
        if self.visual_fusion == "spatial_attention":
            spatial_feature = self.visual_encoder.forward_spatial(image)
            (
                visual_feature,
                spatial_attention_weights,
                spatial_attention_coordinates,
            ) = self.spatial_attention(spatial_feature, language_feature)
            if self.use_eye_in_hand:
                eye_spatial = self.visual_encoder.forward_spatial(
                    eye_in_hand_image
                )
                (
                    eye_in_hand_visual_feature,
                    eye_in_hand_attention_weights,
                    eye_in_hand_attention_coordinates,
                ) = self.eye_in_hand_spatial_attention(
                    eye_spatial, language_feature
                )
            else:
                eye_in_hand_visual_feature = None
                eye_in_hand_attention_weights = None
                eye_in_hand_attention_coordinates = None
        else:
            visual_feature = self.visual_encoder(image)
            spatial_attention_weights = None
            spatial_attention_coordinates = None
            eye_in_hand_visual_feature = (
                self.visual_encoder(eye_in_hand_image)
                if self.use_eye_in_hand
                else None
            )
            eye_in_hand_attention_weights = None
            eye_in_hand_attention_coordinates = None
        proprio_feature = self.proprio_encoder(proprio)

        batch_sizes = {
            visual_feature.shape[0],
            proprio_feature.shape[0],
            language_feature.shape[0],
        }
        if eye_in_hand_visual_feature is not None:
            batch_sizes.add(eye_in_hand_visual_feature.shape[0])
        if len(batch_sizes) != 1:
            raise ValueError("Vision, proprio, and language batch sizes must match")

        visual_features = [visual_feature]
        if eye_in_hand_visual_feature is not None:
            visual_features.append(eye_in_hand_visual_feature)
        fused = torch.cat(
            [*visual_features, proprio_feature, language_feature], dim=-1
        )
        fusion_feature = self.fusion(fused)
        normalized_action = self.action_head(fusion_feature)
        return {
            "action": normalized_action,
            "visual_feature": visual_feature,
            "proprio_feature": proprio_feature,
            "language_feature": language_feature,
            "fusion_feature": fusion_feature,
            "attention_weights": language_output["attention_weights"],
            "spatial_attention_weights": spatial_attention_weights,
            "spatial_attention_coordinates": spatial_attention_coordinates,
            "eye_in_hand_visual_feature": eye_in_hand_visual_feature,
            "eye_in_hand_attention_weights": eye_in_hand_attention_weights,
            "eye_in_hand_attention_coordinates": eye_in_hand_attention_coordinates,
        }
