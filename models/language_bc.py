"""Language-conditioned single-action behavioral cloning policy."""

import torch
from torch import nn

from language.frozen_minilm import TrainableAttentionPooling
from models.visual_encoder import ResNet18VisualEncoder


LANGUAGE_MODES = ("pooled", "token_attention")


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


class LanguageBCPolicy(nn.Module):
    """Predict one normalized 7D action from vision, state, and language."""

    def __init__(
        self,
        language_mode="pooled",
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
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if len(fusion_hidden_dims) != 2 or min(fusion_hidden_dims) <= 0:
            raise ValueError("fusion_hidden_dims must contain two positive values")

        self.language_mode = language_mode
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

        fusion_input_dim = visual_dim + proprio_feature_dim + language_feature_dim
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
        language_embedding=None,
        token_embeddings=None,
        attention_mask=None,
    ):
        visual_feature = self.visual_encoder(image)
        proprio_feature = self.proprio_encoder(proprio)
        language_output = self._encode_language(
            language_embedding=language_embedding,
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )
        language_feature = language_output["feature"]

        batch_sizes = {
            visual_feature.shape[0],
            proprio_feature.shape[0],
            language_feature.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError("Vision, proprio, and language batch sizes must match")

        fused = torch.cat(
            [visual_feature, proprio_feature, language_feature], dim=-1
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
        }
