"""Causal frame-stack BC with per-camera spatial softmax."""

import torch
from torch import nn

from models.spatial_softmax_visual_bc import SpatialSoftmaxVisualBCPolicy


class HistorySpatialSoftmaxVisualBCPolicy(SpatialSoftmaxVisualBCPolicy):
    policy_type = "history_spatial_softmax_visual_bc"

    def __init__(self, history_size=8, rgb_history=True, **kwargs):
        if history_size < 2:
            raise ValueError("history_size must be at least two")
        super().__init__(**kwargs)
        self.history_size = int(history_size)
        self.rgb_history = bool(rgb_history)
        visual_dim = len(self.camera_names) * self.agentview_encoder.output_dim
        self.fusion_input_dim = (
            visual_dim * (self.history_size if self.rgb_history else 1)
            + self.history_size * self.proprio_dim
            + (self.history_size - 1) * self.action_dim
            + self.history_size + self.history_size - 1
        )
        self.control_head[0] = nn.Linear(self.fusion_input_dim, self.hidden_dim)

    def _encode_history(self, frames, mask, encoder):
        batch, times = mask.shape
        if frames.ndim != 5 or frames.shape[:3] != (batch, times, 3):
            raise ValueError("RGB history must have shape [B, H, 3, height, width]")
        # Exclude padding from the backbone and normalize each real frame independently.
        flat_mask = mask.reshape(-1)
        real_frames = frames.flatten(0, 1)[flat_mask]
        features, weights = self._encode_view(real_frames, encoder)
        padded = features.new_zeros((batch * times, encoder.output_dim))
        padded[flat_mask] = features
        padded_weights = weights.new_zeros((batch * times, *weights.shape[1:]))
        padded_weights[flat_mask] = weights
        return padded.reshape(batch, times, -1), padded_weights.reshape(
            batch, times, *weights.shape[1:]
        )

    def forward(
        self, proprio_history, action_history, proprio_history_mask,
        action_history_mask, *, image_history=None,
        eye_in_hand_image_history=None, image=None, eye_in_hand_image=None,
    ):
        batch = proprio_history.shape[0]
        h = self.history_size
        expected = (
            (proprio_history, (batch, h, self.proprio_dim)),
            (action_history, (batch, h - 1, self.action_dim)),
            (proprio_history_mask, (batch, h)),
            (action_history_mask, (batch, h - 1)),
        )
        for tensor, shape in expected:
            if tensor.shape != shape:
                raise ValueError(f"History shape {tuple(tensor.shape)} != {shape}")
        if proprio_history_mask.dtype != torch.bool or action_history_mask.dtype != torch.bool:
            raise ValueError("History masks must be boolean (True means valid)")
        if not proprio_history_mask[:, -1].all():
            raise ValueError("Current observation must be valid")
        if self.rgb_history:
            if image_history is None or image is not None or eye_in_hand_image is not None:
                raise ValueError("RGB-history policy requires image_history, not current images")
            agent, agent_weights = self._encode_history(
                image_history, proprio_history_mask, self.agentview_encoder
            )
            if self.use_eye_in_hand:
                if eye_in_hand_image_history is None:
                    raise ValueError("Dual-camera policy requires eye_in_hand_image_history")
                eye, eye_weights = self._encode_history(
                    eye_in_hand_image_history, proprio_history_mask, self.eye_in_hand_encoder
                )
            else:
                if eye_in_hand_image_history is not None:
                    raise ValueError("Single-camera policy does not accept wrist RGB history")
                eye = eye_weights = None
        else:
            if image is None or image_history is not None or eye_in_hand_image_history is not None:
                raise ValueError("State-history policy requires current images only")
            agent, agent_weights = self._encode_view(image, self.agentview_encoder)
            if self.use_eye_in_hand:
                if eye_in_hand_image is None:
                    raise ValueError("Dual-camera policy requires eye_in_hand_image")
                eye, eye_weights = self._encode_view(eye_in_hand_image, self.eye_in_hand_encoder)
            else:
                if eye_in_hand_image is not None:
                    raise ValueError("Single-camera policy does not accept wrist image")
                eye = eye_weights = None
        features = [agent.flatten(1)]
        if eye is not None:
            features.append(eye.flatten(1))
        features.extend([
            proprio_history.masked_fill(~proprio_history_mask.unsqueeze(-1), 0).flatten(1),
            action_history.masked_fill(~action_history_mask.unsqueeze(-1), 0).flatten(1),
            proprio_history_mask.to(proprio_history.dtype),
            action_history_mask.to(proprio_history.dtype),
        ])
        fusion = self.control_head(torch.cat(features, dim=-1))
        return {
            "action": self.action_head(fusion),
            "agentview_spatial_feature": agent,
            "agentview_spatial_weights": agent_weights,
            "eye_in_hand_spatial_feature": eye,
            "eye_in_hand_spatial_weights": eye_weights,
            "fusion_feature": fusion,
        }
