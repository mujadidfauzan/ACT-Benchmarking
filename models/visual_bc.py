"""Single-action visual behavior cloning with independent camera encoders."""

import copy

import torch
from torch import nn

from models.visual_encoder import VisualEncoder


# Model architecture is configured in code, not command-line hyperparameters.
HIDDEN_DIMS = (256, 128)
TEMPORAL_HIDDEN_DIM = 256
GRIPPER_LOGIT_THRESHOLD = -0.5
ENCODER_CONFIG = {"conv_layers": [[32, 5, 2, 2], [64, 3, 2, 1], [128, 3, 2, 1], [128, 3, 2, 1]],
                  "pool_size": [4, 4], "feature_dim": 256}


class VisualBCPolicy(nn.Module):
    policy_type = "visual_bc"
    visual_fusion = "independent_cnn_concat"

    def __init__(self, proprio_dim, camera_names=("agentview",), predict_orientation=True,
                 *, encoder_config=None, hidden_dims=None,
                 gripper_logit_threshold=GRIPPER_LOGIT_THRESHOLD,
                 temporal=False, temporal_hidden_dim=TEMPORAL_HIDDEN_DIM):
        super().__init__()
        self.camera_names = tuple(camera_names)
        if self.camera_names not in (("agentview",), ("agentview", "robot0_eye_in_hand")):
            raise ValueError("Use agentview, optionally followed by robot0_eye_in_hand")
        self.proprio_dim = int(proprio_dim)
        self.action_dim = 7
        self.temporal = bool(temporal)
        self.temporal_hidden_dim = int(temporal_hidden_dim)
        self.predict_orientation = bool(predict_orientation)
        self.motion_dim = 6 if self.predict_orientation else 3
        self.gripper_logit_threshold = float(gripper_logit_threshold)
        default_hidden = HIDDEN_DIMS[-1:] if self.temporal else HIDDEN_DIMS
        self.hidden_dims = tuple(default_hidden if hidden_dims is None else hidden_dims)
        if self.proprio_dim <= 0 or self.temporal_hidden_dim <= 0 or not self.hidden_dims or min(self.hidden_dims) <= 0:
            raise ValueError("Network dimensions must be positive")
        # Camera encoders start identically but learn independent parameters.
        self.vision = VisualEncoder(**(ENCODER_CONFIG if encoder_config is None else encoder_config))
        self.eye_vision = copy.deepcopy(self.vision) if len(self.camera_names) == 2 else None
        self.fusion_input_dim = self.vision.output_dim * len(self.camera_names) + self.proprio_dim
        self.gru = nn.GRU(self.fusion_input_dim + 1, self.temporal_hidden_dim, batch_first=True) if self.temporal else None
        layers = []
        previous = self.temporal_hidden_dim if self.temporal else self.fusion_input_dim
        for width in self.hidden_dims:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, self.motion_dim + 1))
        self.head = nn.Sequential(*layers)

    def forward(self, image, proprio, *, eye_in_hand_image=None, previous_gripper=None, hidden=None):
        if proprio.ndim != (3 if self.temporal else 2) or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("Unexpected proprio shape for the policy mode")
        if (self.eye_vision is not None) != (eye_in_hand_image is not None):
            raise ValueError("Camera inputs do not match the checkpoint")
        prefix = proprio.shape[:-1]
        expected_rank = 5 if self.temporal else 4
        for view in (image, eye_in_hand_image):
            if view is not None and (view.ndim != expected_rank or view.shape[:-3] != prefix):
                raise ValueError("Camera and proprio batch/sequence dimensions must match")
        features = [self.vision(image.reshape(-1, *image.shape[-3:])).reshape(*prefix, -1)]
        if self.eye_vision is not None:
            features.append(self.eye_vision(eye_in_hand_image.reshape(-1, *eye_in_hand_image.shape[-3:])).reshape(*prefix, -1))
        features.append(proprio)
        if self.temporal:
            if previous_gripper is None or previous_gripper.shape != (*prefix, 1):
                raise ValueError("Temporal policy requires previous_gripper with shape [B, T, 1]")
            features.append(previous_gripper)
            fused, hidden = self.gru(torch.cat(features, dim=-1), hidden)
        else:
            if previous_gripper is not None or hidden is not None:
                raise ValueError("Frame policy does not accept recurrent inputs")
            fused = torch.cat(features, dim=-1)
        prediction = self.head(fused)
        motion, logit = prediction[..., :self.motion_dim], prediction[..., self.motion_dim:]
        rotation = motion[..., 3:6] if self.predict_orientation else torch.zeros_like(motion[..., :3])
        grip = torch.where(logit > self.gripper_logit_threshold, 1.0, -1.0)
        output = {"motion": motion, "gripper_logit": logit,
                  "action": torch.cat((motion[..., :3], rotation, grip), dim=-1)}
        if self.temporal:
            output["hidden"] = hidden
        return output

    def step(self, image, proprio, *, eye_in_hand_image=None, previous_gripper=-1.0, hidden=None):
        """Predict one action; callers keep hidden state only within an episode."""
        if not self.temporal:
            return self(image, proprio, eye_in_hand_image=eye_in_hand_image)
        previous = proprio.new_full((len(proprio), 1, 1), previous_gripper)
        output = self(image.unsqueeze(1), proprio.unsqueeze(1), previous_gripper=previous,
                      eye_in_hand_image=eye_in_hand_image.unsqueeze(1) if eye_in_hand_image is not None else None,
                      hidden=hidden)
        return {key: value if key == "hidden" else value[:, 0] for key, value in output.items()}

    def get_config(self):
        return {"policy_type": self.policy_type, "camera_names": list(self.camera_names),
                "proprio_dim": self.proprio_dim, "predict_orientation": self.predict_orientation,
                "encoder_config": self.vision.get_config(), "hidden_dims": list(self.hidden_dims),
                "temporal": self.temporal, "temporal_hidden_dim": self.temporal_hidden_dim,
                "gripper_logit_threshold": self.gripper_logit_threshold}

    @classmethod
    def from_config(cls, config):
        return cls(**{key: value for key, value in config.items() if key != "policy_type"})
