"""Image BC-RNN recipe, implemented locally for this repository's datasets.

References: ARISE-Initiative/robomimic, generate_paper_configs.py (image MG
BC-RNN), models/{base_nets,obs_core,policy_nets}.py and algo/bc.py.
No robomimic dependency, pretrained downloads, or legacy BC normalization.
"""

import torch
from torch import nn
from torch.nn import functional as F

from models.visual_encoder import SpatialSoftmax


PROPRIO_KEYS = ("obs__robot0_eef_pos", "obs__robot0_eef_quat", "obs__robot0_gripper_qpos")


class ImagePreprocessor(nn.Module):
    """RGB [0,1], resize to 84, one random crop per frame (center at eval)."""

    def __init__(self, image_size=84, crop_size=76):
        super().__init__()
        if not 0 < crop_size <= image_size:
            raise ValueError("Need 0 < crop size <= image size")
        self.image_size, self.crop_size = int(image_size), int(crop_size)

    def forward(self, frames):
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Expected NCHW RGB frames")
        frames = frames.float().div(255) if frames.dtype == torch.uint8 else frames
        if not frames.is_floating_point():
            raise ValueError("Images must be uint8 RGB or floating RGB in [0,1]")
        if frames.shape[-2:] != (self.image_size, self.image_size):
            frames = F.interpolate(frames, (self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False, antialias=True)
        margin = self.image_size - self.crop_size
        if self.training:
            offsets = torch.randint(margin + 1, (len(frames), 2), device=frames.device)
        else:
            offsets = torch.full((len(frames), 2), margin // 2, device=frames.device)
        # Vectorized indexing avoids a CPU synchronization for each GPU crop.
        row = offsets[:, 0, None, None] + torch.arange(self.crop_size, device=frames.device)[None, :, None]
        col = offsets[:, 1, None, None] + torch.arange(self.crop_size, device=frames.device)[None, None, :]
        return frames.permute(0, 2, 3, 1)[torch.arange(len(frames), device=frames.device)[:, None, None], row, col].permute(0, 3, 1, 2).contiguous()


class KeypointEncoder(nn.Module):
    def __init__(self, num_keypoints=32, feature_dim=64, image_size=84, crop_size=76):
        super().__init__()
        from torchvision.models import resnet18

        self.preprocess = ImagePreprocessor(image_size, crop_size)
        self.backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.keypoints = nn.Conv2d(512, num_keypoints, 1)
        self.pool = SpatialSoftmax(temperature=1.)
        # VisualCore's linear projection is followed by ObservationEncoder's ReLU.
        self.projection = nn.Linear(2 * num_keypoints, feature_dim)

    def forward(self, frames):
        return F.relu(self.projection(self.pool(self.keypoints(self.backbone(self.preprocess(frames))))))


class BCRNNPolicy(nn.Module):
    policy_type = "robomimic_bc_rnn"
    temporal = True
    action_dim = 7
    proprio_dim = 9
    uses_previous_gripper = False

    def __init__(self, camera_names=("agentview", "robot0_eye_in_hand"), *,
                 hidden_dim=1000, num_layers=2, sequence_length=10,
                 num_keypoints=32, feature_dim=64, image_size=84, crop_size=76,
                 predict_orientation=True):
        super().__init__()
        self.camera_names = tuple(camera_names)
        if self.camera_names not in (("agentview",), ("agentview", "robot0_eye_in_hand")):
            raise ValueError("Use agentview, optionally followed by robot0_eye_in_hand")
        if min(hidden_dim, num_layers, sequence_length, num_keypoints, feature_dim) <= 0:
            raise ValueError("Network and sequence dimensions must be positive")
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.sequence_length = int(sequence_length)
        self.num_keypoints, self.feature_dim = int(num_keypoints), int(feature_dim)
        self.image_size, self.crop_size = int(image_size), int(crop_size)
        self.predict_orientation = bool(predict_orientation)
        self.encoders = nn.ModuleDict({name: KeypointEncoder(num_keypoints, feature_dim, image_size, crop_size)
                                       for name in self.camera_names})
        self.lstm = nn.LSTM(len(self.camera_names) * feature_dim + self.proprio_dim,
                            hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, image, proprio, *, eye_in_hand_image=None, hidden=None):
        if proprio.ndim != 3 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("BC-RNN requires proprio [B,T,9] in EEF position/quaternion/gripper order")
        views = {"agentview": image}
        if eye_in_hand_image is not None:
            views["robot0_eye_in_hand"] = eye_in_hand_image
        if set(views) != set(self.camera_names):
            raise ValueError("Camera inputs differ from the checkpoint")
        prefix = proprio.shape[:2]
        features = []
        for name in self.camera_names:
            view = views[name]
            if view.ndim != 5 or view.shape[:2] != prefix:
                raise ValueError("Images must match proprio batch/sequence dimensions")
            features.append(self.encoders[name](view.flatten(0, 1)).reshape(*prefix, self.feature_dim))
        features.append(proprio)
        encoded, state = self.lstm(torch.cat(features, -1), hidden)
        return {"action": self.head(encoded).tanh(), "hidden": state}

    def step(self, image, proprio, *, eye_in_hand_image=None, hidden=None, previous_gripper=None):
        """One inference step with the reference periodic LSTM reset.

        State is explicit so each episode owns both LSTM memory and its counter.
        previous_gripper is accepted only for the shared simulator interface;
        it is never passed to the network.
        """
        if self.training:
            raise RuntimeError("Call eval() before rollout")
        state, count = (None, 0) if hidden is None else hidden
        reset = count % self.sequence_length == 0
        if reset:
            state = None
        result = self(image.unsqueeze(1), proprio.unsqueeze(1),
                      eye_in_hand_image=eye_in_hand_image.unsqueeze(1) if eye_in_hand_image is not None else None,
                      hidden=state)
        return {"action": result["action"][:, 0], "hidden": (result["hidden"], count + 1),
                "hidden_state_reset": reset}

    def get_config(self):
        return dict(camera_names=list(self.camera_names), hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers, sequence_length=self.sequence_length,
                    num_keypoints=self.num_keypoints, feature_dim=self.feature_dim,
                    image_size=self.image_size, crop_size=self.crop_size,
                    predict_orientation=self.predict_orientation)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RawPolicyInputs:
    """Shared evaluator interface: robot proprio and controller actions stay raw."""

    proprio_keys = PROPRIO_KEYS

    @staticmethod
    def normalize_proprio(value):
        return value

    @staticmethod
    def denormalize_action(value):
        return value
