"""Native observation adapter around the pinned LeRobot ACT neural network."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.act.temporal_ensemble import ChunkState
from .vendor.core import ACT, ACTION, OBS_IMAGES, OBS_STATE


def reference_config(proprio_dim, camera_names, chunk_size, hidden_dim, num_heads,
                     encoder_layers, decoder_layers, feedforward_dim, latent_dim, dropout):
    feature = lambda shape: SimpleNamespace(shape=shape)
    return SimpleNamespace(action_feature=feature((7,)), robot_state_feature=feature((proprio_dim,)),
        env_state_feature=None, image_features={name: feature((3,)) for name in camera_names},
        use_vae=True, dim_model=hidden_dim, n_heads=num_heads, n_encoder_layers=encoder_layers,
        n_vae_encoder_layers=encoder_layers, n_decoder_layers=decoder_layers,
        dim_feedforward=feedforward_dim, latent_dim=latent_dim, dropout=dropout,
        chunk_size=chunk_size, vision_backbone="resnet18", pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=False, pre_norm=False, feedforward_activation="relu")


class LeRobotACTPolicy(nn.Module):
    policy_type = "act_lerobot"
    temporal = True
    uses_previous_gripper = False
    action_dim = 7

    def __init__(self, proprio_dim=16, camera_names=("agentview", "robot0_eye_in_hand"), *,
                 chunk_size=32, hidden_dim=512, num_heads=8, encoder_layers=4,
                 decoder_layers=1, feedforward_dim=3200, latent_dim=32, dropout=.1,
                 image_size=128, predict_orientation=True, execution_steps=1,
                 temporal_ensemble=False, ensemble_decay=.01):
        super().__init__()
        self.camera_names = tuple(camera_names)
        if self.camera_names not in (("agentview",), ("agentview", "robot0_eye_in_hand")):
            raise ValueError("Use agentview, optionally robot0_eye_in_hand")
        dimensions = (proprio_dim, chunk_size, hidden_dim, num_heads, encoder_layers,
                      decoder_layers, feedforward_dim, latent_dim, image_size)
        if min(dimensions) <= 0 or hidden_dim % 4 or hidden_dim % num_heads:
            raise ValueError("Positive dimensions required; hidden_dim must divide into heads and 4")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        for key, value in zip(("proprio_dim", "chunk_size", "hidden_dim", "num_heads", "encoder_layers",
                               "decoder_layers", "feedforward_dim", "latent_dim", "image_size"), dimensions):
            setattr(self, key, int(value))
        self.dropout, self.predict_orientation = float(dropout), bool(predict_orientation)
        config = reference_config(proprio_dim, camera_names, chunk_size, hidden_dim, num_heads,
                                  encoder_layers, decoder_layers, feedforward_dim, latent_dim, dropout)
        self.core = ACT(config)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        self.configure_inference(execution_steps, temporal_ensemble, ensemble_decay)

    @property
    def backbone(self):
        return self.core.backbone

    def load_backbone_weights(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Expected a torchvision ResNet18 state_dict")
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        self.backbone.load_state_dict(state, strict=True)

    def preprocess_images(self, image, eye_in_hand_image=None):
        views = {"agentview": image}
        if eye_in_hand_image is not None:
            views["robot0_eye_in_hand"] = eye_in_hand_image
        if set(views) != set(self.camera_names):
            raise ValueError("Camera inputs differ from checkpoint")
        result = []
        for name in self.camera_names:
            frame = views[name]
            if frame.ndim != 4 or frame.shape[1] != 3:
                raise ValueError("Expected current RGB [B,3,H,W], no history dimension")
            if frame.dtype == torch.uint8:
                frame = frame.float() / 255.
            elif not frame.is_floating_point() or not torch.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
                raise ValueError("Floating RGB must be finite and in [0,1]")
            if frame.shape[-2:] != (self.image_size, self.image_size):
                frame = F.interpolate(frame, (self.image_size, self.image_size), mode="bilinear",
                                      align_corners=False, antialias=True)
            result.append((frame - self.image_mean) / self.image_std)
        return result

    def forward(self, image, proprio, *, eye_in_hand_image=None, actions=None, is_pad=None,
                sample_posterior=True):
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("Expected current normalized robot state [B,proprio_dim]")
        batch = {OBS_IMAGES: self.preprocess_images(image, eye_in_hand_image), OBS_STATE: proprio}
        # LeRobot uses the posterior only in train mode. Validation and rollout
        # both use z=0; the adapter intentionally preserves this behavior.
        if self.training:
            if actions is None or actions.shape != (len(proprio), self.chunk_size, 7) or is_pad is None or is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
                raise ValueError("Training requires actions [B,K,7] and bool is_pad [B,K]")
            batch.update({ACTION: actions.masked_fill(is_pad[..., None], 0), "action_is_pad": is_pad})
        action, (mu, logvar) = self.core(batch)
        if not self.predict_orientation:
            action = torch.cat((action[..., :3], torch.zeros_like(action[..., 3:6]), action[..., 6:]), -1)
        return dict(action=action, mu=mu, logvar=logvar)

    def configure_inference(self, execution_steps=1, temporal_ensemble=False, ensemble_decay=.01):
        if not 1 <= execution_steps <= self.chunk_size:
            raise ValueError("execution_steps must be between 1 and chunk_size")
        if temporal_ensemble and execution_steps != 1:
            raise ValueError("Temporal ensemble requires execution_steps=1")
        if not math.isfinite(ensemble_decay) or ensemble_decay < 0:
            raise ValueError("ensemble_decay must be finite and nonnegative")
        self.execution_steps, self.temporal_ensemble, self.ensemble_decay = int(execution_steps), bool(temporal_ensemble), float(ensemble_decay)

    def step(self, image, proprio, *, eye_in_hand_image=None, hidden=None, previous_gripper=None):
        if self.training:
            raise RuntimeError("Call eval() before rollout")
        state = ChunkState() if hidden is None else hidden
        query = self.temporal_ensemble or state.time % self.execution_steps == 0
        chunk = None
        if query:
            chunk = self(image, proprio, eye_in_hand_image=eye_in_hand_image)["action"]
            if not self.temporal_ensemble:
                state.predictions.clear()
            state.add(chunk)
        offset = state.time - state.predictions[-1][0]
        # LeRobot weights the oldest prediction most strongly; keep that
        # convention explicit rather than using legacy ChunkState's age rule.
        if self.temporal_ensemble:
            predictions = [(s, c) for s, c in state.predictions if state.time < s + c.shape[1]]
            values = torch.stack([c[:, state.time-s] for s, c in predictions])
            rank = values.new_tensor([s for s, _ in predictions])
            weights = torch.softmax(-self.ensemble_decay * (rank-rank.min()), 0)
            action = (values * weights[:, None, None]).sum(0)
            state.predictions = predictions
            state.time += 1
        else:
            action = state.current()
        return dict(action=action, hidden=state, action_chunk=chunk,
                    chunk_offset=offset, chunk_queried=query)

    def get_config(self):
        keys = ("proprio_dim", "chunk_size", "hidden_dim", "num_heads", "encoder_layers", "decoder_layers",
                "feedforward_dim", "latent_dim", "dropout", "image_size", "predict_orientation",
                "execution_steps", "temporal_ensemble", "ensemble_decay")
        return dict(camera_names=list(self.camera_names), **{k: getattr(self, k) for k in keys})

    @classmethod
    def from_config(cls, config):
        return cls(**config)
