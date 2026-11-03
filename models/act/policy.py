"""ACT: spatial visual tokens, DETR-style Transformer, and action CVAE.

Reference: https://github.com/tonyzhaozh/act (MIT).
Adaptations: native 7D OSC actions, configurable robot proprioception, no
automatic weight downloads, and padding-normalized reconstruction loss.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .temporal_ensemble import ChunkState


def sinusoidal_positions(length, dim, device, dtype):
    position = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.) / dim))
    result = torch.zeros(length, dim, device=device, dtype=dtype)
    result[:, 0::2], result[:, 1::2] = (position * frequency).sin(), (position * frequency).cos()
    return result


class VisualBackbone(nn.Module):
    """Shared ResNet18 feature map; frozen BN statistics as in original ACT."""

    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.network = resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
        self.network.avgpool = nn.Identity()
        self.network.fc = nn.Identity()

    def forward(self, image):
        net = self.network
        image = net.maxpool(net.relu(net.bn1(net.conv1(image))))
        return net.layer4(net.layer3(net.layer2(net.layer1(image))))

    def load_local_weights(self, path):
        """Load a local torchvision ResNet18 state_dict; never access network."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        self.network.load_state_dict(state, strict=True)


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, feedforward, dropout, decoder=False):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True) if decoder else None
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3 if decoder else 2))
        self.feedforward = nn.Sequential(nn.Linear(dim, feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feedforward, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, position, *, padding=None, memory=None, memory_position=None):
        query = value + position
        # need_weights=True selects explicit bmm -> softmax -> bmm instead of
        # fused SDPA kernels. Keep this reference path for ACT numerical
        # debugging across CUDA / PyTorch builds; parameter shapes stay intact.
        attention = self.self_attention(query, query, value, key_padding_mask=padding, need_weights=True)[0]
        value = self.norms[0](value + self.dropout(attention))
        if self.cross_attention is not None:
            attention = self.cross_attention(value + position, memory + memory_position, memory, need_weights=True)[0]
            value = self.norms[1](value + self.dropout(attention))
        return self.norms[-1](value + self.dropout(self.feedforward(value)))


class ACTPolicy(nn.Module):
    policy_type = "act"
    temporal = True  # Explicit chunk state uses the shared evaluator's state slot.
    uses_previous_gripper = False
    action_dim = 7

    def __init__(self, proprio_dim=16, camera_names=("agentview", "robot0_eye_in_hand"), *,
                 chunk_size=32, hidden_dim=512, num_heads=8, encoder_layers=4,
                 decoder_layers=7, feedforward_dim=3200, latent_dim=32,
                 dropout=.1, image_size=128, predict_orientation=True,
                 execution_steps=1, temporal_ensemble=False, ensemble_decay=.01):
        super().__init__()
        self.camera_names = tuple(camera_names)
        if self.camera_names not in (("agentview",), ("agentview", "robot0_eye_in_hand")):
            raise ValueError("Use agentview, optionally followed by robot0_eye_in_hand")
        if min(proprio_dim, chunk_size, hidden_dim, num_heads, encoder_layers, decoder_layers,
               feedforward_dim, latent_dim, image_size) <= 0 or hidden_dim % 4 or hidden_dim % num_heads:
            raise ValueError("Positive dimensions required; hidden_dim must be divisible by 4 and num_heads")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.proprio_dim, self.chunk_size, self.hidden_dim = int(proprio_dim), int(chunk_size), int(hidden_dim)
        self.num_heads, self.encoder_layers, self.decoder_layers = int(num_heads), int(encoder_layers), int(decoder_layers)
        self.feedforward_dim, self.latent_dim, self.dropout = int(feedforward_dim), int(latent_dim), float(dropout)
        self.image_size, self.predict_orientation = int(image_size), bool(predict_orientation)
        self.backbone = VisualBackbone()
        self.image_projection = nn.Conv2d(512, hidden_dim, 1)
        self.camera_embedding = nn.Embedding(len(self.camera_names), hidden_dim)
        self.proprio_projection = nn.Linear(proprio_dim, hidden_dim)
        self.action_projection = nn.Linear(7, hidden_dim)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * .02)
        self.latent_encoder = nn.ModuleList(AttentionBlock(hidden_dim, num_heads, feedforward_dim, dropout) for _ in range(encoder_layers))
        self.latent_statistics = nn.Linear(hidden_dim, 2 * latent_dim)
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.extra_positions = nn.Embedding(2, hidden_dim)
        self.encoder = nn.ModuleList(AttentionBlock(hidden_dim, num_heads, feedforward_dim, dropout) for _ in range(encoder_layers))
        self.decoder = nn.ModuleList(AttentionBlock(hidden_dim, num_heads, feedforward_dim, dropout, decoder=True) for _ in range(decoder_layers))
        self.queries = nn.Embedding(chunk_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 7)
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.configure_inference(execution_steps, temporal_ensemble, ensemble_decay)

    def configure_inference(self, execution_steps=1, temporal_ensemble=False, ensemble_decay=.01):
        if not 1 <= execution_steps <= self.chunk_size:
            raise ValueError("execution_steps must be between 1 and chunk_size")
        if temporal_ensemble and execution_steps != 1:
            raise ValueError("Temporal ensembling requires a new prediction every step (execution_steps=1)")
        if not math.isfinite(ensemble_decay) or ensemble_decay < 0:
            raise ValueError("ensemble_decay must be finite and nonnegative")
        self.execution_steps, self.temporal_ensemble, self.ensemble_decay = int(execution_steps), bool(temporal_ensemble), float(ensemble_decay)

    def visual_tokens(self, image, eye_in_hand_image=None):
        views = {"agentview": image}
        if eye_in_hand_image is not None:
            views["robot0_eye_in_hand"] = eye_in_hand_image
        if set(views) != set(self.camera_names):
            raise ValueError("Camera inputs differ from the ACT checkpoint")
        tokens, positions = [], []
        for index, name in enumerate(self.camera_names):
            frame = views[name]
            if frame.ndim != 4 or frame.shape[1] != 3:
                raise ValueError("ACT expects current RGB images [B,3,H,W], without a history dimension")
            frame = frame.float().div(255) if frame.dtype == torch.uint8 else frame
            frame = F.interpolate(frame, (self.image_size, self.image_size), mode="bilinear", align_corners=False, antialias=True)
            features = self.image_projection(self.backbone((frame - self.image_mean) / self.image_std))
            _, dim, height, width = features.shape
            y = sinusoidal_positions(height, dim // 2, features.device, features.dtype)[:, None].expand(height, width, -1)
            x = sinusoidal_positions(width, dim // 2, features.device, features.dtype)[None].expand(height, width, -1)
            position = torch.cat((y, x), -1).reshape(1, height * width, dim)
            tokens.append(features.flatten(2).transpose(1, 2))
            positions.append(position + self.camera_embedding.weight[index])
        return torch.cat(tokens, 1), torch.cat(positions, 1)

    def forward(self, image, proprio, *, eye_in_hand_image=None, actions=None, is_pad=None,
                sample_posterior=True):
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("ACT expects current proprioception [B,proprio_dim]")
        batch = len(proprio)
        mu = logvar = None
        if actions is not None:
            if actions.shape != (batch, self.chunk_size, 7) or is_pad is None or is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
                raise ValueError("Expected actions [B,K,7] and bool is_pad [B,K]")
            # Padding cannot carry information into the posterior encoder.
            action_tokens = self.action_projection(actions.masked_fill(is_pad[..., None], 0))
            value = torch.cat((self.cls_embedding.expand(batch, -1, -1), self.proprio_projection(proprio)[:, None], action_tokens), 1)
            position = sinusoidal_positions(self.chunk_size + 2, self.hidden_dim, proprio.device, proprio.dtype)[None]
            padding = torch.cat((torch.zeros(batch, 2, dtype=torch.bool, device=proprio.device), is_pad), 1)
            for block in self.latent_encoder:
                value = block(value, position, padding=padding)
            mu, logvar = self.latent_statistics(value[:, 0]).chunk(2, -1)
            latent = mu + torch.randn_like(mu) * (.5 * logvar).exp() if sample_posterior else mu
        else:
            if is_pad is not None:
                raise ValueError("is_pad is only valid with training actions")
            latent = proprio.new_zeros(batch, self.latent_dim)
        visual, visual_position = self.visual_tokens(image, eye_in_hand_image)
        memory = torch.cat((self.latent_projection(latent)[:, None], self.proprio_projection(proprio)[:, None], visual), 1)
        position = torch.cat((self.extra_positions.weight[None], visual_position), 1)
        for block in self.encoder:
            memory = block(memory, position)
        query = self.queries.weight[None].expand(batch, -1, -1)
        value = torch.zeros_like(query)
        for block in self.decoder:
            value = block(value, query, memory=memory, memory_position=position)
        action = self.action_head(value)
        if not self.predict_orientation:
            action = torch.cat((action[..., :3], torch.zeros_like(action[..., 3:6]), action[..., 6:]), -1)
        return dict(action=action, mu=mu, logvar=logvar)

    def step(self, image, proprio, *, eye_in_hand_image=None, hidden=None, previous_gripper=None):
        if self.training:
            raise RuntimeError("Call eval() before ACT rollout")
        state = ChunkState() if hidden is None else hidden
        query = self.temporal_ensemble or state.time % self.execution_steps == 0
        if query:
            if not self.temporal_ensemble:
                state.predictions.clear()
            state.add(self(image, proprio, eye_in_hand_image=eye_in_hand_image)["action"])
        action = state.current(ensemble=self.temporal_ensemble, decay=self.ensemble_decay)
        return dict(action=action, hidden=state)

    def get_config(self):
        keys = ("proprio_dim", "chunk_size", "hidden_dim", "num_heads", "encoder_layers", "decoder_layers",
                "feedforward_dim", "latent_dim", "dropout", "image_size", "predict_orientation",
                "execution_steps", "temporal_ensemble", "ensemble_decay")
        return {"camera_names": list(self.camera_names), **{key: getattr(self, key) for key in keys}}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
