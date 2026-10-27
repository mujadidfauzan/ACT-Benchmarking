"""Raw 23D state -> MLP -> LSTM -> continuous seven-dimensional action."""

import torch
from torch import nn

from .observations import STATE_DIM, STATE_FIELDS, pack_state


class LowDimBCRNNPolicy(nn.Module):
    policy_type = "low_dim_bc_rnn"
    temporal = True
    action_dim = 7
    uses_previous_gripper = False
    camera_names = ()
    video_camera_names = ("agentview",)
    render_observation_cameras = False
    observation_mode = "low_dim"

    def __init__(self, *, hidden_dim=1000, num_layers=2, sequence_length=10,
                 encoder_dim=128, predict_orientation=True):
        super().__init__()
        if min(hidden_dim, num_layers, sequence_length, encoder_dim) <= 0:
            raise ValueError("Network and sequence dimensions must be positive")
        if not predict_orientation:
            raise ValueError("The Place baseline requires seven-dimensional pose actions")
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.sequence_length, self.encoder_dim = int(sequence_length), int(encoder_dim)
        self.predict_orientation = True
        self.encoder = nn.Sequential(nn.Linear(STATE_DIM, encoder_dim), nn.ReLU(),
                                     nn.Linear(encoder_dim, encoder_dim), nn.ReLU())
        self.lstm = nn.LSTM(encoder_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, self.action_dim)

    def forward(self, state, *, hidden=None):
        if state.ndim != 3 or state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected state [B,T,{STATE_DIM}]")
        encoded, hidden = self.lstm(self.encoder(state), hidden)
        return {"action": self.head(encoded).tanh(), "hidden": hidden}

    def step(self, state, *, hidden=None, previous_gripper=None):
        if self.training:
            raise RuntimeError("Call eval() before rollout")
        memory, count = (None, 0) if hidden is None else hidden
        reset = count % self.sequence_length == 0
        result = self(state.unsqueeze(1), hidden=None if reset else memory)
        return {"action": result["action"][:, 0],
                "hidden": (result["hidden"], count + 1), "hidden_state_reset": reset}

    def observation_inputs(self, observation, device):
        """The shared evaluator calls this instead of camera preprocessing."""
        return {"state": torch.from_numpy(pack_state(observation)).unsqueeze(0).to(device)}

    def get_config(self):
        return dict(hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                    sequence_length=self.sequence_length, encoder_dim=self.encoder_dim,
                    predict_orientation=self.predict_orientation,
                    state_fields=[[key, dim] for key, dim in STATE_FIELDS])

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        if config.pop("state_fields") != [[key, dim] for key, dim in STATE_FIELDS]:
            raise ValueError("Checkpoint low-dim observation order differs")
        return cls(**config)


class RawStateInputs:
    """Controller outputs stay raw; no visual-BC normalization is applied."""

    @staticmethod
    def denormalize_action(value):
        return value
