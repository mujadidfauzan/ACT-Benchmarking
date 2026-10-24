"""Read existing manifests / T+1 observation HDF5 episodes without conversion."""

import numpy as np
import torch

from data.manipulation_dataset import ManipulationDataset, CAMERA_OBSERVATION_KEYS
from .policy import PROPRIO_KEYS


class BCRNNSequenceDataset(ManipulationDataset):
    """One forward window per action, repeated last frame/action at episode end.

    Padding participates in the MSE, matching robomimic's default padded
    SequenceDataset + deterministic BC-RNN loss (no valid-mask weighting).
    """

    def __init__(self, dataset_root, split_manifest, camera_names, sequence_length=10):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        super().__init__(dataset_root, split_manifest, proprio_keys=PROPRIO_KEYS,
                         camera_keys=[CAMERA_OBSERVATION_KEYS[name] for name in camera_names])
        self.camera_names, self.sequence_length = tuple(camera_names), int(sequence_length)

    def __getitem__(self, index):
        episode_index, start = self._sample_index[index]
        end = min(start + self.sequence_length, int(self.episodes[episode_index]["steps"]))

        def padded(value):
            value = torch.from_numpy(np.ascontiguousarray(value))
            tail = self.sequence_length - len(value)
            return torch.cat((value, value[-1:].expand(tail, *value.shape[1:]))) if tail else value

        with self._episode(episode_index) as episode:
            action = episode["actions"][start:end].astype(np.float32)
            if not np.isfinite(action).all() or (np.abs(action) > 1.000001).any():
                raise ValueError("BC-RNN requires finite controller actions in [-1,1]")
            parts = [episode[key][start:end].astype(np.float32) for key in self.proprio_keys]
            proprio = np.concatenate(parts, -1)
            if proprio.shape[-1] != 9 or not np.isfinite(proprio).all():
                raise ValueError("Expected finite 9D EEF/gripper proprio")
            sample = dict(proprio=padded(proprio), action=padded(action))
            for name in self.camera_names:
                frames = episode[CAMERA_OBSERVATION_KEYS[name]][start:end]
                if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
                    raise ValueError("Expected uint8 THWC camera observations")
                key = "image" if name == "agentview" else "eye_in_hand_image"
                sample[key] = padded(frames.transpose(0, 3, 1, 2))
            return sample
