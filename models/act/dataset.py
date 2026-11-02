"""Current observation -> future action chunk, without converting HDF5 demos."""

import numpy as np
import torch

from data.manipulation_dataset import ManipulationDataset, CAMERA_OBSERVATION_KEYS


class ACTDataset(ManipulationDataset):
    def __init__(self, dataset_root, split_manifest, camera_names, proprio_keys,
                 chunk_size=32):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        super().__init__(dataset_root, split_manifest, proprio_keys=proprio_keys,
                         camera_keys=[CAMERA_OBSERVATION_KEYS[n] for n in camera_names])
        self.camera_names, self.chunk_size = tuple(camera_names), int(chunk_size)

    def __getitem__(self, index):
        episode_index, start = self._sample_index[index]
        steps = int(self.episodes[episode_index]["steps"])
        end = min(start + self.chunk_size, steps)
        with self._episode(episode_index) as episode:
            action = np.zeros((self.chunk_size, 7), dtype=np.float32)
            action[:end - start] = episode["actions"][start:end]
            if not np.isfinite(action).all() or (np.abs(action) > 1.000001).any():
                raise ValueError("ACT requires finite controller actions in [-1,1]")
            proprio = np.concatenate([episode[k][start] for k in self.proprio_keys]).astype(np.float32)
            if not np.isfinite(proprio).all():
                raise ValueError("ACT requires finite robot proprioception")
            sample = dict(proprio=torch.from_numpy(proprio), action=torch.from_numpy(action),
                          is_pad=torch.arange(self.chunk_size) >= end - start)
            for name in self.camera_names:
                frame = episode[CAMERA_OBSERVATION_KEYS[name]][start]
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("Expected uint8 HWC camera observation")
                key = "image" if name == "agentview" else "eye_in_hand_image"
                sample[key] = torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        return sample
