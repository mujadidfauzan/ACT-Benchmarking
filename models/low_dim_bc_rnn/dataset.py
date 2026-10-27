"""Read only small numeric arrays from existing HDF5 demonstration episodes."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.episode_io import open_episode
from data.manipulation_dataset import read_split_manifest
from .observations import STATE_FIELDS, pack_state


class LowDimSequenceDataset(Dataset):
    """Stride-one windows, with repeated end padding included in the loss.

    Numeric states/actions are validated and cached in RAM once. For the current
    1,000-demo dataset this is about 27 MiB; RGB arrays are never requested.
    """

    def __init__(self, dataset_root, split_manifest, sequence_length=10):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = int(sequence_length)
        self.episodes = read_split_manifest(split_manifest)
        self._arrays, self._sample_index = [], []
        for index, entry in enumerate(self.episodes):
            path = Path(dataset_root) / entry["path"]
            steps = int(entry["steps"])
            if entry["task"] != "place" or steps <= 0:
                raise ValueError(f"{path}: expected a nonempty Place episode")
            with open_episode(path) as episode:
                missing = [key for key in ("actions", *("obs__" + name for name, _ in STATE_FIELDS))
                           if key not in episode]
                if missing:
                    raise ValueError(f"{path}: missing low-dim keys {missing}")
                action = episode["actions"][:].astype(np.float32)
                if action.shape != (steps, 7) or not np.isfinite(action).all() or (np.abs(action) > 1.000001).any():
                    raise ValueError(f"{path}: expected finite controller actions [{steps},7] in [-1,1]")
                arrays = {}
                for name, dim in STATE_FIELDS:
                    key = "obs__" + name
                    if episode[key].shape != (steps + 1, dim):
                        raise ValueError(f"{path}: {key} must have shape ({steps + 1},{dim})")
                    arrays[key] = episode[key][:]
                state = pack_state(arrays, prefix="obs__")
            self._arrays.append((torch.from_numpy(state), torch.from_numpy(action)))
            self._sample_index.extend((index, start) for start in range(steps))

    def __len__(self):
        return len(self._sample_index)

    def __getitem__(self, index):
        episode, start = self._sample_index[index]
        state, action = self._arrays[episode]
        end = min(start + self.sequence_length, len(action))

        def padded(value):
            value = value[start:end]
            tail = self.sequence_length - len(value)
            return torch.cat((value, value[-1:].expand(tail, -1))) if tail else value

        # obs[t] predicts action[t]; the terminal obs[T] is never an input window.
        return {"state": padded(state), "action": padded(action)}

    def close(self):
        """No open file handles: provided for the trainer dataset interface."""
