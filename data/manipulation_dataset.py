"""Frame and episode-local recurrent BC datasets with RGB and robot proprioception."""

import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from data.episode_io import HDF5_SUFFIXES, open_episode

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as error:
    raise ImportError(
        "data.manipulation_dataset requires PyTorch. Install torch in the "
        "active environment before creating a dataset."
    ) from error


DEFAULT_POLICY_PROPRIO_KEYS = (
    "obs__robot0_joint_pos",
    "obs__robot0_eef_pos",
    "obs__robot0_gripper_qpos",
    "obs__robot0_eef_quat",
)
CAMERA_OBSERVATION_KEYS = {
    "agentview": "obs__agentview_image",
    "robot0_eye_in_hand": "obs__robot0_eye_in_hand_image",
}
DEFAULT_POLICY_CAMERA_KEYS = (CAMERA_OBSERVATION_KEYS["agentview"],)


def read_split_manifest(path):
    entries = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("task", "path", "steps"):
                if key not in entry:
                    raise ValueError(
                        f"{path}:{line_number} is missing required key {key!r}"
                    )
            entries.append(entry)
    if not entries:
        raise ValueError(f"Split manifest contains no episodes: {path}")
    return entries


class ManipulationDataset(Dataset):
    """Expose each trajectory timestep as a one-step BC training sample.

    The policy receives current RGB and robot proprioception. The current
    default includes joint positions, gripper state, and EEF pose. Prepared
    normalization metadata specifies the selected keys. Object poses and other
    privileged observations remain inaccessible.
    """

    def __init__(
        self,
        dataset_root,
        split_manifest,
        cache_size=16,
        proprio_keys=DEFAULT_POLICY_PROPRIO_KEYS,
        camera_keys=DEFAULT_POLICY_CAMERA_KEYS,
    ):
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")

        self.dataset_root = Path(dataset_root)
        self.split_manifest = Path(split_manifest)
        self.cache_size = int(cache_size)
        self.proprio_keys = tuple(proprio_keys)
        self.camera_keys = tuple(camera_keys)
        if not self.proprio_keys:
            raise ValueError("proprio_keys cannot be empty")
        if any(not key.startswith("obs__") for key in self.proprio_keys):
            raise ValueError("proprio_keys must use stored obs__ names")
        if not self.camera_keys:
            raise ValueError("camera_keys cannot be empty")
        unknown_cameras = set(self.camera_keys) - set(CAMERA_OBSERVATION_KEYS.values())
        if unknown_cameras:
            raise ValueError(f"Unsupported camera keys: {sorted(unknown_cameras)}")
        if CAMERA_OBSERVATION_KEYS["agentview"] not in self.camera_keys:
            raise ValueError("agentview must remain a policy camera")
        self.policy_observation_keys = (*self.camera_keys, *self.proprio_keys)
        self.episodes = read_split_manifest(self.split_manifest)
        self._cache = OrderedDict()
        self._pid = os.getpid()
        self._sample_index = []

        for episode_index, entry in enumerate(self.episodes):
            path = self.dataset_root / entry["path"]
            if path.suffix not in HDF5_SUFFIXES:
                raise ValueError("Training requires HDF5 demos; convert NPZ with python -m data.episode_io")
            if not path.is_file():
                raise FileNotFoundError(f"Episode does not exist: {path}")
            steps = int(entry["steps"])
            if steps <= 0:
                raise ValueError(f"Episode has invalid step count: {path}")
            self._sample_index.extend(
                (episode_index, timestep) for timestep in range(steps)
            )

    def __len__(self):
        return len(self._sample_index)

    @property
    def proprio_dim(self):
        with self._episode(0) as episode:
            return int(sum(episode[key].shape[-1] for key in self.proprio_keys))

    @property
    def action_dim(self):
        with self._episode(0) as episode:
            return int(episode["actions"].shape[-1])

    def _read_episode(self, episode_index):
        entry = self.episodes[episode_index]
        path = self.dataset_root / entry["path"]
        archive = open_episode(path)
        try:
            missing = [
                key
                for key in ("actions", *self.policy_observation_keys)
                if key not in archive
            ]
            if missing:
                raise KeyError(f"{path} is missing policy keys: {missing}")
            actions = archive["actions"]
            if actions.shape != (int(entry["steps"]), 7):
                raise ValueError(f"Action shape mismatch for {path}: {actions.shape}")
            for key in self.policy_observation_keys:
                if not archive[key].ndim or archive[key].shape[0] != len(actions) + 1:
                    raise ValueError(f"Observation alignment mismatch for {path}, key {key}")
            return archive
        except BaseException:
            archive.close()
            raise

    @contextmanager
    def _episode(self, episode_index):
        # Each DataLoader process owns its own bounded set of HDF5 handles.
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if not self.cache_size:
            with self._read_episode(episode_index) as episode:
                yield episode
            return
        if episode_index in self._cache:
            self._cache.move_to_end(episode_index)
            episode = self._cache[episode_index]
        else:
            episode = self._read_episode(episode_index)
            self._cache[episode_index] = episode
            while len(self._cache) > self.cache_size:
                _, previous = self._cache.popitem(last=False)
                previous.close()
        yield episode

    def close(self):
        for handle in getattr(self, "_cache", {}).values():
            handle.close()
        if hasattr(self, "_cache"):
            self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        state["_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _image_tensor(self, image):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image).float().div_(255.0)

    def _proprio_tensor(self, episode, timestep):
        parts = [
            np.asarray(episode[key][timestep], dtype=np.float32).reshape(-1)
            for key in self.proprio_keys
        ]
        return torch.from_numpy(np.concatenate(parts, axis=0))

    def __getitem__(self, index):
        episode_index, timestep = self._sample_index[index]
        entry = self.episodes[episode_index]
        with self._episode(episode_index) as episode:
            sample = {
                "image": self._image_tensor(episode[CAMERA_OBSERVATION_KEYS["agentview"]][timestep]),
                "proprio": self._proprio_tensor(episode, timestep),
                "task": entry["task"],
                "episode_path": entry["path"],
                "timestep": torch.tensor(timestep, dtype=torch.long),
            }
            eye_key = CAMERA_OBSERVATION_KEYS["robot0_eye_in_hand"]
            if eye_key in self.camera_keys:
                sample["eye_in_hand_image"] = self._image_tensor(episode[eye_key][timestep])
            sample["action"] = torch.from_numpy(np.asarray(episode["actions"][timestep], dtype=np.float32))
            return sample


class EpisodeSequenceDataset(ManipulationDataset):
    """Causal, episode-local windows with right padding and previous gripper input."""

    def __init__(self, *args, sequence_length=16, sequence_stride=8, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= sequence_stride <= sequence_length:
            raise ValueError("Need 1 <= sequence-stride <= sequence-length")
        self.sequence_length = int(sequence_length)
        self.windows = [(index, start) for index, entry in enumerate(self.episodes)
                        for start in range(0, int(entry["steps"]), sequence_stride)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, start = self.windows[index]
        entry = self.episodes[episode_index]
        end = min(start + self.sequence_length, int(entry["steps"]))
        valid = end - start

        def pad_last(value):
            if valid == self.sequence_length:
                return value
            return torch.cat((value, value[-1:].expand(self.sequence_length - valid, *value.shape[1:])))

        with self._episode(episode_index) as episode:
            proprio = np.concatenate([episode[key][start:end] for key in self.proprio_keys], axis=-1).astype(np.float32)
            sample = {"task": entry["task"], "episode_path": entry["path"],
                      "timestep": pad_last(torch.arange(start, end)),
                      "proprio": pad_last(torch.from_numpy(proprio)),
                      "action": pad_last(torch.from_numpy(episode["actions"][start:end].astype(np.float32)))}
            for name, output_key in (("agentview", "image"), ("robot0_eye_in_hand", "eye_in_hand_image")):
                key = CAMERA_OBSERVATION_KEYS[name]
                if key in self.camera_keys:
                    frames = np.ascontiguousarray(episode[key][start:end].transpose(0, 3, 1, 2))
                    sample[output_key] = pad_last(torch.from_numpy(frames).float().div_(255.0))
            if start == 0:
                previous = np.concatenate(([-1.0], episode["actions"][:end - 1, 6]))
            else:
                previous = episode["actions"][start - 1:end - 1, 6]
            sample["previous_gripper"] = pad_last(torch.from_numpy(np.where(previous > 0, 1., -1.).astype(np.float32)[:, None]))
            sample["valid_mask"] = torch.arange(self.sequence_length) < valid
            return sample
