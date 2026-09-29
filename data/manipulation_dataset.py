"""PyTorch datasets for one-step BC and action-chunking policies."""

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError as error:
    raise ImportError(
        "data.manipulation_dataset requires PyTorch. Install torch in the "
        "active environment before creating a dataset."
    ) from error


POLICY_OBSERVATION_KEYS = (
    "obs__agentview_image",
    "obs__robot0_joint_pos",
    "obs__robot0_joint_vel",
    "obs__robot0_gripper_qpos",
)


def read_split_manifest(path):
    entries = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("task", "path", "steps", "instruction"):
                if key not in entry:
                    raise ValueError(
                        f"{path}:{line_number} is missing required key {key!r}"
                    )
            entries.append(entry)
    if not entries:
        raise ValueError(f"Split manifest contains no episodes: {path}")
    return entries


class ManipulationDataset(Dataset):
    """Expose each trajectory timestep as a BC or ACT training sample.

    The policy receives one RGB frame and proprioception composed of joint
    position, joint velocity, and gripper position. Object poses and all other
    privileged observations remain inaccessible here.
    """

    def __init__(
        self,
        dataset_root,
        split_manifest,
        mode="bc",
        chunk_size=32,
        vertical_flip=False,
        cache_size=1,
    ):
        if mode not in {"bc", "act"}:
            raise ValueError("mode must be 'bc' or 'act'")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")

        self.dataset_root = Path(dataset_root)
        self.split_manifest = Path(split_manifest)
        self.mode = mode
        self.chunk_size = int(chunk_size)
        self.vertical_flip = bool(vertical_flip)
        self.cache_size = int(cache_size)
        self.episodes = read_split_manifest(self.split_manifest)
        self._cache = OrderedDict()
        self._sample_index = []
        self.episode_sample_ranges = []

        for episode_index, entry in enumerate(self.episodes):
            path = self.dataset_root / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(f"Episode does not exist: {path}")
            steps = int(entry["steps"])
            if steps <= 0:
                raise ValueError(f"Episode has invalid step count: {path}")
            start = len(self._sample_index)
            self._sample_index.extend(
                (episode_index, timestep) for timestep in range(steps)
            )
            self.episode_sample_ranges.append(range(start, start + steps))

    def __len__(self):
        return len(self._sample_index)

    @property
    def proprio_dim(self):
        episode = self._load_episode(0)
        return int(
            sum(episode[key].shape[-1] for key in POLICY_OBSERVATION_KEYS[1:])
        )

    @property
    def action_dim(self):
        return int(self._load_episode(0)["actions"].shape[-1])

    def _read_episode(self, episode_index):
        entry = self.episodes[episode_index]
        path = self.dataset_root / entry["path"]
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in ("actions", *POLICY_OBSERVATION_KEYS) if key not in archive]
            if missing:
                raise KeyError(f"{path} is missing policy keys: {missing}")
            episode = {
                "actions": np.asarray(archive["actions"], dtype=np.float32),
            }
            for key in POLICY_OBSERVATION_KEYS:
                episode[key] = np.asarray(archive[key])

        steps = len(episode["actions"])
        if steps != int(entry["steps"]):
            raise ValueError(
                f"Step count mismatch for {path}: split={entry['steps']}, npz={steps}"
            )
        for key in POLICY_OBSERVATION_KEYS:
            if len(episode[key]) != steps + 1:
                raise ValueError(
                    f"Observation alignment mismatch for {path}, key {key}"
                )
        return episode

    def _load_episode(self, episode_index):
        if episode_index in self._cache:
            self._cache.move_to_end(episode_index)
            return self._cache[episode_index]

        episode = self._read_episode(episode_index)
        if self.cache_size:
            self._cache[episode_index] = episode
            self._cache.move_to_end(episode_index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return episode

    def _image_tensor(self, image):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got {image.shape}")
        if self.vertical_flip:
            image = np.flip(image, axis=0)
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        return torch.from_numpy(image).float().div_(255.0)

    @staticmethod
    def _proprio_tensor(episode, timestep):
        parts = [
            np.asarray(episode[key][timestep], dtype=np.float32).reshape(-1)
            for key in POLICY_OBSERVATION_KEYS[1:]
        ]
        return torch.from_numpy(np.concatenate(parts, axis=0))

    def _bc_target(self, actions, timestep):
        return torch.from_numpy(actions[timestep].copy())

    def _act_target(self, actions, timestep):
        available = min(self.chunk_size, len(actions) - timestep)
        chunk = np.zeros((self.chunk_size, actions.shape[-1]), dtype=np.float32)
        chunk[:available] = actions[timestep : timestep + available]
        padding_mask = np.ones(self.chunk_size, dtype=np.bool_)
        padding_mask[:available] = False
        return torch.from_numpy(chunk), torch.from_numpy(padding_mask)

    def __getitem__(self, index):
        episode_index, timestep = self._sample_index[index]
        entry = self.episodes[episode_index]
        episode = self._load_episode(episode_index)
        sample = {
            "image": self._image_tensor(
                episode["obs__agentview_image"][timestep]
            ),
            "proprio": self._proprio_tensor(episode, timestep),
            "instruction": entry["instruction"],
            "task": entry["task"],
            "episode_path": entry["path"],
            "timestep": torch.tensor(timestep, dtype=torch.long),
        }
        if self.mode == "bc":
            sample["action"] = self._bc_target(episode["actions"], timestep)
        else:
            actions, padding_mask = self._act_target(episode["actions"], timestep)
            sample["actions"] = actions
            sample["action_padding_mask"] = padding_mask
        return sample


class EpisodeBatchSampler(Sampler):
    """Batch timesteps within episodes to avoid repeated NPZ decompression."""

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, seed=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        episode_order = list(range(len(self.dataset.episode_sample_ranges)))
        if self.shuffle:
            episode_order = torch.randperm(
                len(episode_order), generator=generator
            ).tolist()

        for episode_index in episode_order:
            indices = list(self.dataset.episode_sample_ranges[episode_index])
            if self.shuffle:
                permutation = torch.randperm(
                    len(indices), generator=generator
                ).tolist()
                indices = [indices[index] for index in permutation]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        if self.drop_last:
            return sum(
                len(indices) // self.batch_size
                for indices in self.dataset.episode_sample_ranges
            )
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.dataset.episode_sample_ranges
        )


class TaskBalancedBatchSampler(Sampler):
    """Sample an equal number of episode-local batches for every task."""

    def __init__(
        self,
        dataset,
        batch_size,
        drop_last=False,
        seed=0,
        batches_per_task=None,
        task_block_batches=8,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batches_per_task is not None and batches_per_task <= 0:
            raise ValueError("batches_per_task must be positive")
        if task_block_batches <= 0:
            raise ValueError("task_block_batches must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.task_block_batches = int(task_block_batches)
        self.epoch = 0

        self.task_episode_indices = {}
        for episode_index, entry in enumerate(dataset.episodes):
            self.task_episode_indices.setdefault(entry["task"], []).append(
                episode_index
            )
        if not self.task_episode_indices:
            raise ValueError("Dataset contains no tasks")

        natural_batches = sum(
            self._episode_batch_count(len(indices))
            for indices in dataset.episode_sample_ranges
        )
        self.batches_per_task = (
            int(batches_per_task)
            if batches_per_task is not None
            else math.ceil(natural_batches / len(self.task_episode_indices))
        )

    def _episode_batch_count(self, sample_count):
        if self.drop_last:
            return sample_count // self.batch_size
        return math.ceil(sample_count / self.batch_size)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _task_batches(self, task, generator):
        batches = []
        episode_indices = self.task_episode_indices[task]
        episode_order = torch.randperm(
            len(episode_indices), generator=generator
        ).tolist()
        for order_index in episode_order:
            episode_index = episode_indices[order_index]
            indices = list(self.dataset.episode_sample_ranges[episode_index])
            permutation = torch.randperm(
                len(indices), generator=generator
            ).tolist()
            indices = [indices[index] for index in permutation]
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        return batches

    def _select_balanced_batches(self, batches, generator):
        if not batches:
            raise ValueError("A task has no batches after applying drop_last")
        selected = []
        while len(selected) < self.batches_per_task:
            remaining = self.batches_per_task - len(selected)
            selected.extend(batches[:remaining])
            if remaining > len(batches):
                order = torch.randperm(len(batches), generator=generator).tolist()
                batches = [batches[index] for index in order]
        return selected

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        tasks = sorted(self.task_episode_indices)
        selected = {
            task: self._select_balanced_batches(
                self._task_batches(task, generator), generator
            )
            for task in tasks
        }
        positions = {task: 0 for task in tasks}
        while any(position < self.batches_per_task for position in positions.values()):
            task_order = torch.randperm(len(tasks), generator=generator).tolist()
            for task_index in task_order:
                task = tasks[task_index]
                start = positions[task]
                end = min(start + self.task_block_batches, self.batches_per_task)
                yield from selected[task][start:end]
                positions[task] = end

    def __len__(self):
        return self.batches_per_task * len(self.task_episode_indices)


def create_dataloader(
    dataset,
    batch_size,
    shuffle=None,
    num_workers=0,
    pin_memory=None,
    drop_last=False,
    seed=0,
    task_balanced=False,
    batches_per_task=None,
    task_block_batches=8,
):
    """Create a DataLoader with conservative defaults for compressed NPZ data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if shuffle is None:
        shuffle = "train" in dataset.split_manifest.stem
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if task_balanced:
        if not shuffle:
            raise ValueError("task_balanced sampling requires shuffle=True")
        batch_sampler = TaskBalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
            batches_per_task=batches_per_task,
            task_block_batches=task_block_batches,
        )
    else:
        batch_sampler = EpisodeBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
