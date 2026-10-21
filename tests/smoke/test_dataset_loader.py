"""Check a prepared Visual-BC dataset without training or simulation."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.bc_preparation import default_training_dir, load_bc_training
from data.manipulation_dataset import CAMERA_OBSERVATION_KEYS, EpisodeSequenceDataset, ManipulationDataset
from data.normalization import PolicyNormalizer
from data.episode_io import open_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Check prepared frame or temporal BC inputs")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and worker count nonnegative")
    training_dir = args.training_dir or default_training_dir(args.dataset)
    config, training_dir = load_bc_training(args.dataset, training_dir)
    normalizer = PolicyNormalizer.from_statistics(config["normalization"])
    for split in ("train", "val"):
        dataset_class = EpisodeSequenceDataset if args.temporal else ManipulationDataset
        sequence = dict(sequence_length=args.sequence_length, sequence_stride=args.sequence_stride) if args.temporal else {}
        dataset = dataset_class(
            args.dataset,
            training_dir / f"{split}.jsonl",
            proprio_keys=config["proprio_keys"],
            camera_keys=tuple(CAMERA_OBSERVATION_KEYS[name] for name in config["camera_names"]),
            **sequence,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        batch = next(iter(loader))
        size = len(batch["task"])
        prefix = (size, args.sequence_length) if args.temporal else (size,)
        assert batch["proprio"].shape == (*prefix, normalizer.proprio_dim)
        assert batch["action"].shape == (*prefix, normalizer.action_dim)
        for name, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.isfinite(tensor).all(), name
        for key in ("image", "eye_in_hand_image"):
            if key in batch:
                assert batch[key].shape[:-2] == (*prefix, 3)
                assert 0 <= batch[key].min() and batch[key].max() <= 1
        with open_episode(args.dataset / dataset.episodes[0]["path"]) as archive:
            expected = archive[CAMERA_OBSERVATION_KEYS["agentview"]][0]
        frame = batch["image"][0, 0] if args.temporal else batch["image"][0]
        actual = frame.mul(255).round().byte().permute(1, 2, 0).numpy()
        assert np.array_equal(actual, expected), "RGB must not be vertically flipped"
        assert torch.isfinite(normalizer.normalize_proprio(batch["proprio"])).all()
        normalized_action = normalizer.normalize_action(batch["action"])
        torch.testing.assert_close(normalizer.denormalize_action(normalized_action), batch["action"])
        print(f"{split}: {len(dataset)} samples, {len(dataset.episodes)} episodes, {len(loader)} batches")
        print(f"  RGB: {tuple(batch['image'].shape)}, proprio: {tuple(batch['proprio'].shape)}, action: {tuple(batch['action'].shape)}")
        dataset.close()
    print("Dataset loader smoke test: PASS")


if __name__ == "__main__":
    main()
