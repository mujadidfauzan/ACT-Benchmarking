"""Smoke-test BC and ACT dataset loading on a small number of samples."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from data.manipulation_dataset import (
    CAMERA_OBSERVATION_KEYS,
    DEFAULT_POLICY_PROPRIO_KEYS,
    EEF_GRIPPER_PROPRIO_KEYS,
    ManipulationDataset,
    create_dataloader,
)


PROPRIO_PRESETS = {
    "full": DEFAULT_POLICY_PROPRIO_KEYS,
    "eef_gripper": EEF_GRIPPER_PROPRIO_KEYS,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test manipulation loaders")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/demos/pilot_v01"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/train.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--history-size",
        type=int,
        default=1,
        help="Emit BC proprio/action history when greater than one",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--proprio-preset",
        choices=tuple(PROPRIO_PRESETS),
        default="full",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        choices=tuple(CAMERA_OBSERVATION_KEYS),
        default=["agentview"],
    )
    parser.add_argument(
        "--vertical-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip legacy OpenGL images vertically (enabled for pilot_v01)",
    )
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def assert_finite(name, tensor):
    require(torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf")


def verify_common_sample(sample, proprio_dim, image_size=(256, 256)):
    image = sample["image"]
    proprio = sample["proprio"]
    require(image.shape == (3, *image_size), f"unexpected image shape: {image.shape}")
    require(image.dtype == torch.float32, f"unexpected image dtype: {image.dtype}")
    require(0.0 <= image.min().item(), "image contains values below zero")
    require(image.max().item() <= 1.0, "image contains values above one")
    require(proprio.shape == (proprio_dim,), f"unexpected proprio shape: {proprio.shape}")
    require(proprio.dtype == torch.float32, f"unexpected proprio dtype: {proprio.dtype}")
    require(isinstance(sample["instruction"], str), "instruction must be a string")
    require(isinstance(sample["task"], str), "task must be a string")
    assert_finite("image", image)
    assert_finite("proprio", proprio)
    if "eye_in_hand_image" in sample:
        eye_image = sample["eye_in_hand_image"]
        require(
            eye_image.shape == (3, *image_size),
            f"unexpected eye-in-hand shape: {eye_image.shape}",
        )
        require(
            eye_image.dtype == torch.float32,
            f"unexpected eye-in-hand dtype: {eye_image.dtype}",
        )
        require(0.0 <= eye_image.min().item(), "eye image contains values below zero")
        require(eye_image.max().item() <= 1.0, "eye image contains values above one")
        assert_finite("eye-in-hand image", eye_image)


def verify_vertical_flip(dataset):
    sample = dataset[0]
    entry = dataset.episodes[0]
    path = dataset.dataset_root / entry["path"]
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["obs__agentview_image"][0]
    expected = np.flip(raw, axis=0) if dataset.vertical_flip else raw
    loaded = sample["image"].mul(255).round().byte().permute(1, 2, 0).numpy()
    require(np.array_equal(loaded, expected), "loaded image transform is incorrect")


def verify_bc(args):
    dataset = ManipulationDataset(
        dataset_root=args.dataset,
        split_manifest=args.split,
        mode="bc",
        vertical_flip=args.vertical_flip,
        camera_keys=tuple(
            CAMERA_OBSERVATION_KEYS[name] for name in args.camera_names
        ),
        proprio_keys=PROPRIO_PRESETS[args.proprio_preset],
        history_size=args.history_size,
    )
    sample = dataset[0]
    verify_common_sample(sample, dataset.proprio_dim)
    require(sample["action"].shape == (dataset.action_dim,), "invalid BC action shape")
    require(sample["action"].dtype == torch.float32, "BC action must be float32")
    assert_finite("BC action", sample["action"])
    verify_vertical_flip(dataset)

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    batch_size = len(batch["task"])
    require(batch["image"].shape == (batch_size, 3, 256, 256), "invalid BC image batch")
    require(
        batch["proprio"].shape == (batch_size, dataset.proprio_dim),
        "invalid BC proprio batch",
    )
    require(
        batch["action"].shape == (batch_size, dataset.action_dim),
        "invalid BC action batch",
    )
    if "robot0_eye_in_hand" in args.camera_names:
        require(
            batch["eye_in_hand_image"].shape
            == (batch_size, 3, 256, 256),
            "invalid BC eye-in-hand image batch",
        )
    if args.history_size > 1:
        require(
            batch["proprio_history"].shape
            == (batch_size, args.history_size, dataset.proprio_dim),
            "invalid BC proprio history batch",
        )
        require(
            batch["action_history"].shape
            == (batch_size, args.history_size - 1, dataset.action_dim),
            "invalid BC action history batch",
        )
        require(
            batch["proprio_history_mask"].shape
            == (batch_size, args.history_size),
            "invalid BC proprio history mask",
        )
        require(
            batch["action_history_mask"].shape
            == (batch_size, args.history_size - 1),
            "invalid BC action history mask",
        )
    return dataset, batch


def verify_act(args):
    dataset = ManipulationDataset(
        dataset_root=args.dataset,
        split_manifest=args.split,
        mode="act",
        chunk_size=args.chunk_size,
        vertical_flip=args.vertical_flip,
        camera_keys=tuple(
            CAMERA_OBSERVATION_KEYS[name] for name in args.camera_names
        ),
        proprio_keys=PROPRIO_PRESETS[args.proprio_preset],
    )
    sample = dataset[0]
    verify_common_sample(sample, dataset.proprio_dim)
    require(
        sample["actions"].shape == (args.chunk_size, dataset.action_dim),
        "invalid ACT action chunk shape",
    )
    require(
        sample["action_padding_mask"].shape == (args.chunk_size,),
        "invalid ACT padding-mask shape",
    )
    require(
        sample["action_padding_mask"].dtype == torch.bool,
        "ACT padding mask must be boolean",
    )
    assert_finite("ACT actions", sample["actions"])

    final_index = dataset.episode_sample_ranges[0].stop - 1
    final_sample = dataset[final_index]
    final_mask = final_sample["action_padding_mask"]
    require((~final_mask).sum().item() == 1, "final ACT chunk must have one real action")
    require(final_mask[1:].all().item(), "final ACT chunk padding is incorrect")
    require(
        torch.count_nonzero(final_sample["actions"][1:]).item() == 0,
        "padded ACT actions must be zero",
    )

    loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))
    batch_size = len(batch["task"])
    require(batch["image"].shape == (batch_size, 3, 256, 256), "invalid ACT image batch")
    require(
        batch["proprio"].shape == (batch_size, dataset.proprio_dim),
        "invalid ACT proprio batch",
    )
    require(
        batch["actions"].shape == (
            batch_size,
            args.chunk_size,
            dataset.action_dim,
        ),
        "invalid ACT action batch",
    )
    require(
        batch["action_padding_mask"].shape == (batch_size, args.chunk_size),
        "invalid ACT mask batch",
    )
    if "robot0_eye_in_hand" in args.camera_names:
        require(
            batch["eye_in_hand_image"].shape
            == (batch_size, 3, 256, 256),
            "invalid ACT eye-in-hand image batch",
        )
    return dataset, batch


def print_batch(name, dataset, batch):
    print(f"\n{name}")
    print("-" * len(name))
    print("Dataset samples:", len(dataset))
    print("Episodes:", len(dataset.episodes))
    print("Image:", tuple(batch["image"].shape), batch["image"].dtype)
    if "eye_in_hand_image" in batch:
        print(
            "Eye-in-hand image:",
            tuple(batch["eye_in_hand_image"].shape),
            batch["eye_in_hand_image"].dtype,
        )
    print("Proprio:", tuple(batch["proprio"].shape), batch["proprio"].dtype)
    if "proprio_history" in batch:
        print("Proprio history:", tuple(batch["proprio_history"].shape))
        print("Action history:", tuple(batch["action_history"].shape))
        print("Proprio history mask:", tuple(batch["proprio_history_mask"].shape))
        print("Action history mask:", tuple(batch["action_history_mask"].shape))
    if "action" in batch:
        print("Action:", tuple(batch["action"].shape), batch["action"].dtype)
    else:
        print("Actions:", tuple(batch["actions"].shape), batch["actions"].dtype)
        print(
            "Padding mask:",
            tuple(batch["action_padding_mask"].shape),
            batch["action_padding_mask"].dtype,
        )
    print("Tasks:", list(batch["task"]))
    print("First instruction:", batch["instruction"][0])


def verify_task_balancing(dataset, batch_size):
    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        task_balanced=True,
        seed=42,
    )
    counts = Counter()
    for indices in loader.batch_sampler:
        episode_indices = {
            dataset._sample_index[index][0] for index in indices
        }
        require(len(episode_indices) == 1, "a batch crosses episode boundaries")
        episode_index = next(iter(episode_indices))
        counts[dataset.episodes[episode_index]["task"]] += 1
    require(len(set(counts.values())) == 1, f"task batches are not balanced: {counts}")
    print("\nTASK-BALANCED SAMPLER")
    print("---------------------")
    print("Batches per task:", dict(sorted(counts.items())))


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.history_size <= 0:
        raise ValueError("--history-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    print("Dataset:", args.dataset)
    print("Split:", args.split)
    print("Vertical flip:", args.vertical_flip)
    print("Cameras:", args.camera_names)
    print("Proprio preset:", args.proprio_preset)
    bc_dataset, bc_batch = verify_bc(args)
    print_batch("BC LOADER", bc_dataset, bc_batch)
    act_dataset, act_batch = verify_act(args)
    print_batch("ACT LOADER", act_dataset, act_batch)
    verify_task_balancing(bc_dataset, args.batch_size)
    print("\nDataset loader smoke test: PASS")


if __name__ == "__main__":
    main()
