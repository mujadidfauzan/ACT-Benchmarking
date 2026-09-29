"""Smoke-test policy normalization and action round-trip behavior."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.normalization import PolicyNormalizer


PROPRIO_KEYS = (
    "obs__robot0_joint_pos",
    "obs__robot0_joint_vel",
    "obs__robot0_gripper_qpos",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test normalization")
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/normalization.json"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/train.jsonl"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/demos/pilot_v01"),
    )
    return parser.parse_args()


def first_split_entry(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"Split contains no entries: {path}")


def load_real_sample(dataset_root, entry):
    path = dataset_root / entry["path"]
    with np.load(path, allow_pickle=False) as archive:
        proprio = np.concatenate(
            [
                np.asarray(archive[key][0], dtype=np.float32).reshape(-1)
                for key in PROPRIO_KEYS
            ]
        )
        action = np.asarray(archive["actions"][0], dtype=np.float32)
    return torch.from_numpy(proprio), torch.from_numpy(action)


def main():
    args = parse_args()
    normalizer = PolicyNormalizer.from_json(args.statistics)
    normalizer.validate_split(args.split)
    normalizer.eval()

    if any(parameter.requires_grad for parameter in normalizer.parameters()):
        raise AssertionError("Normalizer must not contain trainable parameters")
    if normalizer.proprio_dim != 16:
        raise AssertionError(f"Expected proprio dim 16, got {normalizer.proprio_dim}")
    if normalizer.action_dim != 7:
        raise AssertionError(f"Expected action dim 7, got {normalizer.action_dim}")

    entry = first_split_entry(args.split)
    proprio, action = load_real_sample(args.dataset, entry)
    normalized_proprio = normalizer.normalize_proprio(proprio)
    normalized_action = normalizer.normalize_action(action)
    recovered_action = normalizer.denormalize_action(normalized_action)
    if not torch.allclose(action, recovered_action, atol=1e-6, rtol=1e-6):
        error = torch.max(torch.abs(action - recovered_action)).item()
        raise AssertionError(f"Action round trip failed; max error={error}")

    batch = action.repeat(2, 4, 1)
    recovered_batch = normalizer.denormalize_action(
        normalizer.normalize_action(batch)
    )
    if not torch.allclose(batch, recovered_batch, atol=1e-6, rtol=1e-6):
        raise AssertionError("Batched ACT action round trip failed")

    zero_proprio = normalizer.normalize_proprio(normalizer.proprio_mean)
    zero_action = normalizer.normalize_action(normalizer.action_mean)
    if not torch.equal(zero_proprio, torch.zeros_like(zero_proprio)):
        raise AssertionError("Proprio mean must normalize to zero")
    if not torch.equal(zero_action, torch.zeros_like(zero_action)):
        raise AssertionError("Action mean must normalize to zero")

    state = normalizer.state_dict()
    expected_buffers = {
        "proprio_mean",
        "proprio_std",
        "action_mean",
        "action_std",
    }
    if set(state) != expected_buffers:
        raise AssertionError(f"Unexpected normalizer state: {set(state)}")

    print("Statistics:", args.statistics)
    print("Split hash validation: PASS")
    print("Proprio shape:", tuple(proprio.shape))
    print("Action shape:", tuple(action.shape))
    print("ACT test shape:", tuple(batch.shape))
    print("Normalized proprio range:", (
        float(normalized_proprio.min()), float(normalized_proprio.max())
    ))
    print("Normalized action range:", (
        float(normalized_action.min()), float(normalized_action.max())
    ))
    print("Action round trip: PASS")
    print("Normalization smoke test: PASS")


if __name__ == "__main__":
    main()
