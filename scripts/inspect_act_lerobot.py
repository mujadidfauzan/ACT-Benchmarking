"""Inspect source alignment, conditioning, and all action chunk offsets offline."""

import argparse
import os
from pathlib import Path

import numpy as np

from data.bc_preparation import load_bc_training, save_json
from models.act.dataset import ACTDataset
from models.act_lerobot.audit import audit_inputs
from scripts.evaluate_act_lerobot import load_policy
from scripts.train_act import select_episodes
from scripts.train_visual_bc import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    return args


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    prepared, directory = load_bc_training(args.dataset, args.training_dir)
    device = resolve_device(args.device)
    model, normalizer, checkpoint = load_policy(args.checkpoint, device)
    if checkpoint["data_config"] != prepared:
        raise ValueError("Dataset/split/normalization differs from checkpoint")
    data = ACTDataset(args.dataset, directory / f"{args.split}.jsonl", model.camera_names,
                      normalizer.proprio_keys, model.chunk_size)
    try:
        if args.split == "train":
            select_episodes(data, [e["path"] for e in checkpoint["training_subset"]])
        report, arrays = audit_inputs(model, normalizer, data, device, args.samples)
    finally:
        data.close()
    args.output_dir.mkdir(parents=True)
    save_json(args.output_dir / "summary.json", dict(checkpoint=str(args.checkpoint), split=args.split,
                                                    checkpoint_updates=checkpoint["updates"], **report))
    np.savez_compressed(args.output_dir / "chunks.npz", **arrays)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    for i, sample in enumerate(report["samples"]):
        fig, axes = plt.subplots(4, 2, figsize=(11, 10), layout="constrained")
        valid = sample["valid_actions"]
        for j, label in enumerate(labels):
            axes.flat[j].plot(arrays["predicted_chunk_raw"][i, :, j], label="Prediction z=0")
            axes.flat[j].plot(np.arange(valid), arrays["target_chunk_raw"][i, :valid, j], "--", label="Expert target")
            axes.flat[j].set(title=label, xlabel="Future action offset", ylabel="Controller units")
            axes.flat[j].grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        error = abs(arrays["predicted_chunk_normalized"][i, :valid]-arrays["target_chunk_normalized"][i, :valid]).mean(-1)
        axes.flat[7].plot(error);axes.flat[7].set(title="Normalized L1 per valid offset",xlabel="Future action offset")
        fig.suptitle(f"{sample['path']} | observation t={sample['start']} | offline chunk prediction")
        fig.savefig(args.output_dir / f"sample_{i:03d}_chunk.png", dpi=120);plt.close(fig)
        frames = [arrays[f"image_{name}"][i].transpose(1,2,0) for name in model.camera_names]
        cv2.imwrite(str(args.output_dir / f"sample_{i:03d}_observation.png"),
                    cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR))
    print("Source alignment and normalization: PASS")
    print("First-action std across observations:", report["first_action_std_across_samples"])
    print("Camera shuffle sensitivity:", report["camera_shuffle_mean_abs_delta"])
    print("Proprio shuffle sensitivity:", report["proprio_shuffle_mean_abs_delta"])
    print("All chunk offsets saved:", args.output_dir / "chunks.npz")


if __name__ == "__main__":
    main()
