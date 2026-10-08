"""Overfit one fixed batch to validate Visual-BC optimization mechanics."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from data.manipulation_dataset import CAMERA_OBSERVATION_KEYS, ManipulationDataset
from data.normalization import PolicyNormalizer
from models.visual_bc import (
    VISUAL_BC_FUSION_MODES,
    HistoryVisualBCPolicy,
    VisualBCPolicy,
)
from models.spatial_softmax_visual_bc import SpatialSoftmaxVisualBCPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-overfit Visual-BC")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--architecture",
        choices=("standard", "spatial_softmax"),
        default="standard",
        help="standard keeps the existing Visual-BC model",
    )
    parser.add_argument("--spatial-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument(
        "--history-size",
        type=int,
        default=1,
        help="Use History-Visual-BC when greater than one",
    )
    parser.add_argument(
        "--visual-fusion",
        choices=VISUAL_BC_FUSION_MODES,
        default="spatial_attention",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        choices=tuple(CAMERA_OBSERVATION_KEYS),
        default=["agentview"],
    )
    parser.add_argument("--no-vertical-flip", action="store_true")
    parser.add_argument("--pretrained-visual", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--required-reduction", type=float, default=0.95)
    parser.add_argument("--max-final-loss", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resize(image, size):
    if image.shape[-2:] == (size, size):
        return image
    return F.interpolate(
        image,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def gradient_norm(parameters):
    total = torch.zeros((), dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum().cpu()
    return float(torch.sqrt(total))


def main():
    args = parse_args()
    if min(args.batch_size, args.steps, args.log_every) <= 0:
        raise ValueError("Batch size, steps, and log interval must be positive")
    if args.learning_rate <= 0 or args.image_size < 32:
        raise ValueError("Invalid learning rate or image size")
    if args.history_size <= 0:
        raise ValueError("--history-size must be positive")
    if args.architecture == "spatial_softmax" and args.history_size != 1:
        raise ValueError("spatial_softmax supports no temporal history")
    if args.spatial_channels <= 0 or args.spatial_channels % 8:
        raise ValueError("--spatial-channels must be a positive multiple of 8")
    if args.camera_names[0] != "agentview":
        raise ValueError("--camera-names must start with agentview")
    set_seed(args.seed)

    normalizer = PolicyNormalizer.from_json(args.normalization)
    normalizer.validate_split(args.split)
    dataset = ManipulationDataset(
        dataset_root=args.dataset,
        split_manifest=args.split,
        mode="bc",
        vertical_flip=not args.no_vertical_flip,
        cache_size=1,
        proprio_keys=normalizer.proprio_keys,
        camera_keys=tuple(
            CAMERA_OBSERVATION_KEYS[name] for name in args.camera_names
        ),
        history_size=args.history_size,
    )
    episode = dataset.episode_sample_ranges[0]
    indices = np.linspace(
        episode.start,
        episode.stop - 1,
        num=args.batch_size,
        dtype=int,
    )
    batch = default_collate([dataset[int(index)] for index in indices])
    inputs = {"image": resize(batch["image"], args.image_size)}
    if args.history_size > 1:
        proprio_mask = batch["proprio_history_mask"]
        action_mask = batch["action_history_mask"]
        inputs.update(
            {
                "proprio_history": normalizer.normalize_proprio(
                    batch["proprio_history"]
                )
                * proprio_mask.unsqueeze(-1),
                "action_history": normalizer.normalize_action(
                    batch["action_history"]
                )
                * action_mask.unsqueeze(-1),
                "proprio_history_mask": proprio_mask,
                "action_history_mask": action_mask,
            }
        )
    else:
        inputs["proprio"] = normalizer.normalize_proprio(batch["proprio"])
    if "eye_in_hand_image" in batch:
        inputs["eye_in_hand_image"] = resize(
            batch["eye_in_hand_image"], args.image_size
        )
    target = normalizer.normalize_action(batch["action"])

    if args.architecture == "spatial_softmax":
        model_class = SpatialSoftmaxVisualBCPolicy
        model_kwargs = {
            "camera_names": args.camera_names,
            "action_dim": normalizer.action_dim,
            "proprio_dim": normalizer.proprio_dim,
            "spatial_channels": args.spatial_channels,
            "hidden_dim": args.hidden_dim,
            "pretrained_visual": args.pretrained_visual,
            "freeze_visual_backbone": True,
            "dropout": 0.0,
        }
    else:
        model_class = HistoryVisualBCPolicy if args.history_size > 1 else VisualBCPolicy
        model_kwargs = {
            "visual_fusion": args.visual_fusion,
            "camera_names": args.camera_names,
            "action_dim": normalizer.action_dim,
            "proprio_dim": normalizer.proprio_dim,
            "pretrained_visual": args.pretrained_visual,
            "freeze_visual_backbone": True,
            "dropout": 0.0,
        }
        if args.history_size > 1:
            model_kwargs["history_size"] = args.history_size
    model = model_class(**model_kwargs)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    losses = []
    maximum_gradient_norm = 0.0
    model.train()
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**inputs)["action"]
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        losses.append(float(loss.detach()))
        if step == args.steps:
            break
        loss.backward()
        current_norm = gradient_norm(parameters)
        maximum_gradient_norm = max(maximum_gradient_norm, current_norm)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            print(
                f"step {step + 1:4d}/{args.steps}: "
                f"loss={losses[-1]:.8f}, grad_norm={current_norm:.4f}"
            )

    model.eval()
    with torch.no_grad():
        expected = model(**inputs)["action"]
    model_kwargs["pretrained_visual"] = False
    restored = model_class(**model_kwargs)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored.eval()
    with torch.no_grad():
        actual = restored(**inputs)["action"]
    if not torch.equal(expected, actual):
        raise AssertionError("Checkpoint round trip changed Visual-BC output")

    initial = losses[0]
    final = losses[-1]
    reduction = 1.0 - final / max(initial, 1e-12)
    passed = reduction >= args.required_reduction and final <= args.max_final_loss
    report = {
        "dataset": str(args.dataset),
        "split": str(args.split),
        "camera_names": list(args.camera_names),
        "architecture": args.architecture,
        "visual_fusion": args.visual_fusion,
        "spatial_channels": args.spatial_channels,
        "hidden_dim": args.hidden_dim,
        "history_size": args.history_size,
        "batch_size": args.batch_size,
        "timesteps": batch["timestep"].tolist(),
        "initial_loss": initial,
        "final_loss": final,
        "loss_reduction": reduction,
        "maximum_gradient_norm": maximum_gradient_norm,
        "passed": passed,
        "losses": losses,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"initial={initial:.8f}, final={final:.8f}, "
        f"reduction={reduction:.2%}, passed={passed}"
    )
    print(f"Report: {report_path}")
    if not passed:
        raise SystemExit("Visual-BC mechanical tiny overfit: FAIL")
    print("Visual-BC mechanical tiny overfit: PASS")


if __name__ == "__main__":
    main()
