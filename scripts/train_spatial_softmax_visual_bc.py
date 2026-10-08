"""Train two-view spatial-softmax Visual-BC and measure closed-loop progress."""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.manipulation_dataset import (
    CAMERA_OBSERVATION_KEYS,
    ManipulationDataset,
    create_dataloader,
)
from data.normalization import PolicyNormalizer
from models.spatial_softmax_visual_bc import SpatialSoftmaxVisualBCPolicy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def json_value(value):
    return str(value) if isinstance(value, Path) else value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class LossMeter:
    def __init__(self):
        self.total_squared_error = 0.0
        self.total_elements = 0
        self.task_squared_error = defaultdict(float)
        self.task_elements = defaultdict(int)
        self.phase_squared_error = defaultdict(float)
        self.phase_elements = defaultdict(int)

    def update(self, prediction, target, tasks, phases):
        squared_error = (prediction.detach() - target.detach()).square()
        self.total_squared_error += float(squared_error.sum().cpu())
        self.total_elements += squared_error.numel()
        for task in set(tasks):
            indices = [index for index, value in enumerate(tasks) if value == task]
            error = squared_error[indices]
            self.task_squared_error[task] += float(error.sum().cpu())
            self.task_elements[task] += error.numel()
        for phase in set(phases):
            indices = [index for index, value in enumerate(phases) if value == phase]
            error = squared_error[indices]
            self.phase_squared_error[phase] += float(error.sum().cpu())
            self.phase_elements[phase] += error.numel()

    def result(self):
        if not self.total_elements:
            raise RuntimeError("No batches were processed")
        return {
            "loss": self.total_squared_error / self.total_elements,
            "loss_per_task": {
                key: self.task_squared_error[key] / self.task_elements[key]
                for key in sorted(self.task_elements)
            },
            "loss_per_phase": {
                key: self.phase_squared_error[key] / self.phase_elements[key]
                for key in sorted(self.phase_elements)
            },
        }


def prepare_batch(batch, normalizer, device, image_size):
    def resize(image):
        if image.shape[-2:] == (image_size, image_size):
            return image
        return F.interpolate(
            image, size=(image_size, image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )

    inputs = {"image": resize(batch["image"].to(device, non_blocking=True))}
    inputs["proprio"] = normalizer.normalize_proprio(
        batch["proprio"].to(device, non_blocking=True)
    )
    if "eye_in_hand_image" in batch:
        inputs["eye_in_hand_image"] = resize(
            batch["eye_in_hand_image"].to(device, non_blocking=True)
        )
    target = normalizer.normalize_action(batch["action"].to(device, non_blocking=True))
    return inputs, target, list(batch["task"]), list(batch["phase"])


def run_epoch(
    model, loader, normalizer, device, image_size, optimizer=None,
    gradient_clip=0.0, max_batches=None, log_every=25,
):
    training = optimizer is not None
    model.train(training)
    meter = LossMeter()
    started = time.monotonic()
    batches = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            inputs, target, tasks, phases = prepare_batch(
                batch, normalizer, device, image_size
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(**inputs)["action"]
            loss = F.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
            if training:
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        gradient_clip,
                    )
                optimizer.step()
            meter.update(prediction, target, tasks, phases)
            batches += 1
            if training and batch_index % log_every == 0:
                print(f"  batch {batch_index}: loss={float(loss.detach()):.6f}")
    metrics = meter.result()
    metrics["batches"] = batches
    metrics["seconds"] = time.monotonic() - started
    return metrics


def build_loaders(args, proprio_keys):
    common = {
        "dataset_root": args.dataset,
        "mode": "bc",
        "vertical_flip": not args.no_vertical_flip,
        "cache_size": args.cache_size,
        "proprio_keys": proprio_keys,
        "camera_keys": tuple(CAMERA_OBSERVATION_KEYS[name] for name in args.camera_names),
    }
    train_dataset = ManipulationDataset(split_manifest=args.train_split, **common)
    val_dataset = ManipulationDataset(split_manifest=args.val_split, **common)
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=not args.phase_balanced,
        seed=args.seed, task_balanced=not args.phase_balanced,
        batches_per_task=args.batches_per_task, phase_balanced=args.phase_balanced,
        batches_per_phase=args.batches_per_phase,
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, seed=args.seed,
        task_balanced=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def format_metrics(name, metrics):
    tasks = ", ".join(
        f"{task}={loss:.6f}" for task, loss in metrics["loss_per_task"].items()
    )
    phases = ", ".join(
        f"{phase}={loss:.6f}" for phase, loss in metrics["loss_per_phase"].items()
    )
    output = (
        f"{name}: loss={metrics['loss']:.6f} ({tasks}), "
        f"batches={metrics['batches']}, time={metrics['seconds']:.1f}s"
    )
    return f"{output}\n    phases: {phases}" if phases else output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train spatial-softmax Visual-BC for fixed-target Pick"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--fixed-target-color", default="red")
    parser.add_argument(
        "--camera-names",
        nargs="+",
        choices=tuple(CAMERA_OBSERVATION_KEYS),
        default=["agentview", "robot0_eye_in_hand"],
    )
    parser.add_argument("--spatial-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--phase-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Balance optimizer batches across expert phases",
    )
    parser.add_argument("--batches-per-phase", type=int)
    parser.add_argument("--batches-per-task", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--pretrained-visual",
        action="store_true",
        help="Use ImageNet ResNet18 weights; may download them on first use",
    )
    parser.add_argument(
        "--no-vertical-flip",
        action="store_true",
        help="Use for OpenCV-convention datasets such as pick_fixed_red_v01",
    )
    parser.add_argument(
        "--rollout-eval-every",
        type=int,
        default=20,
        help="Run deterministic closed-loop evaluation every N epochs; 0 disables it",
    )
    parser.add_argument("--rollout-eval-episodes", type=int, default=20)
    parser.add_argument("--rollout-eval-seed-offset", type=int, default=0)
    parser.add_argument("--rollout-eval-max-steps", type=int, default=400)
    parser.add_argument("--rollout-eval-success-hold-steps", type=int, default=5)
    return parser.parse_args()


def validate_spatial_args(args):
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "image_size": args.image_size,
        "log_every": args.log_every,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.weight_decay < 0 or args.gradient_clip < 0:
        raise ValueError("Weight decay and gradient clip cannot be negative")
    for name in ("num_workers", "cache_size"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "batches_per_task", "batches_per_phase", "max_train_batches",
        "max_val_batches",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.phase_balanced and args.batches_per_task is not None:
        raise ValueError("--batches-per-task cannot be combined with --phase-balanced")
    if not args.phase_balanced and args.batches_per_phase is not None:
        raise ValueError("--batches-per-phase requires --phase-balanced")
    if not args.camera_names or args.camera_names[0] != "agentview":
        raise ValueError("--camera-names must start with agentview")
    if len(set(args.camera_names)) != len(args.camera_names):
        raise ValueError("--camera-names cannot contain duplicates")
    if args.spatial_channels <= 0 or args.spatial_channels % 8:
        raise ValueError("--spatial-channels must be a positive multiple of 8")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.rollout_eval_every < 0:
        raise ValueError("--rollout-eval-every cannot be negative")
    for name in (
        "rollout_eval_episodes",
        "rollout_eval_max_steps",
        "rollout_eval_success_hold_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def checkpoint_payload(
    args,
    model,
    optimizer,
    epoch,
    global_step,
    best_val_loss,
    best_rollout_success_rate,
    history,
    rollout_evaluations,
):
    return {
        "format_version": 1,
        "policy_type": model.policy_type,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_rollout_success_rate": best_rollout_success_rate,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "policy_type": model.policy_type,
            "visual_fusion": model.visual_fusion,
            "camera_names": list(model.camera_names),
            "proprio_dim": model.proprio_dim,
            "action_dim": model.action_dim,
            "spatial_channels": model.spatial_channels,
            "hidden_dim": model.hidden_dim,
            "dropout": model.dropout,
            "fixed_target_color": args.fixed_target_color,
            "pretrained_visual": args.pretrained_visual,
            "freeze_visual_backbone": True,
        },
        "training_config": {
            key: json_value(value) for key, value in vars(args).items()
        },
        "history": history,
        "rollout_evaluations": rollout_evaluations,
    }


def resume_training(args, model, optimizer, device):
    if args.resume is None:
        return 1, 0, float("inf"), -1.0, [], []
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    if checkpoint.get("policy_type") != model.policy_type:
        raise ValueError("Resume checkpoint is not a spatial-softmax Visual-BC checkpoint")
    config = checkpoint["model_config"]
    expected = (
        tuple(config.get("camera_names", [])),
        config.get("fixed_target_color"),
        int(config.get("spatial_channels", -1)),
        int(config.get("hidden_dim", -1)),
        int(config.get("proprio_dim", -1)),
    )
    actual = (
        tuple(args.camera_names),
        args.fixed_target_color,
        args.spatial_channels,
        args.hidden_dim,
        model.proprio_dim,
    )
    if expected != actual:
        raise ValueError(f"Resume configuration mismatch: {expected} != {actual}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
        float(checkpoint.get("best_rollout_success_rate", -1.0)),
        list(checkpoint.get("history", [])),
        list(checkpoint.get("rollout_evaluations", [])),
    )


def main():
    args = parse_args()
    validate_spatial_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {key: json_value(value) for key, value in vars(args).items()},
    )

    normalizer = PolicyNormalizer.from_json(args.normalization).to(device)
    normalizer.validate_split(args.train_split)
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        args, normalizer.proprio_keys
    )
    if train_dataset.proprio_dim != normalizer.proprio_dim:
        raise ValueError("Dataset and normalizer proprio dimensions do not match")
    if train_dataset.action_dim != normalizer.action_dim:
        raise ValueError("Dataset and normalizer action dimensions do not match")

    model = SpatialSoftmaxVisualBCPolicy(
        camera_names=args.camera_names,
        action_dim=normalizer.action_dim,
        proprio_dim=normalizer.proprio_dim,
        spatial_channels=args.spatial_channels,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pretrained_visual=args.pretrained_visual,
        freeze_visual_backbone=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    (
        start_epoch,
        global_step,
        best_val_loss,
        best_rollout_success_rate,
        history,
        rollout_evaluations,
    ) = resume_training(args, model, optimizer, device)

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Device: {device}")
    print("Policy: Spatial-softmax Visual-BC (no language, no history)")
    print(f"Fixed target color: {args.fixed_target_color}")
    print(f"Cameras: {', '.join(args.camera_names)}")
    print(f"Proprio dimension: {normalizer.proprio_dim}")
    print(
        f"Spatial channels per camera: {args.spatial_channels}; "
        f"MLP: {args.hidden_dim} -> {args.hidden_dim}; "
        f"trainable parameters: {trainable_parameters:,}"
    )
    print(
        f"Samples: train={len(train_dataset)}, val={len(val_dataset)}; "
        f"loader batches: train={len(train_loader)}, val={len(val_loader)}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model, train_loader, normalizer, device, args.image_size,
            optimizer=optimizer, gradient_clip=args.gradient_clip,
            max_batches=args.max_train_batches, log_every=args.log_every,
        )
        val_metrics = run_epoch(
            model, val_loader, normalizer, device, args.image_size,
            max_batches=args.max_val_batches, log_every=args.log_every,
        )
        global_step += train_metrics["batches"]
        print(format_metrics("  train", train_metrics))
        print(format_metrics("  val", val_metrics))
        history.append({
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        })
        improved_val = val_metrics["loss"] < best_val_loss
        if improved_val:
            best_val_loss = val_metrics["loss"]

        rollout_result = None
        if args.rollout_eval_every and epoch % args.rollout_eval_every == 0:
            # Keep the offline trainer usable in environments without Robosuite.
            from scripts.evaluate_visual_bc import run_rollout_evaluation

            print(
                f"  rollout evaluation: {args.rollout_eval_episodes} episodes "
                f"at epoch {epoch}"
            )
            rollout_result = run_rollout_evaluation(
                model,
                normalizer,
                image_size=args.image_size,
                device=device,
                target_color=args.fixed_target_color,
                episodes=args.rollout_eval_episodes,
                seed_offset=args.rollout_eval_seed_offset,
                max_steps=args.rollout_eval_max_steps,
                success_hold_steps=args.rollout_eval_success_hold_steps,
            )
            rollout_result.update({"epoch": epoch, "val_loss": val_metrics["loss"]})
            rollout_evaluations.append(rollout_result)
            print(
                f"  rollout: {rollout_result['successful']}/"
                f"{rollout_result['attempted']} "
                f"({rollout_result['success_rate']:.1%}), "
                f"failures={rollout_result['failure_counts']}"
            )

        if rollout_result is not None:
            better_rollout = rollout_result["success_rate"] > best_rollout_success_rate
            tied_rollout = math.isclose(
                rollout_result["success_rate"], best_rollout_success_rate
            )
            if better_rollout or (tied_rollout and improved_val):
                best_rollout_success_rate = rollout_result["success_rate"]
        payload = checkpoint_payload(
            args, model, optimizer, epoch, global_step, best_val_loss,
            best_rollout_success_rate, history, rollout_evaluations,
        )
        save_checkpoint(args.output_dir / "latest.pt", payload)
        if improved_val:
            save_checkpoint(args.output_dir / "best_val.pt", payload)
        if rollout_result is not None:
            if better_rollout or (tied_rollout and improved_val):
                save_checkpoint(args.output_dir / "best_rollout.pt", payload)
            save_checkpoint(args.output_dir / f"epoch_{epoch:03d}.pt", payload)
            save_json(args.output_dir / "rollout_evaluations.json", rollout_evaluations)
        save_json(args.output_dir / "metrics.json", history)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    if best_rollout_success_rate >= 0:
        print(f"Best rollout success rate: {best_rollout_success_rate:.1%}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
