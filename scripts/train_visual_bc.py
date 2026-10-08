"""Train fixed-target Visual-BC from RGB and proprioception."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.manipulation_dataset import CAMERA_OBSERVATION_KEYS
from data.normalization import PolicyNormalizer
from models.visual_bc import VISUAL_BC_FUSION_MODES, VisualBCPolicy
from scripts.train_language_bc import (
    LossMeter,
    build_loaders,
    format_metrics,
    json_value,
    resolve_device,
    save_checkpoint,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train fixed-target Visual-BC")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--fixed-target-color", default="red")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--phase-balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--batches-per-phase", type=int)
    parser.add_argument("--batches-per-task", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained-visual", action="store_true")
    parser.add_argument("--no-vertical-flip", action="store_true")
    return parser.parse_args()


def validate_args(args):
    for name in ("epochs", "batch_size", "image_size", "log_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.gradient_clip < 0:
        raise ValueError("Invalid optimizer configuration")
    for name in (
        "num_workers",
        "cache_size",
        "batches_per_phase",
        "batches_per_task",
        "max_train_batches",
        "max_val_batches",
    ):
        value = getattr(args, name)
        if value is not None and value < (1 if name.startswith("batches") or name.startswith("max") else 0):
            raise ValueError(f"Invalid --{name.replace('_', '-')}")
    if args.phase_balanced and args.batches_per_task is not None:
        raise ValueError("--batches-per-task cannot be combined with --phase-balanced")
    if args.camera_names[0] != "agentview":
        raise ValueError("--camera-names must start with agentview")
    if len(set(args.camera_names)) != len(args.camera_names):
        raise ValueError("--camera-names cannot contain duplicates")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_batch(batch, normalizer, device, image_size):
    image = batch["image"].to(device, non_blocking=True)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    inputs = {
        "image": image,
        "proprio": normalizer.normalize_proprio(
            batch["proprio"].to(device, non_blocking=True)
        ),
    }
    if "eye_in_hand_image" in batch:
        eye = batch["eye_in_hand_image"].to(device, non_blocking=True)
        if eye.shape[-2:] != (image_size, image_size):
            eye = F.interpolate(
                eye,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        inputs["eye_in_hand_image"] = eye
    target = normalizer.normalize_action(
        batch["action"].to(device, non_blocking=True)
    )
    return inputs, target, list(batch["task"]), list(batch["phase"])


def run_epoch(
    model,
    loader,
    normalizer,
    device,
    image_size,
    optimizer=None,
    gradient_clip=0.0,
    max_batches=None,
    log_every=25,
):
    import time

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
                        [p for p in model.parameters() if p.requires_grad],
                        gradient_clip,
                    )
                optimizer.step()
            meter.update(prediction, target, tasks, phases)
            batches += 1
            if training and batch_index % log_every == 0:
                print(f"  batch {batch_index}: loss={float(loss.detach()):.6f}")
    result = meter.result()
    result["batches"] = batches
    result["seconds"] = time.monotonic() - started
    return result


def checkpoint_payload(args, model, optimizer, epoch, global_step, best, history):
    return {
        "format_version": 1,
        "policy_type": "visual_bc",
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "policy_type": "visual_bc",
            "visual_fusion": model.visual_fusion,
            "camera_names": list(model.camera_names),
            "proprio_dim": model.proprio_encoder.input_dim,
            "action_dim": model.action_dim,
            "fixed_target_color": args.fixed_target_color,
            "pretrained_visual": args.pretrained_visual,
            "freeze_visual_backbone": True,
        },
        "training_config": {key: json_value(value) for key, value in vars(args).items()},
        "history": history,
    }


def main():
    args = parse_args()
    validate_args(args)
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

    model = VisualBCPolicy(
        visual_fusion=args.visual_fusion,
        camera_names=args.camera_names,
        action_dim=normalizer.action_dim,
        proprio_dim=normalizer.proprio_dim,
        pretrained_visual=args.pretrained_visual,
        freeze_visual_backbone=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history = []
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("policy_type") != "visual_bc":
            raise ValueError("Resume checkpoint is not a Visual-BC checkpoint")
        config = checkpoint["model_config"]
        expected = (
            config.get("visual_fusion"),
            tuple(config.get("camera_names", [])),
            config.get("fixed_target_color"),
        )
        actual = (args.visual_fusion, tuple(args.camera_names), args.fixed_target_color)
        if expected != actual:
            raise ValueError(f"Resume configuration mismatch: {expected} != {actual}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = list(checkpoint.get("history", []))

    print(f"Device: {device}")
    print("Policy: Visual-BC (no language)")
    print(f"Fixed target color: {args.fixed_target_color}")
    print(f"Visual fusion: {args.visual_fusion}")
    print(f"Cameras: {', '.join(args.camera_names)}")
    print(f"Proprio dimension: {normalizer.proprio_dim}")
    print(f"Phase-balanced sampling: {args.phase_balanced}")
    print(
        f"Samples: train={len(train_dataset)}, val={len(val_dataset)}; "
        f"loader batches: train={len(train_loader)}, val={len(val_loader)}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            normalizer,
            device,
            args.image_size,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
            max_batches=args.max_train_batches,
            log_every=args.log_every,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            normalizer,
            device,
            args.image_size,
            max_batches=args.max_val_batches,
            log_every=args.log_every,
        )
        global_step += train_metrics["batches"]
        print(format_metrics("  train", train_metrics))
        print(format_metrics("  val", val_metrics))
        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
        payload = checkpoint_payload(
            args, model, optimizer, epoch, global_step, best_val_loss, history
        )
        save_checkpoint(args.output_dir / "latest.pt", payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        save_json(args.output_dir / "metrics.json", history)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
