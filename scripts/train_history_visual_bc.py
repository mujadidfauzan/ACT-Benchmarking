"""Train fixed-target Visual-BC with proprioception and action history."""

import argparse
from pathlib import Path

import torch

from data.manipulation_dataset import CAMERA_OBSERVATION_KEYS
from data.normalization import PolicyNormalizer
from models.visual_bc import (
    VISUAL_BC_FUSION_MODES,
    HistoryVisualBCPolicy,
)
from scripts.train_language_bc import (
    build_loaders,
    format_metrics,
    json_value,
    resolve_device,
    save_checkpoint,
    save_json,
)
from scripts.train_visual_bc import run_epoch, set_seed, validate_args


def parse_args():
    parser = argparse.ArgumentParser(description="Train fixed-target History-Visual-BC")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--val-split", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--fixed-target-color", default="red")
    parser.add_argument("--history-size", type=int, default=8)
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
    parser.add_argument("--epochs", type=int, default=80)
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
        help="Balance training batches across expert phases",
    )
    parser.add_argument(
        "--batches-per-phase",
        type=int,
        help="Batches sampled for each phase when --phase-balanced is active",
    )
    parser.add_argument(
        "--batches-per-task",
        type=int,
        help="Batches sampled for each task when phase balancing is disabled",
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrained-visual", action="store_true")
    parser.add_argument("--no-vertical-flip", action="store_true")
    return parser.parse_args()


def checkpoint_payload(args, model, optimizer, epoch, global_step, best, history):
    return {
        "format_version": 1,
        "policy_type": "history_visual_bc",
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "policy_type": "history_visual_bc",
            "visual_fusion": model.visual_fusion,
            "camera_names": list(model.camera_names),
            "proprio_dim": model.temporal_encoder.proprio_dim,
            "action_dim": model.action_dim,
            "history_size": model.history_size,
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
    if args.history_size <= 1:
        raise ValueError("--history-size must be greater than one")
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
    if train_dataset.history_size != args.history_size:
        raise ValueError("Dataset history size does not match command")
    if train_dataset.proprio_dim != normalizer.proprio_dim:
        raise ValueError("Dataset and normalizer proprio dimensions do not match")
    if train_dataset.action_dim != normalizer.action_dim:
        raise ValueError("Dataset and normalizer action dimensions do not match")

    model = HistoryVisualBCPolicy(
        history_size=args.history_size,
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
        if checkpoint.get("policy_type") != "history_visual_bc":
            raise ValueError("Resume checkpoint is not History-Visual-BC")
        config = checkpoint["model_config"]
        expected = (
            config.get("visual_fusion"),
            tuple(config.get("camera_names", [])),
            config.get("fixed_target_color"),
            int(config.get("history_size", -1)),
        )
        actual = (
            args.visual_fusion,
            tuple(args.camera_names),
            args.fixed_target_color,
            args.history_size,
        )
        if expected != actual:
            raise ValueError(f"Resume configuration mismatch: {expected} != {actual}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = list(checkpoint.get("history", []))

    print(f"Device: {device}")
    print("Policy: History-Visual-BC (no language)")
    print(f"History size: {args.history_size}")
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
