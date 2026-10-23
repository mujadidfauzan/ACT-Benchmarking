"""Visual-BC training: complete shuffled epochs and motion MSE + gripper BCE."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data.bc_preparation import load_bc_training, prepare_bc_training, save_json
from data.manipulation_dataset import (
    CAMERA_OBSERVATION_KEYS,
    EpisodeSequenceDataset,
    ManipulationDataset,
)
from data.normalization import PolicyNormalizer
from models.visual_bc import ENCODER_CONFIG, VisualBCPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Visual-BC from automatically prepared demos"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir", type=Path, required=True
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Frames per batch (default 128), or temporal windows (default 8)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int,
        help="Physical batches per optimizer update (default 1; restored on resume)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use a GRU with RGB/proprio history and previous gripper input",
    )
    parser.add_argument(
        "--sequence-length", type=int, help="Temporal window length (default 16)"
    )
    parser.add_argument(
        "--sequence-stride", type=int, help="Temporal window stride (default 8)"
    )
    parser.add_argument(
        "--visual-pooling", choices=("avg", "spatial_softmax"),
        help="CNN pooling (default avg; restored from checkpoint on resume)",
    )
    parser.add_argument(
        "--visual-backbone", choices=("cnn", "resnet18"),
        help="Vision backbone trained from scratch (default cnn; restored on resume)",
    )
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--rollout-eval-every",
        type=int,
        default=10,
        help="Rollout interval in epochs; 0 disables",
    )
    parser.add_argument(
        "--rollout-eval-episodes",
        type=int,
        default=10,
        help="Episodes per rollout seed",
    )
    parser.add_argument("--rollout-eval-num-seeds", type=int, default=1)
    parser.add_argument("--rollout-eval-seed", type=int, default=1000)
    parser.add_argument(
        "--rollout-eval-max-steps", type=int,
        help="Rollout horizon (default: Pick 500, Place 3000; restored on resume)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=7,
        help="Epochs without lower val loss; 0 disables",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if value == "auto"
        else torch.device(value)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def prepare_batch(batch, normalizer, device):
    inputs = {}
    for key in ("image", "eye_in_hand_image"):
        if key in batch:
            value = batch[key].to(device)
            if value.shape[-2:] != (128, 128):
                prefix = value.shape[:-3]
                value = F.interpolate(
                    value.reshape(-1, *value.shape[-3:]),
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).reshape(*prefix, 3, 128, 128)
            inputs[key] = value
    inputs["proprio"] = normalizer.normalize_proprio(batch["proprio"].to(device))
    if "previous_gripper" in batch:
        inputs["previous_gripper"] = batch["previous_gripper"].to(device)
    action = batch["action"].to(device)
    return inputs, normalizer.normalize_action(action), (action[..., 6:7] > 0).float()


def run_epoch(model, loader, normalizer, device, optimizer=None, gradient_accumulation_steps=1):
    if gradient_accumulation_steps < 1:
        raise ValueError("Gradient accumulation steps must be positive")
    model.train(optimizer is not None)
    totals = np.zeros(4, dtype=np.float64)
    count = 0
    batches = 0
    accumulated_samples = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    def update():
        # Normalize by valid timesteps, including padded GRU windows and the tail group.
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(accumulated_samples)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            inputs, action, grip = prepare_batch(batch, normalizer, device)
            output = model(**inputs)
            mask = (
                batch["valid_mask"].to(device)
                if "valid_mask" in batch
                else torch.ones(action.shape[:-1], dtype=torch.bool, device=device)
            )
            size = int(mask.sum())
            if size == 0:
                raise ValueError("Batch contains no valid timesteps")
            xyz = (
                (output["motion"][..., :3] - action[..., :3])
                .square()
                .mean(-1)[mask]
                .mean()
            )
            rotation = (
                (output["motion"][..., 3:6] - action[..., 3:6])
                .square()
                .mean(-1)[mask]
                .mean()
                if model.predict_orientation
                else xyz.new_zeros(())
            )
            gripper = (
                F.binary_cross_entropy_with_logits(
                    output["gripper_logit"], grip, reduction="none"
                )
                .mean(-1)[mask]
                .mean()
            )
            loss = xyz + rotation + 0.2 * gripper
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BC loss")
            if optimizer is not None:
                if gradient_accumulation_steps == 1:
                    # Preserve the original numerical path for baseline runs.
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    (loss * size).backward()
                    accumulated_samples += size
                    if (batches + 1) % gradient_accumulation_steps == 0:
                        update()
                        accumulated_samples = 0
            totals += (
                np.asarray([x.detach().item() for x in (loss, xyz, rotation, gripper)])
                * size
            )
            count += size
            batches += 1
        if optimizer is not None and accumulated_samples:
            update()
    if not count:
        raise ValueError("DataLoader contains no samples")
    return {
        **dict(
            zip(
                ("loss", "xyz_mse", "rotation_mse", "gripper_bce"),
                (totals / count).tolist(),
            )
        ),
        "samples": count,
        "batches": batches,
        "optimizer_steps": ((batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
                            if optimizer is not None else 0),
    }


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_rollout(args, epoch, device):
    # A separate process keeps simulator imports and rollout RNG out of training.
    output = args.output_dir / "rollouts" / f"epoch_{epoch:04d}"
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_visual_bc",
        "--checkpoint",
        str(args.output_dir / "latest.pt"),
        "--episodes",
        str(args.rollout_eval_episodes),
        "--num-seeds",
        str(args.rollout_eval_num_seeds),
        "--seed",
        str(args.rollout_eval_seed),
        "--device",
        str(device),
        "--output-dir",
        str(output),
    ]
    if args.rollout_eval_max_steps is not None:
        command.extend(("--max-steps", str(args.rollout_eval_max_steps)))
    subprocess.run(command, check=True)
    with (output / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    rate = float(summary["overall_success_rate"])
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Invalid rollout success rate")
    return {
        "epoch": epoch,
        "success_rate": rate,
        "summary": str(output / "summary.json"),
    }


def main():
    args = parse_args()
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    saved_config = checkpoint.get("model_config", {}) if checkpoint else {}
    encoder_config = dict(saved_config.get("encoder_config", ENCODER_CONFIG))
    saved_backbone = encoder_config.get("backbone", "cnn")
    if checkpoint and args.visual_backbone is not None and args.visual_backbone != saved_backbone:
        raise ValueError("--visual-backbone differs from checkpoint; start a new run")
    args.visual_backbone = args.visual_backbone or saved_backbone
    if args.visual_backbone != "cnn":
        encoder_config["backbone"] = args.visual_backbone
    saved_pooling = encoder_config.get("pooling", "avg")
    if checkpoint and args.visual_pooling is not None and args.visual_pooling != saved_pooling:
        raise ValueError("--visual-pooling differs from checkpoint; start a new run")
    args.visual_pooling = args.visual_pooling or saved_pooling
    if args.visual_pooling != "avg":
        encoder_config["pooling"] = args.visual_pooling
    if checkpoint and args.temporal and not saved_config.get("temporal", False):
        raise ValueError("--temporal does not match the checkpoint architecture")
    args.temporal = args.temporal or saved_config.get("temporal", False)
    saved_training = checkpoint.get("training_config", {}) if checkpoint else {}
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = saved_training.get("gradient_accumulation_steps", 1)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation steps must be positive")
    if args.temporal:
        for name, default in (("sequence_length", 16), ("sequence_stride", 8)):
            value = getattr(args, name)
            saved = saved_training.get(name, default)
            if checkpoint and value is not None and value != saved:
                raise ValueError(f"--{name.replace('_', '-')} differs from checkpoint")
            setattr(args, name, saved if value is None else value)
        if not 1 <= args.sequence_stride <= args.sequence_length:
            raise ValueError("Need 1 <= sequence-stride <= sequence-length")
    elif args.sequence_length is not None or args.sequence_stride is not None:
        raise ValueError("Sequence options require --temporal")
    args.batch_size = (
        args.batch_size
        if args.batch_size is not None
        else (8 if args.temporal else 128)
    )
    if min(args.epochs, args.batch_size, args.learning_rate) <= 0:
        raise ValueError("Epochs, batch size, and learning rate must be positive")
    if min(args.rollout_eval_every, args.early_stopping_patience) < 0:
        raise ValueError(
            "Rollout interval and early stopping patience must be nonnegative"
        )
    if min(args.rollout_eval_episodes, args.rollout_eval_num_seeds) <= 0:
        raise ValueError("Rollout episodes and seeds must be positive")
    if args.rollout_eval_max_steps is not None and args.rollout_eval_max_steps <= 0:
        raise ValueError("Rollout max steps must be positive")
    if args.resume and args.output_dir.resolve() != args.resume.resolve().parent:
        raise ValueError("--output-dir must be the directory containing --resume")
    set_seed(args.seed)
    device = resolve_device(args.device)
    try:
        prepared, directory = load_bc_training(args.dataset, args.training_dir)
    except FileNotFoundError:
        if args.resume:
            raise
        # Old completed recordings can enter the same workflow without a preparation command.
        prepare_bc_training(args.dataset, args.training_dir)
        prepared, directory = load_bc_training(args.dataset, args.training_dir)
    if prepared["tasks"] not in (["pick"], ["place"]) or prepared["fixed_target_color"] is None:
        raise ValueError(
            "Visual-BC requires one fixed-target Pick or matching-color Place dataset"
        )
    default_horizon = 3000 if prepared["tasks"] == ["place"] else 500
    saved_horizon = saved_training.get("rollout_eval_max_steps") or default_horizon
    if checkpoint and args.rollout_eval_max_steps is not None and args.rollout_eval_max_steps != saved_horizon:
        raise ValueError("--rollout-eval-max-steps differs from checkpoint; start a new run")
    if args.rollout_eval_max_steps is None:
        args.rollout_eval_max_steps = saved_horizon
    normalizer = PolicyNormalizer.from_statistics(prepared["normalization"]).to(device)
    common = dict(
        dataset_root=args.dataset,
        proprio_keys=prepared["proprio_keys"],
        camera_keys=[
            CAMERA_OBSERVATION_KEYS[name] for name in prepared["camera_names"]
        ],
        cache_size=16,
    )
    dataset_class = EpisodeSequenceDataset if args.temporal else ManipulationDataset
    if args.temporal:
        common.update(
            sequence_length=args.sequence_length, sequence_stride=args.sequence_stride
        )
    train = dataset_class(split_manifest=directory / "train.jsonl", **common)
    val = dataset_class(split_manifest=directory / "val.jsonl", **common)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = VisualBCPolicy(
        normalizer.proprio_dim,
        prepared["camera_names"],
        prepared["predict_orientation"],
        encoder_config=encoder_config,
        temporal=args.temporal,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, best, history = 1, float("inf"), []
    stale_epochs, best_rollout, rollout_history = 0, -1.0, []
    if args.resume:
        if (
            checkpoint.get("policy_type") != model.policy_type
            or {**model.get_config(), **checkpoint["model_config"]}
            != model.get_config()
        ):
            raise ValueError("Resume architecture mismatch")
        if checkpoint["data_config"] != prepared:
            raise ValueError("Resume dataset/split/normalization mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        start_epoch = int(checkpoint["epoch"]) + 1
        best, history = float(checkpoint["best_val_loss"]), list(checkpoint["history"])
        stale_epochs = int(checkpoint.get("epochs_without_improvement", 0))
        best_rollout = float(checkpoint.get("best_rollout_success_rate", -1.0))
        rollout_history = list(checkpoint.get("rollout_history", []))
        generator.set_state(checkpoint["shuffle_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if start_epoch > args.epochs:
            raise ValueError("--epochs must be greater than the saved checkpoint epoch")
    elif any((args.output_dir / name).exists() for name in ("latest.pt", "best.pt")):
        raise FileExistsError(
            "Output already contains checkpoints; use --resume or a new output directory"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "config.json",
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "model_config": model.get_config(),
            "data_config": prepared,
        },
    )
    print(
        f"Device: {device}; policy: {model.policy_type}; cameras: {model.camera_names}"
    )
    print(
        f"Proprio: {model.proprio_dim}D; orientation: {model.predict_orientation}; head: {model.hidden_dims}"
    )
    print(f"Visual backbone: {model.vision.backbone}; pooling: {model.vision.pooling}; rollout max steps: {args.rollout_eval_max_steps}")
    unit = "windows" if args.temporal else "frames"
    print(
        f"Train: {len(train.episodes)} episodes/{len(train)} {unit}; val: {len(val.episodes)} episodes/{len(val)} {unit}"
    )
    if args.temporal:
        print(
            f"GRU: {model.temporal_hidden_dim}; sequence length={args.sequence_length}, stride={args.sequence_stride}"
        )
    print(
        f"Batch size: {args.batch_size}; complete shuffled epochs; AdamW LR={args.learning_rate}, weight_decay=0.01"
    )
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}; windows/frames per full update: {args.batch_size * args.gradient_accumulation_steps}")
    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.early_stopping_patience
            and stale_epochs >= args.early_stopping_patience
        ):
            print(
                "Early stopping patience already reached; increase patience or set it to 0 to continue"
            )
            break
        tr = run_epoch(model, train_loader, normalizer, device, optimizer,
                       gradient_accumulation_steps=args.gradient_accumulation_steps)
        va = run_epoch(model, val_loader, normalizer, device)
        improved = va["loss"] < best
        best = min(best, va["loss"])
        stale_epochs = 0 if improved else stale_epochs + 1
        history.append({"epoch": epoch, "train": tr, "val": va})
        payload = {
            "format_version": 1,
            "policy_type": model.policy_type,
            "model_config": model.get_config(),
            "data_config": prepared,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best,
            "history": history,
            "epochs_without_improvement": stale_epochs,
            "best_rollout_success_rate": best_rollout,
            "rollout_history": rollout_history,
            "shuffle_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "training_config": {
                "image_size": 128,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "visual_backbone": args.visual_backbone,
                "visual_pooling": args.visual_pooling,
                "seed": args.seed,
                "sequence_length": args.sequence_length,
                "sequence_stride": args.sequence_stride,
                **{
                    key: value
                    for key, value in vars(args).items()
                    if key.startswith("rollout_eval_")
                    or key == "early_stopping_patience"
                },
            },
        }
        save_checkpoint(args.output_dir / "latest.pt", payload)
        if args.rollout_eval_every and epoch % args.rollout_eval_every == 0:
            result = run_rollout(args, epoch, device)
            rollout_history.append(result)
            rollout_improved = result["success_rate"] > best_rollout
            best_rollout = max(best_rollout, result["success_rate"])
            payload.update(
                best_rollout_success_rate=best_rollout, rollout_history=rollout_history
            )
            if rollout_improved:
                save_checkpoint(args.output_dir / "best_rollout.pt", payload)
            save_checkpoint(args.output_dir / "latest.pt", payload)
            save_json(args.output_dir / "rollout_evaluations.json", rollout_history)
            print(f"Rollout: {result['success_rate']:.1%}; best={best_rollout:.1%}")
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        save_json(args.output_dir / "metrics.json", history)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train={tr['loss']:.5f} val={va['loss']:.5f} "
            f"(xyz={va['xyz_mse']:.5f}, rotation={va['rotation_mse']:.5f}, gripper={va['gripper_bce']:.5f}) best={best:.5f}"
        )
        if (
            args.early_stopping_patience
            and stale_epochs >= args.early_stopping_patience
        ):
            print(
                f"Early stopping: {stale_epochs} epochs without validation improvement"
            )
            break
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}; val loss={best:.6f}")


if __name__ == "__main__":
    main()
