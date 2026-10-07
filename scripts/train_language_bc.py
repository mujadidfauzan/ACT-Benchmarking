"""Train Language-BC with task-balanced sampling and cached language features."""

import argparse
import json
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
from language.frozen_minilm import CachedLanguageEmbeddings, CachedTokenEmbeddings
from models.language_bc import (
    LANGUAGE_MODES,
    VISUAL_FUSION_MODES,
    LanguageBCPolicy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Language-BC")
    parser.add_argument("--dataset", type=Path, default=Path("data/demos/pilot_v01"))
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/train.jsonl"),
    )
    parser.add_argument(
        "--val-split",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/val.jsonl"),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/normalization.json"),
    )
    parser.add_argument(
        "--pooled-cache",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/language_embeddings.npz"),
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=Path(
            "data/splits/pilot_v01_semantic/language_token_embeddings.npz"
        ),
    )
    parser.add_argument("--language-mode", choices=LANGUAGE_MODES, default="pooled")
    parser.add_argument(
        "--visual-fusion",
        choices=VISUAL_FUSION_MODES,
        default="spatial_attention",
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        choices=tuple(CAMERA_OBSERVATION_KEYS),
        default=["agentview"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--batches-per-task", type=int)
    parser.add_argument(
        "--phase-balanced",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Balance optimizer batches across expert phases",
    )
    parser.add_argument("--batches-per-phase", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an explicit torch device such as cuda:0",
    )
    parser.add_argument(
        "--pretrained-visual",
        action="store_true",
        help="Use ImageNet ResNet18 weights; may download them on first use",
    )
    parser.add_argument(
        "--unfreeze-layer4-epoch",
        type=int,
        help="One-based epoch at which ResNet layer4 becomes trainable",
    )
    parser.add_argument(
        "--no-vertical-flip",
        action="store_true",
        help="Disable the correction required by pilot_v01 OpenGL images",
    )
    return parser.parse_args()


def validate_args(args):
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
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")
    for name in (
        "batches_per_task",
        "batches_per_phase",
        "max_train_batches",
        "max_val_batches",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.unfreeze_layer4_epoch is not None and args.unfreeze_layer4_epoch <= 0:
        raise ValueError("--unfreeze-layer4-epoch must be positive")
    if args.phase_balanced and args.batches_per_task is not None:
        raise ValueError(
            "--batches-per-task cannot be combined with --phase-balanced"
        )
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


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


class LanguageLookup:
    def __init__(self, mode, pooled_path, token_path):
        self.mode = mode
        if mode == "pooled":
            self.cache = CachedLanguageEmbeddings(pooled_path)
        else:
            self.cache = CachedTokenEmbeddings(token_path)

    @property
    def embedding_dim(self):
        return self.cache.embedding_dim

    def inputs(self, instructions, device):
        if self.mode == "pooled":
            return {"language_embedding": self.cache.encode(instructions, device)}
        cached = self.cache.lookup(instructions, device)
        return {
            "token_embeddings": cached["token_embeddings"],
            "attention_mask": cached["attention_mask"],
        }


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
        for task in sorted(set(tasks)):
            indices = [index for index, value in enumerate(tasks) if value == task]
            task_error = squared_error[indices]
            self.task_squared_error[task] += float(task_error.sum().cpu())
            self.task_elements[task] += task_error.numel()
        for phase in sorted(set(phases)):
            indices = [index for index, value in enumerate(phases) if value == phase]
            phase_error = squared_error[indices]
            self.phase_squared_error[phase] += float(phase_error.sum().cpu())
            self.phase_elements[phase] += phase_error.numel()

    def result(self):
        if self.total_elements == 0:
            raise RuntimeError("No batches were processed")
        return {
            "loss": self.total_squared_error / self.total_elements,
            "loss_per_task": {
                task: self.task_squared_error[task] / self.task_elements[task]
                for task in sorted(self.task_elements)
            },
            "loss_per_phase": {
                phase: self.phase_squared_error[phase] / self.phase_elements[phase]
                for phase in sorted(self.phase_elements)
            },
        }


def prepare_batch(batch, normalizer, language, device, image_size):
    image = batch["image"].to(device, non_blocking=True)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    proprio = normalizer.normalize_proprio(
        batch["proprio"].to(device, non_blocking=True)
    )
    action = normalizer.normalize_action(
        batch["action"].to(device, non_blocking=True)
    )
    instructions = list(batch["instruction"])
    model_inputs = {
        "image": image,
        "proprio": proprio,
        **language.inputs(instructions, device),
    }
    if "eye_in_hand_image" in batch:
        eye_in_hand_image = batch["eye_in_hand_image"].to(
            device, non_blocking=True
        )
        if eye_in_hand_image.shape[-2:] != (image_size, image_size):
            eye_in_hand_image = F.interpolate(
                eye_in_hand_image,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        model_inputs["eye_in_hand_image"] = eye_in_hand_image
    return (
        model_inputs,
        action,
        list(batch["task"]),
        list(batch["phase"]),
    )


def run_epoch(
    model,
    loader,
    normalizer,
    language,
    device,
    image_size,
    optimizer=None,
    gradient_clip=0.0,
    max_batches=None,
    log_every=25,
):
    training = optimizer is not None
    model.train(training)
    meter = LossMeter()
    start = time.monotonic()
    processed_batches = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_index > max_batches:
                break
            model_inputs, target, tasks, phases = prepare_batch(
                batch, normalizer, language, device, image_size
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(**model_inputs)["action"]
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
            processed_batches += 1
            if training and batch_index % log_every == 0:
                print(f"  batch {batch_index}: loss={float(loss.detach()):.6f}")

    result = meter.result()
    result["batches"] = processed_batches
    result["seconds"] = time.monotonic() - start
    return result


def build_loaders(args, proprio_keys):
    common = {
        "dataset_root": args.dataset,
        "mode": "bc",
        "vertical_flip": not args.no_vertical_flip,
        "cache_size": args.cache_size,
        "proprio_keys": proprio_keys,
        "camera_keys": tuple(
            CAMERA_OBSERVATION_KEYS[name] for name in args.camera_names
        ),
    }
    train_dataset = ManipulationDataset(
        split_manifest=args.train_split,
        **common,
    )
    val_dataset = ManipulationDataset(
        split_manifest=args.val_split,
        **common,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=not args.phase_balanced,
        seed=args.seed,
        task_balanced=not args.phase_balanced,
        batches_per_task=args.batches_per_task,
        phase_balanced=args.phase_balanced,
        batches_per_phase=args.batches_per_phase,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed,
        task_balanced=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def checkpoint_payload(
    args, model, optimizer, epoch, global_step, best_val_loss, history
):
    return {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "language_mode": args.language_mode,
            "visual_fusion": args.visual_fusion,
            "camera_names": list(model.camera_names),
            "proprio_dim": model.proprio_encoder.input_dim,
            "action_dim": model.action_dim,
            "language_input_dim": model.language_head.input_dim,
            "pretrained_visual": args.pretrained_visual,
            "freeze_visual_backbone": True,
        },
        "training_config": {
            key: json_value(value) for key, value in vars(args).items()
        },
        "history": history,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device, expected_mode):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    mode = checkpoint.get("model_config", {}).get("language_mode")
    if mode != expected_mode:
        raise ValueError(
            f"Checkpoint language mode {mode!r} does not match {expected_mode!r}"
        )
    visual_fusion = checkpoint.get("model_config", {}).get(
        "visual_fusion", "global"
    )
    return checkpoint, visual_fusion


def restore_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng = checkpoint.get("rng_state")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint


def format_metrics(name, metrics):
    tasks = ", ".join(
        f"{task}={loss:.6f}"
        for task, loss in metrics["loss_per_task"].items()
    )
    phases = ", ".join(
        f"{phase}={loss:.6f}"
        for phase, loss in metrics["loss_per_phase"].items()
    )
    return (
        f"{name}: loss={metrics['loss']:.6f} ({tasks}), "
        f"batches={metrics['batches']}, time={metrics['seconds']:.1f}s\n"
        f"    phases: {phases}"
    )


def build_optimizer(model, learning_rate, weight_decay, layer4_unfrozen=False):
    layer4_parameters = list(model.visual_encoder.backbone.layer4.parameters())
    layer4_ids = {id(parameter) for parameter in layer4_parameters}
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in layer4_ids
    ]
    optimizer = torch.optim.AdamW(
        base_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    if layer4_unfrozen:
        optimizer.add_param_group(
            {"params": [parameter for parameter in layer4_parameters if parameter.requires_grad]}
        )
    return optimizer


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
    language = LanguageLookup(
        args.language_mode, args.pooled_cache, args.token_cache
    )
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(
        args, normalizer.proprio_keys
    )
    if train_dataset.proprio_dim != normalizer.proprio_dim:
        raise ValueError("Dataset and normalizer proprio dimensions do not match")
    if train_dataset.action_dim != normalizer.action_dim:
        raise ValueError("Dataset and normalizer action dimensions do not match")

    model = LanguageBCPolicy(
        language_mode=args.language_mode,
        visual_fusion=args.visual_fusion,
        camera_names=args.camera_names,
        action_dim=normalizer.action_dim,
        proprio_dim=normalizer.proprio_dim,
        language_input_dim=language.embedding_dim,
        pretrained_visual=args.pretrained_visual,
        freeze_visual_backbone=True,
    ).to(device)

    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint, resumed_visual_fusion = load_checkpoint(
            args.resume, device, args.language_mode
        )
        if resumed_visual_fusion != args.visual_fusion:
            raise ValueError(
                "Checkpoint visual fusion does not match command: "
                f"{resumed_visual_fusion!r} != {args.visual_fusion!r}"
            )
        resumed_cameras = tuple(
            resume_checkpoint.get("model_config", {}).get(
                "camera_names", ["agentview"]
            )
        )
        if resumed_cameras != tuple(args.camera_names):
            raise ValueError(
                "Checkpoint cameras do not match command: "
                f"{resumed_cameras!r} != {tuple(args.camera_names)!r}"
            )
        resumed_epoch = int(resume_checkpoint["epoch"])
        if (
            args.unfreeze_layer4_epoch is not None
            and resumed_epoch >= args.unfreeze_layer4_epoch
        ):
            model.visual_encoder.unfreeze_layer4()

    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        layer4_unfrozen=model.visual_encoder._layer4_trainable,
    )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    history = []
    if resume_checkpoint is not None:
        checkpoint = restore_checkpoint(resume_checkpoint, model, optimizer)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        history = list(checkpoint.get("history", []))
        print(f"Resumed {args.resume} at epoch {start_epoch}")

    print(f"Device: {device}")
    print(f"Language mode: {args.language_mode}")
    print(f"Visual fusion: {args.visual_fusion}")
    print(f"Cameras: {', '.join(args.camera_names)}")
    print(f"Proprio dimension: {normalizer.proprio_dim}")
    print(f"Phase-balanced sampling: {args.phase_balanced}")
    print(
        f"Samples: train={len(train_dataset)}, val={len(val_dataset)}; "
        f"loader batches: train={len(train_loader)}, val={len(val_loader)}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        if (
            args.unfreeze_layer4_epoch is not None
            and epoch == args.unfreeze_layer4_epoch
        ):
            model.visual_encoder.unfreeze_layer4()
            optimizer.add_param_group(
                {
                    "params": [
                        parameter
                        for parameter in model.visual_encoder.backbone.layer4.parameters()
                        if parameter.requires_grad
                    ]
                }
            )
            print(f"Unfroze visual layer4 at epoch {epoch}")

        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            train_loader,
            normalizer,
            language,
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
            language,
            device,
            args.image_size,
            max_batches=args.max_val_batches,
            log_every=args.log_every,
        )
        global_step += train_metrics["batches"]
        print(format_metrics("  train", train_metrics))
        print(format_metrics("  val", val_metrics))

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
        payload = checkpoint_payload(
            args,
            model,
            optimizer,
            epoch,
            global_step,
            best_val_loss,
            history,
        )
        save_checkpoint(args.output_dir / "latest.pt", payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        save_json(args.output_dir / "metrics.json", history)

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
