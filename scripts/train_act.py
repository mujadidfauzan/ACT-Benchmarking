"""Train ACT on native HDF5 demos: masked L1 + KL, exact resume, rollout selection."""

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from data.bc_preparation import load_bc_training, prepare_bc_training, save_json
from data.normalization import PolicyNormalizer
from models.act import ACTPolicy
from models.act.dataset import ACTDataset
from scripts.train_robomimic_bc_rnn import WindowSampler, read_batch, validation_indices
from scripts.train_visual_bc import resolve_device, save_checkpoint, set_seed

DEFAULTS = dict(
    batch_size=8,
    learning_rate=1e-4,
    backbone_learning_rate=1e-4,
    weight_decay=1e-4,
    chunk_size=32,
    hidden_dim=512,
    num_heads=8,
    encoder_layers=4,
    decoder_layers=7,
    feedforward_dim=3200,
    latent_dim=32,
    dropout=0.1,
    image_size=128,
    kl_weight=10.0,
    gradient_clip=1.0,
    seed=123,
    checkpoint_every=500,
    validation_batches=50,
    rollout_eval_every=5000,
    rollout_eval_episodes=10,
    rollout_eval_num_seeds=1,
    rollout_eval_seed=1000,
    rollout_eval_max_steps=300,
    execution_steps=1,
    ensemble_decay=0.01,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=100000,
        help="Total optimizer updates, including updates before resume",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        help="Optional local torchvision ResNet18 state_dict; default trains from scratch, no downloads",
    )
    parser.add_argument(
        "--temporal-ensemble",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rollout mode saved in checkpoint; default disabled",
    )
    for key, value in DEFAULTS.items():
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=float if isinstance(value, float) else int,
            help=f"Default {value}; restored on resume",
        )
    return parser.parse_args()


def action_losses(prediction, target, is_pad, mu=None, logvar=None, kl_weight=10.0):
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 7:
        raise ValueError("Expected matching predicted and target action chunks [B,K,7]")
    if is_pad.shape != target.shape[:2] or is_pad.dtype != torch.bool:
        raise ValueError("Expected bool padding mask [B,K]")
    valid = ~is_pad
    if not valid.any():
        raise ValueError("No valid action targets")
    # Index before subtraction: even invalid/padded NaNs cannot affect loss.
    error = (prediction[valid] - target[valid]).abs()
    l1 = error.mean()
    kl = (
        l1.new_zeros(())
        if mu is None
        else (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    )
    return dict(
        loss=l1 + kl_weight * kl,
        l1=l1,
        kl=kl,
        xyz_l1=error[:, :3].mean(),
        rotation_l1=error[:, 3:6].mean(),
        gripper_l1=error[:, 6].mean(),
    )


def run_batches(
    model,
    normalizer,
    dataset,
    batches,
    device,
    *,
    kl_weight,
    optimizer=None,
    gradient_clip=1.0,
):
    training = optimizer is not None
    model.train(training)
    totals, examples, valid_steps = {}, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for indices in batches:
            batch = read_batch(dataset, indices, device)
            target = normalizer.normalize_action(batch.pop("action"))
            padding = batch.pop("is_pad")
            batch["proprio"] = normalizer.normalize_proprio(batch["proprio"])
            output = model(
                **batch, actions=target, is_pad=padding, sample_posterior=training
            )
            losses = action_losses(
                output["action"],
                target,
                padding,
                output["mu"],
                output["logvar"],
                kl_weight,
            )
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite ACT reconstruction/KL loss")
            count = len(indices)
            # L1 statistics are weighted by real target timesteps, KL by samples.
            valid = int((~padding).sum())
            values = {
                key: value.detach().item()
                for key, value in losses.items()
                if key != "loss"
            }
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                values["gradient_norm"] = float(norm)
            else:
                # Deployment diagnostic: no demonstration actions enter the model.
                prior = action_losses(model(**batch)["action"], target, padding)
                values.update(
                    {
                        "prior_" + k: v.item()
                        for k, v in prior.items()
                        if k not in ("loss", "kl")
                    }
                )
            for key, value in values.items():
                weight = count if key in ("kl", "gradient_norm") else valid
                totals[key] = totals.get(key, 0.0) + value * weight
            examples += count
            valid_steps += valid
    if not examples:
        raise ValueError("Empty ACT training/validation batches")
    metrics = {
        k: v / (examples if k in ("kl", "gradient_norm") else valid_steps)
        for k, v in totals.items()
    }
    metrics.update(
        loss=metrics["l1"] + kl_weight * metrics["kl"],
        samples=examples,
        valid_action_steps=valid_steps,
    )
    return metrics


def run_rollout(args, updates, device):
    output = args.output_dir / "rollouts" / f"update_{updates:07d}"
    command = [
        sys.executable,
        "-m",
        "scripts.evaluate_act",
        "--checkpoint",
        str(args.output_dir / "latest.pt"),
        "--output-dir",
        str(output),
        "--device",
        str(device),
        "--episodes",
        str(args.rollout_eval_episodes),
        "--num-seeds",
        str(args.rollout_eval_num_seeds),
        "--seed",
        str(args.rollout_eval_seed),
        "--max-steps",
        str(args.rollout_eval_max_steps),
    ]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    subprocess.run(command, check=True)
    summary = json.loads((output / "summary.json").read_text())
    if any(
        k in summary["tasks"][0]["failure_counts"]
        for k in ("exception", "scene_sampling_failure")
    ):
        raise RuntimeError(f"Rollout infrastructure failed; inspect {output}")
    rate = float(summary["overall_success_rate"])
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Invalid rollout success rate")
    return dict(
        updates=updates, success_rate=rate, summary=str(output / "summary.json")
    )


def main():
    args = parse_args()
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if checkpoint and checkpoint.get("policy_type") != "act":
        raise ValueError("Resume requires an ACT checkpoint")
    saved = checkpoint["training_config"] if checkpoint else {}
    for key, default in {**DEFAULTS, "temporal_ensemble": False}.items():
        value = getattr(args, key)
        if checkpoint and value is not None and value != saved[key]:
            raise ValueError(
                f"--{key.replace('_', '-')} differs from checkpoint; start a new run"
            )
        setattr(args, key, saved.get(key, default) if value is None else value)
    positive = (
        "batch_size",
        "chunk_size",
        "hidden_dim",
        "num_heads",
        "encoder_layers",
        "decoder_layers",
        "feedforward_dim",
        "latent_dim",
        "image_size",
        "checkpoint_every",
        "execution_steps",
        "rollout_eval_episodes",
        "rollout_eval_num_seeds",
        "rollout_eval_max_steps",
    )
    if args.max_updates <= 0 or any(getattr(args, k) <= 0 for k in positive):
        raise ValueError("Update, network, batch, and rollout sizes must be positive")
    for key in ("learning_rate", "backbone_learning_rate", "gradient_clip"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if any(
        not math.isfinite(getattr(args, k)) or getattr(args, k) < 0
        for k in ("kl_weight", "weight_decay", "ensemble_decay")
    ):
        raise ValueError(
            "KL, weight decay, and ensemble decay must be finite and nonnegative"
        )
    if min(args.validation_batches, args.rollout_eval_every) < 0:
        raise ValueError(
            "Validation batch limit and rollout interval must be nonnegative"
        )
    if args.resume and (
        args.resume.resolve().parent != args.output_dir.resolve()
        or args.backbone_weights
    ):
        raise ValueError(
            "Resume in the checkpoint's output directory, without --backbone-weights"
        )
    if not checkpoint and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory or --resume")
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Loading dataset from {args.dataset} ...", flush=True)
    try:
        prepared, directory = load_bc_training(args.dataset, args.training_dir)
    except FileNotFoundError:
        if checkpoint:
            raise
        print("No prepared split found; running prepare_bc_training (may take a minute) ...", flush=True)
        prepare_bc_training(args.dataset, args.training_dir)
        prepared, directory = load_bc_training(args.dataset, args.training_dir)
    print(f"Dataset loaded: {prepared['tasks']} | {prepared.get('train_episodes', '?')} train / {prepared.get('val_episodes', '?')} val episodes", flush=True)
    if (
        prepared["tasks"] not in (["pick"], ["place"])
        or prepared["fixed_target_color"] is None
    ):
        raise ValueError("ACT supports fixed-target Pick or Place datasets")
    if checkpoint and checkpoint["data_config"] != prepared:
        raise ValueError(
            "Resume dataset, normalization, or train/validation split differs"
        )
    normalizer = PolicyNormalizer.from_statistics(prepared["normalization"]).to(device)
    model_keys = (
        "chunk_size",
        "hidden_dim",
        "num_heads",
        "encoder_layers",
        "decoder_layers",
        "feedforward_dim",
        "latent_dim",
        "dropout",
        "image_size",
        "execution_steps",
        "temporal_ensemble",
        "ensemble_decay",
    )
    print(f"Building ACTPolicy on {device} ...", flush=True)
    model = ACTPolicy(
        proprio_dim=normalizer.proprio_dim,
        camera_names=prepared["camera_names"],
        predict_orientation=prepared["predict_orientation"],
        **{k: getattr(args, k) for k in model_keys},
    ).to(device)
    if args.backbone_weights:
        model.backbone.load_local_weights(args.backbone_weights)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if id(p) not in backbone_ids],
                "lr": args.learning_rate,
            },
            {"params": model.backbone.parameters(), "lr": args.backbone_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    train = ACTDataset(
        args.dataset,
        directory / "train.jsonl",
        model.camera_names,
        normalizer.proprio_keys,
        model.chunk_size,
    )
    val = ACTDataset(
        args.dataset,
        directory / "val.jsonl",
        model.camera_names,
        normalizer.proprio_keys,
        model.chunk_size,
    )
    try:
        sampler = WindowSampler(len(train), args.batch_size, args.seed)
        updates, best, best_rollout, history, rollout_history = (
            0,
            float("inf"),
            -1.0,
            [],
            [],
        )
        if checkpoint:
            if checkpoint["model_config"] != model.get_config():
                raise ValueError("Resume model configuration differs")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            sampler.load_state_dict(checkpoint["sampler_state"])
            updates, best = checkpoint["updates"], checkpoint["best_val_loss"]
            best_rollout = checkpoint["best_rollout_success_rate"]
            history, rollout_history = (
                checkpoint["history"],
                checkpoint["rollout_history"],
            )
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint["cuda_rng_states"] is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
        if updates >= args.max_updates:
            raise ValueError("--max-updates must exceed checkpoint updates")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        training_config = {
            k: getattr(args, k) for k in (*DEFAULTS, "temporal_ensemble")
        }
        preprocessing = dict(
            proprio_keys=list(normalizer.proprio_keys),
            normalization="embedded_train_split_statistics",
            action_space="7d_OSC_delta_translation_rotation_continuous_gripper",
            image_normalization="ImageNet",
            image_size=model.image_size,
            backbone_initialization=(
                checkpoint["preprocessing"]["backbone_initialization"]
                if checkpoint
                else str(args.backbone_weights) if args.backbone_weights else "scratch"
            ),
            batch_norm="frozen_statistics",
            shared_camera_backbone=True,
            window_stride=1,
            observation_history=1,
            end_padding="zero_masked_out",
            loss="valid_element_mean_L1_plus_weighted_KL",
            optimizer="AdamW",
            validation_posterior="mean_latent",
            validation_prior="zero_latent",
            best_checkpoint_metric="prior_l1",
            ensemble_weighting="exp(-decay*prediction_age)",
        )
        save_json(
            args.output_dir / "config.json",
            {
                **{
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "model_config": model.get_config(),
                "data_config": prepared,
                "preprocessing": preprocessing,
                "updates_per_dataset_pass": len(train) // args.batch_size,
            },
        )
        print(
            f"ACT; {device}; cameras={model.camera_names}; chunk={model.chunk_size}; batch={args.batch_size}; "
            f"parameters={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )
        print(
            f"Train/val={len(train.episodes)}/{len(val.episodes)} episodes; {len(train)} training windows; "
            f"{len(train) // args.batch_size} updates/pass; target={args.max_updates}; best.pt uses z=0 prior L1",
            flush=True,
        )
        while updates < args.max_updates:
            target = min(
                args.max_updates,
                (updates // args.checkpoint_every + 1) * args.checkpoint_every,
            )
            if args.rollout_eval_every:
                target = min(
                    target,
                    (updates // args.rollout_eval_every + 1) * args.rollout_eval_every,
                )
            tr = run_batches(
                model,
                normalizer,
                train,
                (sampler.next_indices() for _ in range(target - updates)),
                device,
                kl_weight=args.kl_weight,
                optimizer=optimizer,
                gradient_clip=args.gradient_clip,
            )
            updates = target
            va = run_batches(
                model,
                normalizer,
                val,
                validation_indices(
                    len(val), args.batch_size, args.validation_batches, args.seed + 1
                ),
                device,
                kl_weight=args.kl_weight,
            )
            improved = va["prior_l1"] < best
            best = min(best, va["prior_l1"])
            history.append(dict(updates=updates, train=tr, val=va))
            payload = dict(
                format_version=1,
                policy_type=model.policy_type,
                model_config=model.get_config(),
                data_config=prepared,
                preprocessing=preprocessing,
                training_config=training_config,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                updates=updates,
                epoch=len(history),
                best_val_loss=best,
                best_val_metric="prior_l1",
                history=history,
                best_rollout_success_rate=best_rollout,
                rollout_history=rollout_history,
                sampler_state=sampler.state_dict(),
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=(
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
                numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),
            )
            save_checkpoint(args.output_dir / "latest.pt", payload)
            save_json(args.output_dir / "metrics.json", history)
            if args.rollout_eval_every and updates % args.rollout_eval_every == 0:
                result = run_rollout(args, updates, device)
                rollout_history.append(result)
                rollout_improved = result["success_rate"] > best_rollout
                best_rollout = max(best_rollout, result["success_rate"])
                payload.update(best_rollout_success_rate=best_rollout)
                if rollout_improved:
                    save_checkpoint(args.output_dir / "best_rollout.pt", payload)
                save_checkpoint(args.output_dir / "latest.pt", payload)
                save_json(args.output_dir / "rollout_evaluations.json", rollout_history)
                print(
                    f"Rollout={result['success_rate']:.1%}; best={best_rollout:.1%}",
                    flush=True,
                )
            if improved:
                save_checkpoint(args.output_dir / "best.pt", payload)
            print(
                f"Update {updates:07d}/{args.max_updates}: train L1={tr['l1']:.6f} KL={tr['kl']:.6f} "
                f"val posterior L1={va['l1']:.6f} prior(z=0) L1={va['prior_l1']:.6f} "
                f"(xyz={va['prior_xyz_l1']:.6f}, rotation={va['prior_rotation_l1']:.6f}, "
                f"gripper={va['prior_gripper_l1']:.6f})",
                flush=True,
            )
    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    main()
