"""Train ACT on native HDF5 demos: masked L1 + KL, exact resume, rollout selection."""

import argparse
import hashlib
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


def parse_args(defaults=None):
    defaults = DEFAULTS if defaults is None else defaults
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
    parser.add_argument("--train-episodes", type=int, help="Use the first N train-split episodes; 0 uses all")
    parser.add_argument("--validation-source", choices=("val", "train"),
                        help="train is an explicitly labelled memorization diagnostic, not held-out validation")
    parser.add_argument("--allow-scratch", action="store_true",
                        help="Explicit diagnostic override for a recipe that requires local pretrained weights")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print training progress every N updates (also the first); 0 disables")
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
    for key, value in defaults.items():
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


def clip_act_gradients(model, max_norm):
    """Fail on invalid gradients; recover only float32 norm-reduction overflow."""
    parameters = [p for p in model.parameters() if p.grad is not None]
    try:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error) and "nonfinite" not in str(error):
            raise
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad:
            raise FloatingPointError("Nonfinite ACT gradients in: " + ", ".join(bad)) from None
        # Individual gradients can be finite while their squared sum
        # overflows float32. Recompute the same global L2 norm in float64.
        if not parameters:
            raise
        norms = torch.stack([p.grad.detach().double().norm(2) for p in parameters])
        norm = norms.norm(2)
        if not torch.isfinite(norm):
            raise FloatingPointError("ACT global gradient norm is nonfinite even in float64") from None
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1.)
        with torch.no_grad():
            for p in parameters:
                p.grad.mul_(coefficient.to(device=p.grad.device, dtype=p.grad.dtype))
        return norm


def save_numerical_failure(directory, model, dataset, indices, update, losses, error):
    if directory is None:
        return None
    gradients = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            gradients.append(dict(parameter=name, nan_count=int(torch.isnan(gradient).sum()),
                                  inf_count=int(torch.isinf(gradient).sum())))
    samples = []
    for index in indices:
        episode, start = dataset._sample_index[index]
        samples.append(dict(index=int(index), episode_path=dataset.episodes[episode]["path"],
                            start=int(start), steps=int(dataset.episodes[episode]["steps"])))
    values = {key: float(value.detach()) for key, value in losses.items()}
    path = Path(directory) / f"nonfinite_update_{update:07d}.json"
    save_json(path, dict(update=update, error=str(error), torch_version=str(torch.__version__),
                        cuda_version=torch.version.cuda, device=str(next(model.parameters()).device),
                        attention_implementation="explicit_softmax", samples=samples,
                        losses={k: v if math.isfinite(v) else str(v) for k, v in values.items()},
                        nonfinite_gradients=gradients, optimizer_step_executed=False))
    return path


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
    log_every=0,
    start_update=0,
    max_updates=None,
    failure_dir=None,
):
    training = optimizer is not None
    model.train(training)
    totals, examples, valid_steps = {}, 0, 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, indices in enumerate(batches, 1):
            update = start_update + batch_index
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
                error = FloatingPointError("Nonfinite ACT reconstruction/KL loss")
                path = save_numerical_failure(failure_dir, model, dataset, indices, update, losses, error)
                raise FloatingPointError(f"{error}; update={update}; diagnostic={path}")
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
                try:
                    norm = clip_act_gradients(model, gradient_clip)
                except FloatingPointError as error:
                    path = save_numerical_failure(failure_dir, model, dataset, indices, update, losses, error)
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(f"{error}; update={update}; diagnostic={path}") from error
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                values["gradient_norm"] = float(norm)
                update = start_update + batch_index
                if log_every and (batch_index == 1 or update % log_every == 0):
                    print(f"Training update {update:07d}/{max_updates or '?'}: "
                          f"L1={values['l1']:.6f} KL={values['kl']:.6f}", flush=True)
            else:
                # Deployment diagnostic: no demonstration actions enter the model.
                prior = (losses if model.policy_type == "act_lerobot" else
                         action_losses(model(**batch)["action"], target, padding))
                if not all(torch.isfinite(value) for value in prior.values()):
                    raise FloatingPointError("Nonfinite ACT validation loss with z=0")
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
    # An interrupted evaluator may have written partial artifacts. Keep them
    # for inspection and give a retried evaluation a fresh output directory.
    attempt = 0
    base = output
    while output.exists() and any(output.iterdir()):
        attempt += 1
        output = base.with_name(f"{base.name}_retry{attempt:02d}")
    command = [
        sys.executable,
        "-m",
        getattr(args, "evaluation_module", "scripts.evaluate_act"),
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


def save_rollout(args, payload, device):
    """Persist a completed rollout, including when recovering a pending one."""
    result = run_rollout(args, payload["updates"], device)
    payload["rollout_history"].append(result)
    improved = result["success_rate"] > payload["best_rollout_success_rate"]
    payload["best_rollout_success_rate"] = max(payload["best_rollout_success_rate"], result["success_rate"])
    if improved:
        save_checkpoint(args.output_dir / "best_rollout.pt", payload)
    save_checkpoint(args.output_dir / "latest.pt", payload)
    save_json(args.output_dir / "rollout_evaluations.json", payload["rollout_history"])
    print(f"Rollout={result['success_rate']:.1%}; best={payload['best_rollout_success_rate']:.1%}", flush=True)


def select_episodes(dataset, paths):
    """Restrict in memory only: source HDF5 and prepared split stay unchanged."""
    selected = set(paths)
    dataset.episodes = [entry for entry in dataset.episodes if entry["path"] in selected]
    if {entry["path"] for entry in dataset.episodes} != selected:
        raise ValueError("Training subset does not belong to selected manifest")
    dataset._sample_index = [(i, t) for i, entry in enumerate(dataset.episodes)
                             for t in range(int(entry["steps"]))]


def main(*, policy_class=ACTPolicy, default_overrides=None,
         evaluation_module="scripts.evaluate_act", require_pretrained=False):
    defaults = {**DEFAULTS, **(default_overrides or {})}
    args = parse_args(defaults)
    args.evaluation_module = evaluation_module
    checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    if checkpoint and checkpoint.get("policy_type") != policy_class.policy_type:
        raise ValueError(f"Resume requires a {policy_class.policy_type} checkpoint; use a new directory for a new baseline")
    saved = checkpoint["training_config"] if checkpoint else {}
    for key, default in {**defaults, "temporal_ensemble": False, "train_episodes": 0, "validation_source": "val"}.items():
        value = getattr(args, key)
        if checkpoint and value is not None and value != saved.get(key, default):
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
    if args.train_episodes < 0:
        raise ValueError("--train-episodes must be nonnegative")
    if require_pretrained and not args.resume and not args.backbone_weights and not args.allow_scratch:
        raise ValueError("This baseline requires --backbone-weights /local/resnet18-f37072fd.pth; "
                         "no automatic downloads. --allow-scratch is only an explicit diagnostic override.")
    if args.backbone_weights and not args.backbone_weights.is_file():
        raise FileNotFoundError(args.backbone_weights)
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
    if min(args.validation_batches, args.rollout_eval_every, args.log_every) < 0:
        raise ValueError(
            "Validation batch limit, rollout interval, and log interval must be nonnegative"
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
    if prepared["tasks"] == ["stack"]:
        from environments.stack_config import validate_fixed_stack_prepared
        validate_fixed_stack_prepared(prepared)
    elif prepared["tasks"] not in (["pick"], ["place"]) or prepared["fixed_target_color"] is None:
        raise ValueError("ACT supports fixed-target Pick/Place or fixed-order Stack datasets")
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
    model = policy_class(
        proprio_dim=normalizer.proprio_dim,
        camera_names=prepared["camera_names"],
        predict_orientation=prepared["predict_orientation"],
        **{k: getattr(args, k) for k in model_keys},
    ).to(device)
    if args.backbone_weights:
        if hasattr(model, "load_backbone_weights"):
            model.load_backbone_weights(args.backbone_weights)
        else:
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
        directory / f"{args.validation_source}.jsonl",
        model.camera_names,
        normalizer.proprio_keys,
        model.chunk_size,
    )
    try:
        if args.train_episodes and args.train_episodes > len(train.episodes):
            raise ValueError("--train-episodes exceeds the original train split")
        paths = [entry["path"] for entry in train.episodes[:args.train_episodes or None]]
        select_episodes(train, paths)
        if args.validation_source == "train":
            select_episodes(val, paths)
        training_subset = [dict(path=e["path"], episode=e.get("episode"), steps=e["steps"]) for e in train.episodes]
        if checkpoint and checkpoint.get("training_subset", training_subset) != training_subset:
            raise ValueError("Resume training subset differs")
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
        pending_rollout = bool(checkpoint and args.rollout_eval_every and updates > 0
                               and updates % args.rollout_eval_every == 0
                               and not any(r["updates"] == updates for r in rollout_history))
        if updates > args.max_updates or (updates == args.max_updates and not pending_rollout):
            raise ValueError("--max-updates must exceed checkpoint updates")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        training_config = {
            k: getattr(args, k) for k in (*defaults, "temporal_ensemble", "train_episodes", "validation_source")
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
            attention_implementation="explicit_softmax",
            gradient_clipping="global_L2_nonfinite_guard_float64_norm_overflow_fallback",
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
        if policy_class.policy_type == "act_lerobot":
            from models.act_lerobot.audit import audit_inputs
            reference_path = Path(__file__).resolve().parents[1] / "models/act_lerobot/reference.json"
            preprocessing.update(reference=json.loads(reference_path.read_text()),
                                 validation_posterior="not_used; eval follows LeRobot z=0",
                                 transformer_initialization="Xavier_uniform_encoder_decoder",
                                 ensemble_weighting="exp(-decay*prediction_rank_oldest_first)")
            if args.backbone_weights:
                preprocessing["backbone_sha256"] = hashlib.sha256(args.backbone_weights.read_bytes()).hexdigest()
            elif checkpoint and "backbone_sha256" in checkpoint["preprocessing"]:
                preprocessing["backbone_sha256"] = checkpoint["preprocessing"]["backbone_sha256"]
            report, arrays = audit_inputs(model, normalizer, train, device)
            save_json(args.output_dir / "input_audit.json", report)
            np.savez_compressed(args.output_dir / "input_audit_chunks.npz", **arrays)
        save_json(args.output_dir / "training_subset.json", training_subset)
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
                "training_subset": training_subset,
                "validation_source": args.validation_source,
            },
        )
        print(
            f"{model.policy_type}; {device}; cameras={model.camera_names}; chunk={model.chunk_size}; batch={args.batch_size}; "
            f"parameters={sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )
        print(
            f"Train/val={len(train.episodes)}/{len(val.episodes)} episodes; {len(train)} training windows; "
            f"{len(train) // args.batch_size} updates/pass; target={args.max_updates}; best.pt uses z=0 prior L1",
            flush=True,
        )
        if pending_rollout:
            print(f"Resuming pending rollout at update {updates}", flush=True)
            save_rollout(args, checkpoint, device)
            best_rollout = checkpoint["best_rollout_success_rate"]
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
                log_every=args.log_every,
                start_update=updates,
                max_updates=args.max_updates,
                failure_dir=args.output_dir / "diagnostics",
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
            if model.policy_type == "act_lerobot":
                report, arrays = audit_inputs(model, normalizer, val, device)
                save_json(args.output_dir / "prediction_diagnostics.json", dict(updates=updates, split=args.validation_source, **report))
                np.savez_compressed(args.output_dir / "prediction_chunks.npz", **arrays)
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
                training_subset=training_subset,
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
            # Preserve the validation winner even if the following rollout
            # fails before it can produce a rollout-selected checkpoint.
            if improved:
                save_checkpoint(args.output_dir / "best.pt", payload)
            if args.rollout_eval_every and updates % args.rollout_eval_every == 0:
                save_rollout(args, payload, device)
                best_rollout = payload["best_rollout_success_rate"]
            print(
                f"Update {updates:07d}/{args.max_updates}: train L1={tr['l1']:.6f} KL={tr['kl']:.6f} "
                f"{args.validation_source} reconstruction L1={va['l1']:.6f} prior(z=0) L1={va['prior_l1']:.6f} "
                f"(xyz={va['prior_xyz_l1']:.6f}, rotation={va['prior_rotation_l1']:.6f}, "
                f"gripper={va['prior_gripper_l1']:.6f})",
                flush=True,
            )
    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    main()
