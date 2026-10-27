"""Train a privileged-state Place BC-RNN using the existing dataset and split.

Only numeric 23D state is loaded; no image encoder, phase labels, or action
normalization. Optimizer-update budgets and checkpoint semantics match the
image baseline. Run this as a new experiment, not a vision checkpoint resume.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from data.bc_preparation import load_bc_training, save_json
from models.low_dim_bc_rnn import LowDimBCRNNPolicy
from models.low_dim_bc_rnn.dataset import LowDimSequenceDataset
from models.low_dim_bc_rnn.observations import STATE_DIM, STATE_FIELDS, validate_scene
from scripts.train_robomimic_bc_rnn import (
    DEFAULTS as IMAGE_DEFAULTS, WindowSampler, run_batches, validation_indices,
)
from scripts.train_visual_bc import resolve_device, save_checkpoint, set_seed


DEFAULTS = {**IMAGE_DEFAULTS, "encoder_dim": 128}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-updates", type=int, default=300000,
                        help="Total optimizer updates, including updates before resume")
    parser.add_argument("--device", default="auto")
    for name, default in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"),
                            type=float if isinstance(default, float) else int,
                            help=f"Default {default}; restored on resume")
    return parser.parse_args()


def run_rollout(args, updates, device):
    output = args.output_dir / "rollouts" / f"update_{updates:07d}"
    command = [sys.executable, "-m", "scripts.evaluate_low_dim_bc_rnn",
               "--checkpoint", str(args.output_dir / "latest.pt"), "--device", str(device),
               "--episodes", str(args.rollout_eval_episodes), "--num-seeds", str(args.rollout_eval_num_seeds),
               "--seed", str(args.rollout_eval_seed), "--max-steps", str(args.rollout_eval_max_steps),
               "--output-dir", str(output)]
    subprocess.run(command, check=True)
    summary = json.loads((output / "summary.json").read_text())
    if any(key in summary["tasks"][0]["failure_counts"] for key in ("exception", "scene_sampling_failure")):
        raise RuntimeError(f"Rollout infrastructure failed; inspect {output}")
    rate = float(summary["overall_success_rate"])
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Invalid rollout success rate")
    return dict(updates=updates, success_rate=rate, summary=str(output / "summary.json"))


def main():
    args = parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint and checkpoint.get("policy_type") != LowDimBCRNNPolicy.policy_type:
        raise ValueError("Resume requires a low_dim_bc_rnn checkpoint")
    saved = checkpoint["training_config"] if checkpoint else {}
    for key, default in DEFAULTS.items():
        value = getattr(args, key)
        if checkpoint and value is not None and value != saved[key]:
            raise ValueError(f"--{key.replace('_', '-')} differs from checkpoint; start a new run")
        setattr(args, key, saved.get(key, default) if value is None else value)
    positive = ("batch_size", "sequence_length", "hidden_dim", "num_layers", "encoder_dim", "checkpoint_every",
                "rollout_eval_episodes", "rollout_eval_num_seeds", "rollout_eval_max_steps")
    if args.max_updates <= 0 or any(getattr(args, key) <= 0 for key in positive):
        raise ValueError("Update, batch, network, and rollout sizes must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or min(args.validation_batches, args.rollout_eval_every) < 0:
        raise ValueError("Need positive finite LR and nonnegative validation/rollout intervals")
    if args.resume and args.resume.resolve().parent != args.output_dir.resolve():
        raise ValueError("--output-dir must contain the resume checkpoint")
    if not checkpoint and any((args.output_dir / name).exists() for name in ("latest.pt", "best.pt", "best_rollout.pt")):
        raise FileExistsError("Choose a new output directory or --resume")
    set_seed(args.seed)
    device = resolve_device(args.device)
    # Reuse the prepared split; never recreate or change it for this comparison.
    print("Verifying prepared dataset hashes and split...", flush=True)
    prepared, directory = load_bc_training(args.dataset, args.training_dir)
    validate_scene(prepared)
    if checkpoint and checkpoint["data_config"] != prepared:
        raise ValueError("Resume dataset, source digest, or train/validation split differs")
    model = LowDimBCRNNPolicy(hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                             sequence_length=args.sequence_length, encoder_dim=args.encoder_dim,
                             predict_orientation=prepared["predict_orientation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    train = LowDimSequenceDataset(args.dataset, directory / "train.jsonl", model.sequence_length)
    val = LowDimSequenceDataset(args.dataset, directory / "val.jsonl", model.sequence_length)
    try:
        sampler = WindowSampler(len(train), args.batch_size, args.seed)
        updates, best, best_rollout, history, rollout_history = 0, float("inf"), -1., [], []
        if checkpoint:
            if checkpoint["model_config"] != model.get_config():
                raise ValueError("Resume model configuration differs")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            sampler.load_state_dict(checkpoint["sampler_state"])
            updates, best = checkpoint["updates"], checkpoint["best_val_loss"]
            best_rollout = checkpoint["best_rollout_success_rate"]
            history, rollout_history = checkpoint["history"], checkpoint["rollout_history"]
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
        if updates >= args.max_updates:
            raise ValueError("--max-updates must exceed checkpoint updates")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        training_config = {key: getattr(args, key) for key in DEFAULTS}
        preprocessing = dict(observation_mode="low_dim", privileged_state=True,
                             state_fields=[[name, dim] for name, dim in STATE_FIELDS], state_dim=STATE_DIM,
                             state_normalization="none", position_frame="world", quaternion_order="xyzw",
                             action_normalization="none", action_head="tanh_continuous_7d",
                             window_stride=1, end_padding="repeat_last_included_in_loss",
                             loss="mean_squared_error_all_7_action_dimensions",
                             optimizer="Adam", weight_decay=0.)
        save_json(args.output_dir / "config.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "model_config": model.get_config(), "data_config": prepared, "preprocessing": preprocessing})
        print(f"Low-dim BC-RNN; {device}; state={STATE_DIM}D; batch={args.batch_size}; "
              f"LSTM={model.num_layers}x{model.hidden_dim}; sequence={model.sequence_length}", flush=True)
        print(f"Train/val={len(train.episodes)}/{len(val.episodes)} episodes; Adam LR={args.learning_rate}; "
              f"target={args.max_updates} updates; no loss-based early stopping", flush=True)
        while updates < args.max_updates:
            target = min(args.max_updates, (updates // args.checkpoint_every + 1) * args.checkpoint_every)
            if args.rollout_eval_every:
                target = min(target, (updates // args.rollout_eval_every + 1) * args.rollout_eval_every)
            tr = run_batches(model, train, (sampler.next_indices() for _ in range(target - updates)), device, optimizer)
            updates = target
            va = run_batches(model, val, validation_indices(len(val), args.batch_size, args.validation_batches, args.seed + 1), device)
            improved = va["loss"] < best
            best = min(best, va["loss"])
            history.append(dict(updates=updates, train=tr, val=va))
            payload = dict(format_version=1, policy_type=model.policy_type, model_config=model.get_config(),
                           data_config=prepared, preprocessing=preprocessing, training_config=training_config,
                           model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                           updates=updates, epoch=len(history), best_val_loss=best, history=history,
                           best_rollout_success_rate=best_rollout, rollout_history=rollout_history,
                           sampler_state=sampler.state_dict(), torch_rng_state=torch.get_rng_state(),
                           cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                           numpy_rng_state=np.random.get_state(), python_rng_state=random.getstate())
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
                print(f"Rollout={result['success_rate']:.1%}; best={best_rollout:.1%}", flush=True)
            if improved:
                save_checkpoint(args.output_dir / "best.pt", payload)
            print(f"Update {updates:07d}/{args.max_updates}: train={tr['loss']:.6f} val={va['loss']:.6f} "
                  f"(xyz={va['xyz_mse']:.6f}, rotation={va['rotation_mse']:.6f}, gripper={va['gripper_mse']:.6f})", flush=True)
    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    main()
