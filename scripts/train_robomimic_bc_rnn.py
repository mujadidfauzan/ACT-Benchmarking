"""Deterministic robomimic-style image BC-RNN on our existing HDF5 episodes.

An epoch in the paper is a fixed update budget; this entry point uses explicit
optimizer updates instead. It never converts or rewrites demonstration files.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from data.bc_preparation import load_bc_training, prepare_bc_training, save_json
from models.robomimic_bc_rnn import BCRNNPolicy
from models.robomimic_bc_rnn.dataset import BCRNNSequenceDataset
from models.robomimic_bc_rnn.policy import PROPRIO_KEYS
from scripts.train_visual_bc import resolve_device, save_checkpoint, set_seed


DEFAULTS = dict(batch_size=8, learning_rate=1e-4, sequence_length=10,
                hidden_dim=1000, num_layers=2, seed=123, checkpoint_every=500,
                validation_batches=50, rollout_eval_every=10000,
                rollout_eval_episodes=10, rollout_eval_num_seeds=5,
                rollout_eval_seed=1000, rollout_eval_max_steps=300)


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


class WindowSampler:
    """Shuffled, without-replacement batches with an exact resumable cursor.

    Drops the final incomplete training batch, as in the reference loader.
    A private RNG keeps validation and sampler draws independent of crop RNG.
    """

    def __init__(self, size, batch_size, seed):
        if not 0 < batch_size <= size:
            raise ValueError("Training split must contain at least one full batch")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.position = torch.empty(0, dtype=torch.long), 0

    def next_indices(self):
        if self.position + self.batch_size > len(self.order):
            self.order = torch.randperm(self.size, generator=self.generator)
            self.position = 0
        indices = self.order[self.position:self.position + self.batch_size].tolist()
        self.position += self.batch_size
        return indices

    def state_dict(self):
        return dict(size=self.size, batch_size=self.batch_size, order=self.order,
                    position=self.position, generator=self.generator.get_state())

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Resume sampler size/batch differs")
        self.order, self.position = state["order"], state["position"]
        if len(self.order) not in (0, self.size) or not 0 <= self.position <= len(self.order):
            raise ValueError("Invalid sampler cursor")
        self.generator.set_state(state["generator"])


def read_batch(dataset, indices, device):
    return {key: value.to(device) for key, value in default_collate([dataset[i] for i in indices]).items()}


def action_losses(prediction, target):
    # Reference deterministic BC: unnormalized seven-dimensional action MSE.
    # L1 and cosine weights are zero, including for the gripper component.
    error = (prediction - target).square()
    return dict(loss=F.mse_loss(prediction, target), xyz_mse=error[..., :3].mean(),
                rotation_mse=error[..., 3:6].mean(), gripper_mse=error[..., 6].mean())


def run_batches(model, dataset, batches, device, optimizer=None):
    model.train(optimizer is not None)
    totals, count, updates = np.zeros(4), 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for indices in batches:
            batch = read_batch(dataset, indices, device)
            action = batch.pop("action")
            losses = action_losses(model(**batch)["action"], action)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite BC-RNN loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                optimizer.step()
                updates += 1
            size = action.shape[0] * action.shape[1]
            totals += np.asarray([value.detach().item() for value in losses.values()]) * size
            count += size
    if not count:
        raise ValueError("Empty training/validation batches")
    return {**dict(zip(("loss", "xyz_mse", "rotation_mse", "gripper_mse"), (totals / count).tolist())),
            "timesteps_including_padding": count, "optimizer_steps": updates}


def validation_indices(size, batch_size, limit, seed):
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    for index, batch in enumerate(order.split(batch_size)):
        if limit and index >= limit:
            break
        yield batch.tolist()


def run_rollout(args, updates, device):
    output = args.output_dir / "rollouts" / f"update_{updates:07d}"
    command = [sys.executable, "-m", "scripts.evaluate_robomimic_bc_rnn",
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
    if checkpoint and checkpoint.get("policy_type") != BCRNNPolicy.policy_type:
        raise ValueError("Resume requires a robomimic_bc_rnn checkpoint")
    saved = checkpoint["training_config"] if checkpoint else {}
    for key, default in DEFAULTS.items():
        value = getattr(args, key)
        if checkpoint and value is not None and value != saved[key]:
            raise ValueError(f"--{key.replace('_', '-')} differs from checkpoint; start a new run")
        setattr(args, key, saved.get(key, default) if value is None else value)
    positive = ("batch_size", "sequence_length", "hidden_dim", "num_layers", "checkpoint_every",
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
    try:
        prepared, directory = load_bc_training(args.dataset, args.training_dir)
    except FileNotFoundError:
        if checkpoint:
            raise
        prepare_bc_training(args.dataset, args.training_dir)
        prepared, directory = load_bc_training(args.dataset, args.training_dir)
    if prepared["tasks"] not in (["pick"], ["place"]) or prepared["fixed_target_color"] is None:
        raise ValueError("BC-RNN supports fixed-target Pick or Place datasets")
    if checkpoint and checkpoint["data_config"] != prepared:
        raise ValueError("Resume dataset, source digest, or train/validation split differs")
    model = BCRNNPolicy(camera_names=prepared["camera_names"], hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, sequence_length=args.sequence_length,
                        predict_orientation=prepared["predict_orientation"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    train = BCRNNSequenceDataset(args.dataset, directory / "train.jsonl", model.camera_names, model.sequence_length)
    val = BCRNNSequenceDataset(args.dataset, directory / "val.jsonl", model.camera_names, model.sequence_length)
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
        preprocessing = dict(proprio_keys=list(PROPRIO_KEYS), proprio_normalization="none",
                             action_normalization="none", action_head="tanh_continuous_7d",
                             image_size=model.image_size, crop_size=model.crop_size,
                             training_crop="random_per_frame", evaluation_crop="center",
                             window_stride=1, end_padding="repeat_last_included_in_loss",
                             loss="mean_squared_error_all_7_action_dimensions",
                             optimizer="Adam", weight_decay=0.)
        save_json(args.output_dir / "config.json", {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "model_config": model.get_config(), "data_config": prepared, "preprocessing": preprocessing})
        print(f"BC-RNN deterministic; {device}; cameras={model.camera_names}; batch={args.batch_size}; "
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
