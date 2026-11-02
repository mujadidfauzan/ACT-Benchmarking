"""Evaluate native OSC ACT with shared task scenes and physical success criteria."""

import argparse
import json
from pathlib import Path

import torch

from data.normalization import PolicyNormalizer
from models.act import ACTPolicy
from scripts.evaluate_visual_bc import evaluate_policy
from scripts.train_visual_bc import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-objects", "--num-cubes", dest="num_cubes", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--num-receptacles", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execution-steps", type=int, help="Actions executed before replanning; default checkpoint setting")
    parser.add_argument("--temporal-ensemble", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ensemble-decay", type=float)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=float, default=20.)
    parser.add_argument("--log-rollout", action="store_true")
    parser.add_argument("--rollout-dir", type=Path)
    return parser.parse_args()


def load_policy(path, device, *, execution_steps=None, temporal_ensemble=None, ensemble_decay=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != "act":
        raise ValueError("Expected an ACT checkpoint")
    normalizer = PolicyNormalizer.from_statistics(checkpoint["data_config"]["normalization"]).to(device)
    model = ACTPolicy.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if model.proprio_dim != normalizer.proprio_dim:
        raise ValueError("ACT checkpoint proprioception and normalization differ")
    model.configure_inference(
        model.execution_steps if execution_steps is None else execution_steps,
        model.temporal_ensemble if temporal_ensemble is None else temporal_ensemble,
        model.ensemble_decay if ensemble_decay is None else ensemble_decay)
    model.observation_image_size = int(checkpoint["data_config"]["camera_sizes"][0][0])
    model.eval()
    return model, normalizer, checkpoint


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, normalizer, checkpoint = load_policy(args.checkpoint, device, execution_steps=args.execution_steps,
                                               temporal_ensemble=args.temporal_ensemble, ensemble_decay=args.ensemble_decay)
    args.rnn_reset_interval, args.success_hold_steps = 0, 20
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty evaluation output directory")
    summary = evaluate_policy(args, model, normalizer, checkpoint, device)
    summary.update(checkpoint_epoch=None, checkpoint_updates=checkpoint["updates"],
                   checkpoint_best_val_metric=checkpoint["best_val_metric"], chunk_size=model.chunk_size,
                   execution_steps=model.execution_steps, temporal_ensemble=model.temporal_ensemble,
                   ensemble_decay=model.ensemble_decay, ensemble_weighting="exp(-decay*prediction_age)",
                   image_size=model.image_size, source_image_size=model.observation_image_size,
                   preprocessing=checkpoint["preprocessing"], latent_at_inference="zero",
                   recurrent_type=None, uses_previous_gripper=False,
                   action_space="OSC_delta_7d_continuous", gripper_logit_threshold=None)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if any(k in summary["tasks"][0]["failure_counts"] for k in ("exception", "scene_sampling_failure")):
        raise RuntimeError(f"Evaluation encountered infrastructure errors; inspect {args.output_dir}")
    return summary


if __name__ == "__main__":
    main()
