"""Evaluate deterministic BC-RNN with the existing task factory and criteria."""

import argparse
import json
from pathlib import Path

import torch

from models.robomimic_bc_rnn import BCRNNPolicy
from models.robomimic_bc_rnn.policy import RawPolicyInputs
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
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=float, default=20.)
    parser.add_argument("--log-rollout", action="store_true")
    parser.add_argument("--rollout-dir", type=Path)
    return parser.parse_args()


def load_policy(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("policy_type") != BCRNNPolicy.policy_type:
        raise ValueError("Expected a robomimic_bc_rnn checkpoint; use evaluate_visual_bc for A/B/C")
    model = BCRNNPolicy.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if "data_config" in checkpoint:
        model.observation_image_size = int(checkpoint["data_config"]["camera_sizes"][0][0])
    model.eval()
    return model, checkpoint


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_policy(args.checkpoint, device)
    # The shared evaluator resets the explicit state at the reference horizon;
    # BCRNNPolicy.step also enforces this when called outside this evaluator.
    args.rnn_reset_interval = model.sequence_length
    args.success_hold_steps = 20
    summary = evaluate_policy(args, model, RawPolicyInputs(), checkpoint, device)
    summary.update(checkpoint_updates=checkpoint["updates"], image_size=model.image_size,
                   source_image_size=checkpoint["data_config"]["camera_sizes"][0][0],
                   crop_size=model.crop_size, preprocessing=checkpoint["preprocessing"],
                   recurrent_type="LSTM", uses_previous_gripper=False,
                   action_space="raw_controller_7d_continuous", gripper_logit_threshold=None)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    failures = summary["tasks"][0]["failure_counts"]
    if any(key in failures for key in ("exception", "scene_sampling_failure")):
        raise RuntimeError(f"Evaluation encountered infrastructure errors; inspect {args.output_dir}")
    return summary


if __name__ == "__main__":
    main()
