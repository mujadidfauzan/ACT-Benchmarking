"""Evaluate privileged-state BC-RNN with the existing Place success criteria."""

import argparse
import json
from pathlib import Path

import torch

from models.low_dim_bc_rnn import LowDimBCRNNPolicy
from models.low_dim_bc_rnn.policy import RawStateInputs
from models.low_dim_bc_rnn.observations import STATE_DIM, validate_scene
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
    parser.add_argument("--num-objects", "--num-cubes", dest="num_cubes", type=int, choices=(1,))
    parser.add_argument("--num-receptacles", type=int, choices=(1,))
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
    if checkpoint.get("policy_type") != LowDimBCRNNPolicy.policy_type:
        raise ValueError("Expected a low_dim_bc_rnn checkpoint")
    model = LowDimBCRNNPolicy.from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    validate_scene(checkpoint["data_config"])
    model.eval()
    return model, checkpoint


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_policy(args.checkpoint, device)
    # The shared evaluator resets the explicit state at the reference horizon;
    # LowDimBCRNNPolicy.step also enforces this when called outside this evaluator.
    model.render_observation_cameras = args.record_video
    args.rnn_reset_interval = model.sequence_length
    args.success_hold_steps = 20
    summary = evaluate_policy(args, model, RawStateInputs(), checkpoint, device)
    summary.update(checkpoint_updates=checkpoint["updates"], observation_mode="low_dim",
                   privileged_state=True, state_dim=STATE_DIM, image_size=None,
                   video_camera_names=list(model.video_camera_names) if args.record_video else [],
                   preprocessing=checkpoint["preprocessing"], recurrent_type="LSTM",
                   uses_previous_gripper=False, action_space="raw_controller_7d_continuous",
                   gripper_logit_threshold=None)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    failures = summary["tasks"][0]["failure_counts"]
    if any(key in failures for key in ("exception", "scene_sampling_failure")):
        raise RuntimeError(f"Evaluation encountered infrastructure errors; inspect {args.output_dir}")
    return summary


if __name__ == "__main__":
    main()
