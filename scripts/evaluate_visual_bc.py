"""Closed-loop Pick evaluation for fixed-target Visual-BC policies."""

import argparse
import json
import random
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import robosuite.macros as macros
import torch
from robosuite import load_composite_controller_config
from robosuite.utils import RandomizationError

from data.normalization import PolicyNormalizer
from environments.pick_env import PickEnv
from models.visual_bc import HistoryVisualBCPolicy, VisualBCPolicy
from scripts.evaluate_language_bc import (
    EvaluationVideoRecorder,
    compose_camera_frame,
    entity_positions,
    policy_observation,
)


macros.IMAGE_CONVENTION = "opencv"
DEFAULT_MAX_STEPS = 400


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fixed-target Visual-BC")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-cubes", type=int, choices=(3, 4), default=3)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument(
        "--log-rollout",
        action="store_true",
        help="Write per-timestep policy, EEF, and target diagnostics as JSONL",
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        help="Directory for rollout JSONL files; defaults to <output-dir>/rollouts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, default=json_default, sort_keys=True) + "\n")


def write_jsonl(handle, value):
    handle.write(json.dumps(value, default=json_default, sort_keys=True) + "\n")


def load_policy(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    policy_type = checkpoint.get("policy_type", config.get("policy_type"))
    if policy_type not in {"visual_bc", "history_visual_bc"}:
        raise ValueError("Checkpoint is not a supported Visual-BC checkpoint")
    normalizer = PolicyNormalizer.from_json(args.normalization).to(device)
    if int(config.get("proprio_dim", -1)) != normalizer.proprio_dim:
        raise ValueError("Checkpoint and normalizer proprio dimensions do not match")
    model_class = (
        HistoryVisualBCPolicy
        if policy_type == "history_visual_bc"
        else VisualBCPolicy
    )
    model_kwargs = {
        "visual_fusion": config["visual_fusion"],
        "camera_names": tuple(config["camera_names"]),
        "action_dim": normalizer.action_dim,
        "proprio_dim": normalizer.proprio_dim,
        "pretrained_visual": False,
        "freeze_visual_backbone": True,
    }
    if policy_type == "history_visual_bc":
        model_kwargs["history_size"] = int(config["history_size"])
    model = model_class(
        **model_kwargs
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    training = checkpoint.get("training_config", {})
    image_size = args.image_size or int(training.get("image_size", 224))
    target_color = config.get("fixed_target_color")
    if target_color is None:
        raise ValueError("Checkpoint does not define fixed_target_color")
    return model, normalizer, checkpoint, image_size, target_color


def build_environment(
    seed, num_cubes, render, camera_names, target_color, max_steps
):
    config = load_composite_controller_config(controller="BASIC")
    return PickEnv(
        robots="Panda",
        controller_configs=config,
        num_objects=num_cubes,
        fixed_target_color=target_color,
        instruction_split="train",
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=list(camera_names),
        camera_heights=[256] * len(camera_names),
        camera_widths=[256] * len(camera_names),
        control_freq=20,
        horizon=max_steps,
        hard_reset=False,
        seed=seed,
    )


def evaluate_episode(
    env,
    model,
    normalizer,
    image_size,
    device,
    max_steps,
    success_hold_steps,
    render,
    episode_index,
    video_path,
    video_fps,
    target_color,
    rollout_path=None,
):
    observation = env.reset()
    metadata = env.get_task_metadata()
    target_id = metadata["target_object"]
    if metadata["target_color"] != target_color:
        raise RuntimeError("Environment target color does not match checkpoint")
    label = f"Vision-only target: {target_color} cube"
    video = None
    if video_path is not None:
        frame = compose_camera_frame(observation, model.camera_names)
        video = EvaluationVideoRecorder(
            video_path, frame, label, "pick", episode_index, video_fps
        )
        video.write(frame, 0)

    rollout_handle = None
    if rollout_path is not None:
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_handle = rollout_path.open("w", encoding="utf-8")
        write_jsonl(
            rollout_handle,
            {
                "event": "episode_start",
                "episode": episode_index,
                "target_id": target_id,
                "target_color": target_color,
                "metadata": metadata,
            },
        )

    initial_positions = entity_positions(env, "pick")
    maximum_lifts = {object_id: 0.0 for object_id in env.object_ids}
    success_streak = 0
    clipped_values = 0
    action_values = 0
    action_low, action_high = (
        np.asarray(value, dtype=np.float32) for value in env.action_spec
    )
    started = time.monotonic()
    uses_history = isinstance(model, HistoryVisualBCPolicy)
    proprio_queue = deque(maxlen=model.history_size if uses_history else 1)
    action_queue = deque(maxlen=model.history_size - 1 if uses_history else 1)
    try:
        for step in range(1, max_steps + 1):
            eef_before = np.asarray(
                observation["robot0_eef_pos"], dtype=np.float32
            ).copy()
            gripper_before = np.asarray(
                observation.get("robot0_gripper_qpos", []), dtype=np.float32
            ).copy()
            target_before = env.get_object_position(target_id).copy()
            target_delta_before = target_before - eef_before
            camera_inputs, proprio = policy_observation(
                observation,
                image_size,
                device,
                normalizer.proprio_keys,
                model.camera_names,
            )
            if uses_history:
                proprio_queue.append(proprio[0].detach().clone())
                proprio_history = torch.zeros(
                    (1, model.history_size, normalizer.proprio_dim),
                    dtype=proprio.dtype,
                    device=device,
                )
                proprio_mask = torch.zeros(
                    (1, model.history_size), dtype=torch.bool, device=device
                )
                proprio_count = len(proprio_queue)
                proprio_history[0, -proprio_count:] = torch.stack(
                    tuple(proprio_queue)
                )
                proprio_mask[0, -proprio_count:] = True
                action_history = torch.zeros(
                    (1, model.history_size - 1, normalizer.action_dim),
                    dtype=proprio.dtype,
                    device=device,
                )
                action_mask = torch.zeros(
                    (1, model.history_size - 1),
                    dtype=torch.bool,
                    device=device,
                )
                action_count = len(action_queue)
                if action_count:
                    action_history[0, -action_count:] = torch.stack(
                        tuple(action_queue)
                    )
                    action_mask[0, -action_count:] = True
                policy_inputs = {
                    "proprio_history": normalizer.normalize_proprio(
                        proprio_history
                    )
                    * proprio_mask.unsqueeze(-1),
                    "action_history": normalizer.normalize_action(
                        action_history
                    )
                    * action_mask.unsqueeze(-1),
                    "proprio_history_mask": proprio_mask,
                    "action_history_mask": action_mask,
                }
            else:
                proprio_count = 1
                action_count = 0
                policy_inputs = {
                    "proprio": normalizer.normalize_proprio(proprio)
                }
            with torch.inference_mode():
                prediction = model(
                    **camera_inputs,
                    **policy_inputs,
                )["action"]
                raw_action = normalizer.denormalize_action(prediction)[0]
            raw_action = raw_action.cpu().numpy().astype(np.float32)
            action = np.clip(raw_action, action_low, action_high)
            if uses_history:
                action_queue.append(
                    torch.from_numpy(action.copy()).to(device=device)
                )
            clipped_mask = action != raw_action
            clipped_values += int(np.count_nonzero(clipped_mask))
            action_values += action.size
            observation, _, done, _ = env.step(action)
            if video is not None:
                video.write(compose_camera_frame(observation, model.camera_names), step)
            if render:
                env.render()

            for object_id in env.object_ids:
                lift = float(
                    env.get_object_position(object_id)[2]
                    - initial_positions["objects"][object_id][2]
                )
                maximum_lifts[object_id] = max(maximum_lifts[object_id], lift)
            if rollout_handle is not None:
                eef_after = np.asarray(
                    observation["robot0_eef_pos"], dtype=np.float32
                ).copy()
                target_after = env.get_object_position(target_id).copy()
                target_delta_after = target_after - eef_after
                write_jsonl(
                    rollout_handle,
                    {
                        "event": "step",
                        "step": step,
                        "eef_position_before": eef_before,
                        "target_position_before": target_before,
                        "eef_target_delta_before": target_delta_before,
                        "eef_target_xy_distance_before": float(
                            np.linalg.norm(target_delta_before[:2])
                        ),
                        "target_minus_eef_z_before": float(target_delta_before[2]),
                        "eef_target_distance_before": float(
                            np.linalg.norm(target_delta_before)
                        ),
                        "gripper_qpos_before": gripper_before,
                        "normalized_action": prediction[0].detach().cpu().numpy(),
                        "raw_action": raw_action,
                        "executed_action": action,
                        "action_clipped": clipped_mask,
                        "translation_action": action[:3],
                        "rotation_action": action[3:6],
                        "gripper_action": float(action[6]),
                        "proprio_history_valid": proprio_count,
                        "action_history_valid": action_count,
                        "eef_position_after": eef_after,
                        "target_position_after": target_after,
                        "eef_target_xy_distance_after": float(
                            np.linalg.norm(target_delta_after[:2])
                        ),
                        "target_minus_eef_z_after": float(target_delta_after[2]),
                        "eef_target_distance_after": float(
                            np.linalg.norm(target_delta_after)
                        ),
                        "target_lift": float(
                            target_after[2]
                            - initial_positions["objects"][target_id][2]
                        ),
                        "done": bool(done),
                    },
                )
            if env._check_success():
                success_streak += 1
                if success_streak >= success_hold_steps:
                    return {
                        "success": True,
                        "failure_reason": None,
                        "steps": step,
                        "metadata": metadata,
                        "maximum_lifts": maximum_lifts,
                        "target_maximum_lift": maximum_lifts[target_id],
                        "action_clip_fraction": clipped_values / action_values,
                        "initial_positions": initial_positions,
                        "final_positions": entity_positions(env, "pick"),
                        "elapsed_seconds": time.monotonic() - started,
                        "video_path": str(video_path) if video_path else None,
                        "rollout_path": str(rollout_path) if rollout_path else None,
                    }
            else:
                success_streak = 0
            if done:
                break
    finally:
        if video is not None:
            video.close()
        if rollout_handle is not None:
            rollout_handle.close()

    wrong_lifts = [
        lift
        for object_id, lift in maximum_lifts.items()
        if object_id != target_id
    ]
    if max(wrong_lifts, default=0.0) >= 0.08:
        reason = "wrong_object"
    elif maximum_lifts[target_id] < 0.02:
        reason = "no_lift"
    else:
        reason = "insufficient_lift"
    return {
        "success": False,
        "failure_reason": reason,
        "steps": step,
        "metadata": metadata,
        "maximum_lifts": maximum_lifts,
        "target_maximum_lift": maximum_lifts[target_id],
        "action_clip_fraction": clipped_values / max(action_values, 1),
        "initial_positions": initial_positions,
        "final_positions": entity_positions(env, "pick"),
        "elapsed_seconds": time.monotonic() - started,
        "video_path": str(video_path) if video_path else None,
        "rollout_path": str(rollout_path) if rollout_path else None,
    }


def main():
    args = parse_args()
    if min(args.episodes, args.max_steps, args.success_hold_steps) <= 0:
        raise ValueError("Episode and step counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, normalizer, checkpoint, image_size, target_color = load_policy(args, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = args.output_dir / "pick_attempts.jsonl"
    if attempts_path.exists():
        raise FileExistsError(f"Output already exists: {attempts_path}")

    env = build_environment(
        11000 + args.seed_offset,
        args.num_cubes,
        args.render,
        model.camera_names,
        target_color,
        args.max_steps,
    )
    successes = 0
    failures = Counter()
    clip_fractions = []
    total_steps = 0
    try:
        for episode in range(args.episodes):
            video_path = None
            if args.record_video:
                video_root = args.video_dir or (args.output_dir / "videos")
                video_path = video_root / f"pick_episode_{episode:03d}.mp4"
            rollout_path = None
            if args.log_rollout:
                rollout_root = args.rollout_dir or (args.output_dir / "rollouts")
                rollout_path = rollout_root / f"pick_episode_{episode:03d}.jsonl"
            try:
                result = evaluate_episode(
                    env,
                    model,
                    normalizer,
                    image_size,
                    device,
                    args.max_steps,
                    args.success_hold_steps,
                    args.render,
                    episode,
                    video_path,
                    args.video_fps,
                    target_color,
                    rollout_path,
                )
            except RandomizationError as error:
                result = {
                    "success": False,
                    "failure_reason": "scene_sampling_failure",
                    "steps": 0,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            result.update({"episode": episode, "task": "pick"})
            append_jsonl(attempts_path, result)
            successes += int(result["success"])
            total_steps += int(result["steps"])
            if result["failure_reason"]:
                failures[result["failure_reason"]] += 1
            if "action_clip_fraction" in result:
                clip_fractions.append(result["action_clip_fraction"])
            print(
                f"[pick] {episode + 1}/{args.episodes} "
                f"success={result['success']} steps={result['steps']} "
                f"failure={result['failure_reason']}"
            )
    finally:
        env.close()

    task_summary = {
        "task": "pick",
        "attempted": args.episodes,
        "successful": successes,
        "failed": args.episodes - successes,
        "success_rate": successes / args.episodes,
        "failure_counts": dict(sorted(failures.items())),
        "mean_steps": total_steps / args.episodes,
        "mean_action_clip_fraction": float(np.mean(clip_fractions)),
        "attempts_log": attempts_path.name,
    }
    summary = {
        "policy_type": checkpoint.get(
            "policy_type", checkpoint.get("model_config", {}).get("policy_type")
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_val_loss": float(checkpoint["best_val_loss"]),
        "fixed_target_color": target_color,
        "camera_names": list(model.camera_names),
        "visual_fusion": model.visual_fusion,
        "proprio_dim": normalizer.proprio_dim,
        "history_size": getattr(model, "history_size", 1),
        "image_size": image_size,
        "device": str(device),
        "record_video": args.record_video,
        "log_rollout": args.log_rollout,
        "rollout_dir": (
            str(args.rollout_dir or (args.output_dir / "rollouts"))
            if args.log_rollout
            else None
        ),
        "tasks": [task_summary],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Evaluation summary: pick {successes}/{args.episodes} "
        f"({successes / args.episodes:.1%}) failures={dict(failures)}"
    )
    print(f"Saved: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
