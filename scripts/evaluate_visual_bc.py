"""Closed-loop Pick evaluation for fixed-target Visual-BC policies."""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import robosuite.macros as macros
import torch
from robosuite.utils import RandomizationError

from data.normalization import PolicyNormalizer
from models.visual_bc import VisualBCPolicy
from scripts.evaluation_utils import (
    append_jsonl,
    json_default,
    EvaluationVideoRecorder,
    compose_camera_frame,
    entity_positions,
    policy_observation,
)


macros.IMAGE_CONVENTION = "opencv"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fixed-target Pick or matching-color Place Visual-BC")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, help="Optional verification of embedded checkpoint statistics")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-cubes", "--num-objects", dest="num_cubes",
                        type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-receptacles", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--success-hold-steps", type=int)
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


def write_jsonl(handle, value):
    handle.write(json.dumps(value, default=json_default, sort_keys=True) + "\n")


def load_policy(args, device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    policy_type = checkpoint.get("policy_type", config.get("policy_type"))
    if policy_type != "visual_bc" or checkpoint.get("data_config", {}).get("method") != "visual_bc":
        raise ValueError("Only visual_bc checkpoints are supported; use the earlier code version for old Visual-BC models")
    prepared = checkpoint["data_config"]
    if args.image_size is not None and args.image_size != 128:
        raise ValueError("Visual-BC uses 128x128 policy images")
    normalizer = PolicyNormalizer.from_statistics(prepared["normalization"]).to(device)
    if getattr(args, "normalization", None) is not None:
        supplied = PolicyNormalizer.from_json(args.normalization).to(device)
        if supplied.proprio_keys != normalizer.proprio_keys or any(
            not torch.equal(value, supplied.state_dict()[key]) for key, value in normalizer.state_dict().items()
        ):
            raise ValueError("Supplied normalization differs from checkpoint statistics")
    model = VisualBCPolicy.from_config(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, normalizer, checkpoint, 128, prepared["fixed_target_color"]


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
    task = metadata.get("task", "pick")
    target_id = metadata["target_object"]
    if metadata["target_color"] != target_color:
        raise RuntimeError("Environment target color does not match checkpoint")
    if task == "place" and metadata.get("target_receptacle_color") != target_color:
        raise RuntimeError("Environment receptacle color does not match checkpoint")
    label = f"Vision-only target: {target_color} cube"
    if task == "place":
        label += f" into {target_color} tray"
    video = None
    if video_path is not None:
        frame = compose_camera_frame(observation, model.camera_names)
        video = EvaluationVideoRecorder(
            video_path, frame, label, task, episode_index, video_fps
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

    initial_positions = entity_positions(env)
    maximum_lifts = {object_id: 0.0 for object_id in env.object_ids}
    success_streak = 0
    clipped_values = 0
    action_values = 0
    action_low, action_high = (
        np.asarray(value, dtype=np.float32) for value in env.action_spec
    )
    started = time.monotonic()
    hidden = None
    previous_gripper = -1.0
    placement = None
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
            policy_inputs = {"proprio": normalizer.normalize_proprio(proprio)}
            with torch.inference_mode():
                output = model.step(
                    **camera_inputs,
                    **policy_inputs,
                    previous_gripper=previous_gripper,
                    hidden=hidden,
                )
                prediction = output["action"]
                hidden = output.get("hidden")
                raw_action = normalizer.denormalize_action(prediction)[0]
            raw_action = raw_action.cpu().numpy().astype(np.float32)
            action = np.clip(raw_action, action_low, action_high)
            previous_gripper = float(action[6])
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
            if task == "place":
                placement = env.placement_status(observation["robot0_eef_pos"], initial_positions["objects"])
                placement["gripper_open"] = bool(action[6] < 0)
                achieved = placement["success"] and placement["gripper_open"]
            else:
                lifted = env.get_object_position(target_id)[2] - initial_positions["objects"][target_id][2] >= .18
                achieved = env._check_success() and lifted
            if achieved:
                success_streak += 1
                if success_streak >= success_hold_steps:
                    return {
                        "success": True,
                        "failure_reason": None,
                        "steps": step,
                        "metadata": metadata,
                        "placement": placement,
                        "maximum_lifts": maximum_lifts,
                        "target_maximum_lift": maximum_lifts[target_id],
                        "action_clip_fraction": clipped_values / action_values,
                        "initial_positions": initial_positions,
                        "final_positions": entity_positions(env),
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
    elif task == "place" and placement and placement["wrong_target"]:
        reason = "wrong_receptacle"
    elif task == "place" and placement and not placement["distractors_stable"]:
        reason = "distractor_displacement"
    elif task == "place" and placement and placement["inside_target"] and (not placement["released"] or not placement["gripper_open"]):
        reason = "not_released"
    elif maximum_lifts[target_id] < 0.02:
        reason = "no_lift"
    elif task == "place":
        reason = "place_timeout"
    else:
        reason = "insufficient_lift"
    return {
        "success": False,
        "failure_reason": reason,
        "steps": step,
        "metadata": metadata,
        "placement": placement,
        "maximum_lifts": maximum_lifts,
        "target_maximum_lift": maximum_lifts[target_id],
        "action_clip_fraction": clipped_values / max(action_values, 1),
        "initial_positions": initial_positions,
        "final_positions": entity_positions(env),
        "elapsed_seconds": time.monotonic() - started,
        "video_path": str(video_path) if video_path else None,
        "rollout_path": str(rollout_path) if rollout_path else None,
    }


def build_environment(seed, num_objects, render, model, prepared, max_steps):
    # Use the same factory/controller/reset settings as collection, not an eval-only scene.
    from scripts.record_demonstrations import build_task

    size = int(prepared["camera_sizes"][0][0])
    task = prepared.get("tasks", ["pick"])[0]
    collection = prepared.get("environment_configs", {}).get(task, {})
    args = SimpleNamespace(task=task, num_cubes=num_objects, fixed_target_color=prepared["fixed_target_color"],
                           num_receptacles=prepared.get("evaluation_num_receptacles") or max(value[1] for value in prepared["scene_counts"]) or 3,
                           camera_names=list(model.camera_names), camera_obs=True, render=render,
                           instruction_split="train", image_size=size, seed=seed,
                           max_steps=max_steps,
                           environment_kwargs=collection.get("environment_kwargs", {}),
                           controller_config=collection.get("controller_config"),
                           rotation_mode=collection.get("rotation_mode", prepared.get("rotation_mode", "align_yaw")),
                           yaw_range_deg=collection.get("yaw_range_deg"),
                           joint_noise_degrees=collection.get("joint_noise_degrees", 3))
    env, _ = build_task(args, num_objects=num_objects, seed=seed)
    return env


def evaluate_policy(args, model, normalizer, checkpoint, device):
    prepared = checkpoint["data_config"]
    task = prepared.get("tasks", ["pick"])[0]
    if task not in ("pick", "place"):
        raise ValueError("Only Pick and Place checkpoints are supported")
    if getattr(args, "num_receptacles", None) is not None:
        if task != "place":
            raise ValueError("--num-receptacles requires a Place checkpoint")
        prepared = {**prepared, "evaluation_num_receptacles": args.num_receptacles}
    args.max_steps = (3000 if task == "place" else 500) if args.max_steps is None else args.max_steps
    args.success_hold_steps = 20 if args.success_hold_steps is None else args.success_hold_steps
    if args.success_hold_steps != 20:
        raise ValueError("Evaluation requires 20 consecutive success steps")
    if min(args.episodes, args.num_seeds, args.seed_step, args.max_steps) <= 0:
        raise ValueError("Episodes, seeds, and maximum steps must be positive")
    num_objects = args.num_cubes or max(value[0] for value in prepared["scene_counts"])
    seeds = [args.seed + args.seed_offset + i * args.seed_step for i in range(args.num_seeds)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = args.output_dir / f"{task}_attempts.jsonl"
    if attempts_path.exists():
        raise FileExistsError(attempts_path)
    seed_results = []
    failures = Counter()
    global_episode = 0
    total_steps = 0
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env = build_environment(seed, num_objects, args.render, model, prepared, args.max_steps)
        successful = 0
        try:
            for episode in range(args.episodes):
                video_path = ((args.video_dir or args.output_dir / "videos") / f"{task}_seed{seed}_episode{episode:03d}.mp4"
                              if args.record_video else None)
                rollout_path = ((args.rollout_dir or args.output_dir / "rollouts") / f"{task}_seed{seed}_episode{episode:03d}.jsonl"
                                if args.log_rollout else None)
                try:
                    result = evaluate_episode(env, model, normalizer, 128, device, args.max_steps, 20,
                                              args.render, global_episode, video_path, args.video_fps,
                                              prepared["fixed_target_color"], rollout_path)
                except Exception as error:
                    result = {"success": False, "failure_reason": "scene_sampling_failure" if isinstance(error, RandomizationError) else "exception",
                              "steps": 0, "error_type": type(error).__name__, "error_message": str(error)}
                result.update(seed=seed, episode=episode, global_episode=global_episode, task=task, num_objects=num_objects)
                if task == "place":
                    result["num_receptacles"] = env.num_receptacles
                append_jsonl(attempts_path, result)
                successful += int(result["success"])
                total_steps += result["steps"]
                if result.get("failure_reason"):
                    failures[result["failure_reason"]] += 1
                global_episode += 1
                print(f"Seed {seed} episode {episode + 1}/{args.episodes}: success={result['success']} failure={result.get('failure_reason')}")
        finally:
            env.close()
        seed_results.append({"seed": seed, "successful": successful, "attempted": args.episodes,
                             "success_rate": successful / args.episodes})
    rates = np.asarray([value["success_rate"] for value in seed_results])
    successes = sum(value["successful"] for value in seed_results)
    summary = {"policy_type": model.policy_type, "checkpoint": str(args.checkpoint),
               "checkpoint_epoch": checkpoint["epoch"], "checkpoint_best_val_loss": checkpoint["best_val_loss"],
               "device": str(device), "camera_names": list(model.camera_names), "num_objects": num_objects,
               "temporal": model.temporal, "predict_orientation": model.predict_orientation,
               "rotation_mode": prepared.get("rotation_mode", "align_yaw" if model.predict_orientation else "none"),
               "yaw_range_deg": prepared.get("environment_configs", {}).get("pick", {}).get("yaw_range_deg"),
               "image_convention": "opencv", "vertical_flip": False, "image_size": 128,
               "episodes_per_seed": args.episodes, "seeds": seeds, "seed_results": seed_results,
               "lift_threshold_m": .18 if task == "pick" else None, "success_hold_steps": 20,
               "num_receptacles": (prepared.get("evaluation_num_receptacles") or max(value[1] for value in prepared["scene_counts"])) if task == "place" else None,
               "success_criterion": "lift_0.18m" if task == "pick" else "inside_tray_released_distractors_stable",
               "mean_seed_success_rate": float(rates.mean()),
               "std_seed_success_rate_sample": float(rates.std(ddof=1)) if len(rates) > 1 else 0.,
               "overall_success_rate": successes / global_episode,
               "tasks": [{"task": task, "successful": successes, "attempted": global_episode,
                          "failed": global_episode - successes, "success_rate": successes / global_episode,
                          "failure_counts": dict(failures), "mean_steps": total_steps / global_episode}]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation: {successes}/{global_episode} ({successes / global_episode:.1%}), "
          f"seed mean={rates.mean():.1%}, sample SD={summary['std_seed_success_rate_sample']:.1%}")
    return summary


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, normalizer, checkpoint, _, _ = load_policy(args, device)
    return evaluate_policy(args, model, normalizer, checkpoint, device)


if __name__ == "__main__":
    main()
