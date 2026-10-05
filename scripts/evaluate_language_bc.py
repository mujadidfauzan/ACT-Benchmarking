"""Closed-loop simulator evaluation for trained Language-BC policies."""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import cv2
import robosuite.macros as macros
import torch
from robosuite import load_composite_controller_config
from robosuite.utils import RandomizationError
from torch.nn import functional as F

from data.normalization import PolicyNormalizer
from environments.pick_env import PickEnv
from environments.place_env import PlaceEnv
from environments.stack_env import StackEnv
from language.frozen_minilm import (
    CachedLanguageEmbeddings,
    CachedTokenEmbeddings,
    FrozenMiniLMEncoder,
    FrozenMiniLMTokenEncoder,
)
from models.language_bc import LanguageBCPolicy


macros.IMAGE_CONVENTION = "opencv"

TASKS = ("pick", "place", "stack")
DEFAULT_MAX_STEPS = {"pick": 400, "place": 800, "stack": 1600}
DEFAULT_SEEDS = {"pick": 11000, "place": 12000, "stack": 13000}
PROPRIO_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Language-BC with closed-loop simulator rollouts"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("data/splits/pilot_v02_semantic/normalization.json"),
    )
    parser.add_argument(
        "--pooled-cache",
        type=Path,
        default=Path("data/splits/pilot_v02_semantic/language_embeddings.npz"),
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=Path(
            "data/splits/pilot_v02_semantic/language_token_embeddings.npz"
        ),
    )
    parser.add_argument("--task", choices=("all", *TASKS), default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--num-cubes", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--instruction-split", choices=("train", "held_out"), default="train"
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Save one agentview MP4 with instruction overlay per episode",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Video directory; defaults to <output-dir>/videos",
    )
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument(
        "--strict-language-cache",
        action="store_true",
        help="Fail instead of encoding instructions absent from the cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/language_bc_evaluation"),
    )
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.image_size is not None and args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.success_hold_steps <= 0:
        raise ValueError("--success-hold-steps must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def save_json(path, value):
    path.write_text(
        json.dumps(value, default=json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wrap_text(text, maximum_characters=36):
    words = text.split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join((*current, word))
        if current and len(candidate) > maximum_characters:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


class EvaluationVideoRecorder:
    """Write policy-view RGB frames with readable task instruction overlay."""

    def __init__(self, path, first_frame, instruction, task, episode, fps):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame = np.asarray(first_frame)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Unexpected video frame shape: {frame.shape}")
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.path), fourcc, float(fps), (width, height)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.path}")
        self.instruction_lines = wrap_text(instruction)
        self.task = task
        self.episode = int(episode)

    def write(self, rgb_frame, step):
        frame = np.asarray(rgb_frame).copy()
        overlay_height = 28 + 19 * len(self.instruction_lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv2.putText(
            frame,
            f"{self.task.upper()} | Episode {self.episode + 1} | Step {step}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for line_index, line in enumerate(self.instruction_lines):
            cv2.putText(
                frame,
                line,
                (8, 38 + line_index * 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __del__(self):
        self.close()


def build_environment(task, seed, instruction_split, num_cubes, render):
    controller_config = load_composite_controller_config(controller="BASIC")
    common = {
        "robots": "Panda",
        "controller_configs": controller_config,
        "instruction_split": instruction_split,
        "has_renderer": render,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "use_object_obs": True,
        "camera_names": "agentview",
        "camera_heights": 256,
        "camera_widths": 256,
        "control_freq": 20,
        "hard_reset": False,
        "seed": seed,
    }
    if task == "pick":
        return PickEnv(num_objects=num_cubes, horizon=DEFAULT_MAX_STEPS[task], **common)
    if task == "place":
        return PlaceEnv(horizon=DEFAULT_MAX_STEPS[task], **common)
    return StackEnv(
        num_cubes=num_cubes, horizon=DEFAULT_MAX_STEPS[task], **common
    )


class LanguageProvider:
    def __init__(self, mode, pooled_cache, token_cache, strict, device):
        self.mode = mode
        self.strict = strict
        self.device = device
        self.live_encoder = None
        if mode == "pooled":
            self.cache = CachedLanguageEmbeddings(pooled_cache)
        else:
            self.cache = CachedTokenEmbeddings(token_cache)

    @property
    def embedding_dim(self):
        return self.cache.embedding_dim

    def _live_inputs(self, instruction):
        if self.strict:
            raise KeyError(
                "Evaluation instruction is absent from the language cache: "
                f"{instruction!r}"
            )
        if self.live_encoder is None:
            metadata = self.cache.metadata
            encoder_type = (
                FrozenMiniLMEncoder
                if self.mode == "pooled"
                else FrozenMiniLMTokenEncoder
            )
            self.live_encoder = encoder_type(
                model_name=metadata["model_name"],
                revision=metadata.get("revision"),
                device=str(self.device),
            )
        if self.mode == "pooled":
            embedding = self.live_encoder.encode([instruction]).to(self.device)
            return {"language_embedding": embedding}, "live"

        encoded = self.live_encoder.encode_tokens([instruction])
        tokens = encoded["token_embeddings"][0].unsqueeze(0).to(self.device)
        mask = torch.ones(
            (1, tokens.shape[1]), dtype=torch.bool, device=self.device
        )
        return {
            "token_embeddings": tokens,
            "attention_mask": mask,
        }, "live"

    def inputs(self, instruction):
        if instruction not in self.cache:
            return self._live_inputs(instruction)
        if self.mode == "pooled":
            embedding = self.cache.encode([instruction], self.device)
            return {"language_embedding": embedding}, "cache"
        cached = self.cache.lookup([instruction], self.device)
        return {
            "token_embeddings": cached["token_embeddings"],
            "attention_mask": cached["attention_mask"],
        }, "cache"


def load_policy(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    language_mode = model_config.get("language_mode")
    if language_mode not in ("pooled", "token_attention"):
        raise ValueError(f"Unsupported checkpoint language mode: {language_mode!r}")

    normalizer = PolicyNormalizer.from_json(args.normalization).to(device)
    language = LanguageProvider(
        language_mode,
        args.pooled_cache,
        args.token_cache,
        args.strict_language_cache,
        device,
    )
    model = LanguageBCPolicy(
        language_mode=language_mode,
        action_dim=normalizer.action_dim,
        proprio_dim=normalizer.proprio_dim,
        language_input_dim=language.embedding_dim,
        pretrained_visual=False,
        freeze_visual_backbone=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    training_config = checkpoint.get("training_config", {})
    image_size = args.image_size or int(training_config.get("image_size", 128))
    return model, normalizer, language, checkpoint, image_size


def policy_observation(observation, image_size, device):
    image = np.asarray(observation["agentview_image"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected agentview image shape: {image.shape}")
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    image = torch.from_numpy(image).float().div_(255.0).unsqueeze(0).to(device)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    proprio_parts = [
        np.asarray(observation[key], dtype=np.float32).reshape(-1)
        for key in PROPRIO_KEYS
    ]
    proprio = torch.from_numpy(np.concatenate(proprio_parts))
    return image, proprio.unsqueeze(0).to(device)


def entity_positions(env, task):
    positions = {
        "objects": {
            object_id: env.get_object_position(object_id)
            for object_id in env.object_ids
        }
    }
    if task == "place":
        positions["receptacles"] = {
            receptacle_id: env.get_receptacle_position(receptacle_id)
            for receptacle_id in env.receptacle_ids
        }
    return positions


def lifted_objects(env):
    table_center_z = env.table_offset[2] + env.cube_half_size
    return [
        object_id
        for object_id in env.object_ids
        if env.get_object_position(object_id)[2] - table_center_z >= 0.08
    ]


def placed_pairs(env):
    pairs = []
    for object_id in env.object_ids:
        object_position = env.get_object_position(object_id)
        for receptacle_id in env.receptacle_ids:
            receptacle_position = env.get_receptacle_position(receptacle_id)
            if np.linalg.norm(object_position[:2] - receptacle_position[:2]) <= 0.04:
                pairs.append((object_id, receptacle_id))
    return pairs


def inferred_stack_order(env):
    positions = {
        color: env.get_object_position(color)
        for color in env.stack_order
    }
    ordered = sorted(positions, key=lambda color: positions[color][2])
    bottom_z = env.table_offset[2] + env.cube_half_size
    if abs(positions[ordered[0]][2] - bottom_z) > env.stack_z_tolerance:
        return None
    for support, top in zip(ordered[:-1], ordered[1:]):
        xy_error = np.linalg.norm(positions[top][:2] - positions[support][:2])
        z_error = abs(
            (positions[top][2] - positions[support][2]) - env.cube_height
        )
        if xy_error > env.stack_xy_tolerance or z_error > env.stack_z_tolerance:
            return None
    return ordered


def task_success(task, env, metadata, observation):
    if not env._check_success():
        return False
    eef_position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    if task == "place":
        object_position = env.get_object_position(metadata["target_object"])
        return bool(np.linalg.norm(object_position - eef_position) > 0.08)
    if task == "stack":
        return all(
            np.linalg.norm(env.get_object_position(color) - eef_position) > 0.08
            for color in metadata["stack_order"][1:]
        )
    return True


def classify_failure(task, env, metadata, maximum_lift):
    if task == "pick":
        lifted = lifted_objects(env)
        if any(item != metadata["target_object"] for item in lifted):
            return "wrong_object"
        if maximum_lift < 0.02:
            return "no_lift"
        return "insufficient_lift"

    if task == "place":
        target_pair = (
            metadata["target_object"],
            metadata["target_receptacle"],
        )
        pairs = placed_pairs(env)
        if any(
            receptacle == target_pair[1] and object_id != target_pair[0]
            for object_id, receptacle in pairs
        ):
            return "wrong_object"
        if any(
            object_id == target_pair[0] and receptacle != target_pair[1]
            for object_id, receptacle in pairs
        ):
            return "wrong_target"
        if maximum_lift < 0.02:
            return "no_lift"
        return "place_timeout"

    actual_order = inferred_stack_order(env)
    if actual_order is not None and actual_order != metadata["stack_order"]:
        return "wrong_stack_order"
    if maximum_lift < 0.02:
        return "no_lift"
    return "stack_timeout"


def evaluate_episode(
    task,
    env,
    model,
    normalizer,
    language,
    image_size,
    device,
    max_steps,
    success_hold_steps,
    render,
    episode_index,
    video_path,
    video_fps,
):
    observation = env.reset()
    metadata = env.get_task_metadata()
    instruction = metadata["instruction"]
    language_inputs, language_source = language.inputs(instruction)
    video = None
    if video_path is not None:
        video = EvaluationVideoRecorder(
            video_path,
            observation["agentview_image"],
            instruction,
            task,
            episode_index,
            video_fps,
        )
        video.write(observation["agentview_image"], step=0)
    initial_positions = entity_positions(env, task)
    maximum_lift = 0.0
    success_streak = 0
    action_clipped_values = 0
    action_values = 0
    action_low, action_high = (
        np.asarray(value, dtype=np.float32) for value in env.action_spec
    )
    started = time.monotonic()

    for step in range(1, max_steps + 1):
        image, proprio = policy_observation(observation, image_size, device)
        normalized_proprio = normalizer.normalize_proprio(proprio)
        with torch.inference_mode():
            normalized_action = model(
                image=image,
                proprio=normalized_proprio,
                **language_inputs,
            )["action"]
            action_tensor = normalizer.denormalize_action(normalized_action)
        raw_action = action_tensor[0].cpu().numpy().astype(np.float32)
        action = np.clip(raw_action, action_low, action_high)
        action_clipped_values += int(np.count_nonzero(action != raw_action))
        action_values += action.size
        observation, _, done, _ = env.step(action)
        if video is not None:
            video.write(observation["agentview_image"], step=step)
        if render:
            env.render()

        current_lift = max(
            float(
                env.get_object_position(object_id)[2]
                - initial_positions["objects"][object_id][2]
            )
            for object_id in env.object_ids
        )
        maximum_lift = max(maximum_lift, current_lift)
        if task_success(task, env, metadata, observation):
            success_streak += 1
            if success_streak >= success_hold_steps:
                if video is not None:
                    video.close()
                return {
                    "success": True,
                    "failure_reason": None,
                    "steps": step,
                    "instruction": instruction,
                    "metadata": metadata,
                    "language_source": language_source,
                    "maximum_object_lift": maximum_lift,
                    "action_clip_fraction": action_clipped_values / action_values,
                    "initial_positions": initial_positions,
                    "final_positions": entity_positions(env, task),
                    "elapsed_seconds": time.monotonic() - started,
                    "video_path": str(video_path) if video_path is not None else None,
                }
        else:
            success_streak = 0
        if done:
            break

    if video is not None:
        video.close()
    return {
        "success": False,
        "failure_reason": classify_failure(task, env, metadata, maximum_lift),
        "steps": step,
        "instruction": instruction,
        "metadata": metadata,
        "language_source": language_source,
        "maximum_object_lift": maximum_lift,
        "action_clip_fraction": action_clipped_values / max(action_values, 1),
        "initial_positions": initial_positions,
        "final_positions": entity_positions(env, task),
        "elapsed_seconds": time.monotonic() - started,
        "video_path": str(video_path) if video_path is not None else None,
    }


def evaluate_task(
    task, args, model, normalizer, language, image_size, device, output_path
):
    successes = 0
    failure_counts = Counter()
    language_sources = Counter()
    total_steps = 0
    total_seconds = 0.0
    clip_fractions = []
    seed = DEFAULT_SEEDS[task] + args.seed_offset
    env = build_environment(
        task, seed, args.instruction_split, args.num_cubes, args.render
    )
    try:
        for episode_index in range(args.episodes):
            video_path = None
            if args.record_video:
                video_root = args.video_dir or (args.output_dir / "videos")
                video_path = video_root / (
                    f"{task}_episode_{episode_index:03d}.mp4"
                )
            try:
                result = evaluate_episode(
                    task,
                    env,
                    model,
                    normalizer,
                    language,
                    image_size,
                    device,
                    args.max_steps or DEFAULT_MAX_STEPS[task],
                    args.success_hold_steps,
                    args.render,
                    episode_index,
                    video_path,
                    args.video_fps,
                )
            except RandomizationError as error:
                result = {
                    "success": False,
                    "failure_reason": "scene_sampling_failure",
                    "steps": 0,
                    "instruction": None,
                    "metadata": {"task": task},
                    "language_source": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "elapsed_seconds": 0.0,
                }
            except Exception as error:
                result = {
                    "success": False,
                    "failure_reason": "exception",
                    "steps": int(getattr(env, "timestep", 0)),
                    "instruction": None,
                    "metadata": {"task": task},
                    "language_source": None,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "elapsed_seconds": 0.0,
                }

            result.update(
                {
                    "episode": episode_index,
                    "task": task,
                    "seed_start": seed,
                }
            )
            append_jsonl(output_path, result)
            successes += int(result["success"])
            total_steps += int(result["steps"])
            total_seconds += float(result["elapsed_seconds"])
            if result.get("action_clip_fraction") is not None:
                clip_fractions.append(result["action_clip_fraction"])
            if result["failure_reason"] is not None:
                failure_counts[result["failure_reason"]] += 1
            if result["language_source"] is not None:
                language_sources[result["language_source"]] += 1
            print(
                f"[{task}] {episode_index + 1}/{args.episodes} "
                f"success={result['success']} steps={result['steps']} "
                f"failure={result['failure_reason']}"
            )
    finally:
        env.close()

    return {
        "task": task,
        "attempted": args.episodes,
        "successful": successes,
        "failed": args.episodes - successes,
        "success_rate": successes / args.episodes,
        "failure_counts": dict(sorted(failure_counts.items())),
        "language_sources": dict(sorted(language_sources.items())),
        "mean_steps": total_steps / args.episodes,
        "mean_seconds": total_seconds / args.episodes,
        "mean_action_clip_fraction": (
            float(np.mean(clip_fractions)) if clip_fractions else None
        ),
        "attempts_log": output_path.name,
    }


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, normalizer, language, checkpoint, image_size = load_policy(args, device)
    tasks = TASKS if args.task == "all" else (args.task,)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Language mode: {model.language_mode}")
    print(f"Device: {device}")
    print(f"Image size: {image_size}")
    summaries = []
    for task in tasks:
        attempts_path = args.output_dir / f"{task}_attempts.jsonl"
        if attempts_path.exists():
            raise FileExistsError(
                f"Output already exists: {attempts_path}; use a new output directory"
            )
        summaries.append(
            evaluate_task(
                task,
                args,
                model,
                normalizer,
                language,
                image_size,
                device,
                attempts_path,
            )
        )

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_best_val_loss": float(checkpoint["best_val_loss"]),
        "language_mode": model.language_mode,
        "device": str(device),
        "image_convention": "opencv",
        "vertical_flip": False,
        "image_size": image_size,
        "instruction_split": args.instruction_split,
        "episodes_per_task": args.episodes,
        "success_hold_steps": args.success_hold_steps,
        "record_video": args.record_video,
        "video_fps": args.video_fps if args.record_video else None,
        "video_dir": (
            str(args.video_dir or (args.output_dir / "videos"))
            if args.record_video
            else None
        ),
        "tasks": summaries,
    }
    save_json(args.output_dir / "summary.json", summary)
    print("\nEvaluation summary")
    for item in summaries:
        print(
            f"  {item['task']:<5}: {item['successful']}/{item['attempted']} "
            f"({item['success_rate']:.1%}) failures={item['failure_counts']}"
        )
    print(f"Saved: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
