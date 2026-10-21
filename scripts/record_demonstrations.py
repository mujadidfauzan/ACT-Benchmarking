import argparse
import copy
import json
from pathlib import Path

import numpy as np
import robosuite.macros as macros
from robosuite import load_composite_controller_config
from robosuite.utils import RandomizationError

from controllers.panda_controller import PandaController
from data.trajectory_recorder import TrajectoryRecorder
from data.bc_preparation import dataset_entries, default_training_dir, prepare_bc_training, suspicious_camera_frames
from environments.pick_env import PickEnv
from environments.pick_config import add_pick_arguments, validate_pick_arguments, validate_pick_settings, yaw_range_degrees
from environments.place_env import PlaceEnv
from environments.stack_env import StackEnv
from environments.scene_counts import (
    add_count_arguments, balanced_count_plan, count_distribution, count_options,
)
from experts.pick_expert import PickExpert
from experts.grasp_expert import GraspExpert
from experts.place_expert import PlaceExpert
from experts.stack_expert import StackExpert

macros.IMAGE_CONVENTION = "opencv"


def parse_args():
    parser = argparse.ArgumentParser(description="Record successful expert demos")
    parser.add_argument("--task", choices=("pick", "place", "stack"), required=True)
    quota = parser.add_mutually_exclusive_group()
    quota.add_argument("--num-demos", type=int, help="Target total successful demos for this task (default: 100)")
    quota.add_argument("--episodes", type=int, help="Legacy: add this many successful episodes")
    parser.add_argument("--output", type=Path, default=Path("data/demos"))
    parser.add_argument("--seed", type=int, default=7)
    add_count_arguments(parser)
    add_pick_arguments(parser)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--fixed-target-color",
        choices=tuple(PickEnv.COLOR_PALETTE),
        help="Fixed Pick target (default: red); object ID stays random",
    )
    target.add_argument("--random-target", action="store_true", help="Pick only: vary target colors for collection; not supported by fixed-target BC")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--camera-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--training-dir", type=Path, help="Automatic split/statistics output directory")
    parser.add_argument(
        "--camera-names",
        nargs="+",
        choices=("agentview", "robot0_eye_in_hand"),
        default=["agentview"],
        help="RGB cameras to record; agentview should remain first",
    )
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument(
        "--instruction-split",
        choices=("train", "held_out"),
        default="train",
    )
    return parser.parse_args()


def build_task(args, num_objects=None, num_receptacles=None, seed=None):
    num_objects = args.num_cubes if num_objects is None else num_objects
    if num_receptacles is None:
        num_receptacles = getattr(args, "num_receptacles", None) or 3
    if args.camera_names[0] != "agentview":
        raise ValueError("--camera-names must start with agentview")
    if len(set(args.camera_names)) != len(args.camera_names):
        raise ValueError("--camera-names cannot contain duplicates")
    config = copy.deepcopy(getattr(args, "controller_config", None) or load_composite_controller_config(controller="BASIC"))
    right = config["body_parts"]["right"]
    right.update(type="OSC_POSE", input_type="delta", input_ref_frame="world", impedance_mode="fixed")
    collection_controller_config = copy.deepcopy(config)
    common = {
        "robots": "Panda",
        "controller_configs": config,
        "instruction_split": args.instruction_split,
        "has_renderer": args.render,
        "has_offscreen_renderer": args.camera_obs,
        "use_camera_obs": args.camera_obs,
        "use_object_obs": True,
        "camera_names": list(args.camera_names),
        "camera_heights": [getattr(args, "image_size", 128)] * len(args.camera_names),
        "camera_widths": [getattr(args, "image_size", 128)] * len(args.camera_names),
        "initialization_noise": {"magnitude": float(np.deg2rad(getattr(args, "joint_noise_degrees", 3))), "type": "uniform"},
        "control_freq": 20,
        "hard_reset": False,
        "seed": args.seed if seed is None else seed,
    }
    common.update(getattr(args, "environment_kwargs", {}))

    if args.task == "pick":
        rotation_mode = getattr(args, "rotation_mode", "align_yaw")
        yaw = (yaw_range_degrees(args.random_yaw) if hasattr(args, "random_yaw")
               else getattr(args, "yaw_range_deg", None))
        if yaw is not None:
            validate_pick_settings(rotation_mode, yaw)
            common["yaw_range"] = tuple(np.deg2rad(yaw))
        env = PickEnv(
            num_objects=num_objects,
            fixed_target_color=args.fixed_target_color,
            horizon=getattr(args, "max_steps", 500),
            **common,
        )
    elif args.task == "place":
        env = PlaceEnv(num_objects=num_objects, num_receptacles=num_receptacles,
                       horizon=3000, **common)
    else:
        env = StackEnv(num_cubes=num_objects, horizon=6000, **common)

    env.bc_controller_config = collection_controller_config

    controller = PandaController(env)
    grasp_expert = GraspExpert(controller)
    if args.task == "pick":
        expert = PickExpert(controller, rotation_mode=rotation_mode)
    elif args.task == "place":
        expert = PlaceExpert(controller, grasp_expert)
    else:
        expert = StackExpert(controller, grasp_expert)
    return env, expert


def run_expert(task, expert, observation, metadata, render):
    if task == "pick":
        return expert.run(
            observation,
            object_name=metadata["target_object"],
            render=render,
        )
    if task == "place":
        return expert.run(
            observation,
            object_name=metadata["target_object"],
            receptacle_name=metadata["target_receptacle"],
            render=render,
        )
    return expert.run(observation, render=render)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def append_jsonl(path, entry):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=json_default, sort_keys=True) + "\n")


def snapshot_scene(observation):
    return {
        key: np.asarray(value).copy()
        for key, value in observation.items()
        if key.endswith("_pos") or key.endswith("_quat")
    }


def failure_reason(info):
    stage = info.get("stage", "unknown")
    detail = info.get("detail") or {}
    if stage == "pick":
        return f"pick_{detail.get('stage', 'verification')}"
    if stage in {"approach", "pre_place", "place_pose"}:
        return f"transport_{stage}"
    if stage == "retreat":
        return "release_retreat"
    if stage == "verify_place":
        if info.get("wrong_target"):
            return "wrong_target"
        if not info.get("distractors_stable", True):
            return "distractor_displacement"
        return "place_verification"
    if stage == "verify_stack":
        return "stack_instability"
    if stage == "complete":
        return "task_verification"
    return stage


def main():
    args = parse_args()
    args.yaw_range_deg = validate_pick_arguments(args)
    if args.task == "pick" and args.fixed_target_color is None and not args.random_target:
        args.fixed_target_color = "red"
    if args.random_target and args.task != "pick":
        raise ValueError("--random-target is only valid for Pick")
    if args.image_size < 32:
        raise ValueError("image-size must be at least 32")
    for value in (args.episodes, args.num_demos):
        if value is not None and value <= 0:
            raise ValueError("Episode quotas must be positive")
    if not args.camera_obs and args.camera_names != ["agentview"]:
        raise ValueError("--camera-names requires --camera-obs")
    if args.fixed_target_color is not None and args.task != "pick":
        raise ValueError("--fixed-target-color is only valid for --task pick")
    object_counts, receptacle_counts = count_options(args.task, args)
    task_dir = args.output / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "manifest.jsonl"
    attempts_path = task_dir / "attempts.jsonl"
    existing_entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()] if manifest_path.exists() else []
    dataset_entries(args.output)
    for entry in existing_entries:
        config = entry.get("metadata", {}).get("collection_config")
        if config is None or config["camera_names"] != (args.camera_names if args.camera_obs else []) or config["image_size"] != args.image_size:
            raise ValueError("Existing collection uses another/unknown camera or control setup; choose a new dataset directory")
        if config.get("fixed_target_color") != args.fixed_target_color:
            raise ValueError("Existing collection uses another target configuration; choose a new dataset directory")
        if "rotation_mode" not in config or "yaw_range_deg" not in config:
            raise ValueError("Existing collection is missing yaw/rotation metadata; choose a new dataset directory")
        if args.task == "pick":
            if config["rotation_mode"] != args.rotation_mode or not np.allclose(
                config["yaw_range_deg"], args.yaw_range_deg, rtol=0, atol=1e-10
            ):
                raise ValueError("Existing collection uses another yaw/rotation configuration; choose a new dataset directory")
    if args.episodes is None:
        args.episodes = max(0, (args.num_demos or 100) - len(existing_entries))
    if args.episodes == 0:
        if args.camera_obs:
            prepare_bc_training(args.output, args.training_dir)
            print(f"Target already reached; training artifacts ready: {args.training_dir or default_training_dir(args.output)}")
        return
    groups = len(object_counts) * len(receptacle_counts)
    plan = balanced_count_plan(max(args.episodes, groups), object_counts, receptacle_counts, args.seed)[:args.episodes]
    max_attempts = args.episodes * 4 if args.max_attempts is None else args.max_attempts
    if max_attempts <= 0:
        raise ValueError("max-attempts must be positive")
    existing_indices = [
        int(path.stem.split("_")[-1])
        for path in task_dir.glob("episode_*.npz")
        if path.stem.split("_")[-1].isdigit()
    ]
    start_index = max(existing_indices, default=-1) + 1
    existing_attempts = 0
    if attempts_path.exists():
        existing_attempts = sum(
            1
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    env = expert = None
    current_configuration = None
    environment_seed = None
    saved_configurations = []
    print("Successful episode quotas:", count_distribution(plan))
    saved = 0
    attempts = 0
    interrupted = False
    try:
        while saved < args.episodes and attempts < max_attempts:
            attempts += 1
            attempt_index = existing_attempts + attempts - 1
            configuration = plan[saved]
            counts = {"num_objects": configuration[0]}
            if args.task == "place":
                counts["num_receptacles"] = configuration[1]
            try:
                if env is None or configuration != current_configuration:
                    if env is not None:
                        env.close()
                        env = None
                    environment_seed = (
                        args.seed if attempt_index == 0 else int(
                            np.random.SeedSequence([args.seed, attempt_index]).generate_state(1)[0]
                        )
                    )
                    env, expert = build_task(args, *configuration, seed=environment_seed)
                    current_configuration = configuration
                observation = env.reset()
            except RandomizationError as error:
                append_jsonl(
                    attempts_path,
                    {
                        "attempt": attempt_index,
                        "session_attempt": attempts,
                        "task": args.task,
                        "seed": args.seed,
                        "environment_seed": environment_seed,
                        **counts,
                        "success": False,
                        "failure_reason": "scene_sampling_failure",
                        "saved_episode": None,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "sampling_stats": getattr(env, "last_sampling_stats", None),
                    },
                )
                print(f"Attempt {attempts}: rejected " "(scene_sampling_failure)")
                continue

            metadata = dict(env.get_task_metadata())
            mode = args.rotation_mode if args.task == "pick" else "align_yaw"
            metadata["control_orientation"] = mode != "none"
            metadata["rotation_mode"] = mode
            metadata["collection_config"] = {"method": "visual_bc", "image_convention": "opencv",
                "camera_names": list(args.camera_names) if args.camera_obs else [],
                "image_size": args.image_size, "input_ref_frame": "world", "joint_noise_degrees": 3,
                "rotation_mode": mode,
                "yaw_range_deg": list(args.yaw_range_deg) if args.task == "pick" else None,
                "fixed_target_color": args.fixed_target_color,
                "controller_config": getattr(env, "bc_controller_config", None),
                "environment_kwargs": {key: np.asarray(getattr(env, key)).tolist() for key in
                    ("x_range", "y_range", "yaw_range", "minimum_object_distance", "cube_half_size", "table_full_size", "table_friction")
                    if hasattr(env, key)}}
            initial_scene = snapshot_scene(observation)
            recorder = TrajectoryRecorder(
                env,
                phase_provider=expert.controller.get_phase,
            )
            recorder.start(observation)
            try:
                try:
                    observation, info = run_expert(
                        args.task, expert, observation, metadata, args.render
                    )
                except Exception as error:
                    info = {
                        "success": False,
                        "stage": "exception",
                        "exception_type": type(error).__name__,
                        "exception_message": str(error),
                    }
            finally:
                recorder.stop()

            if info["success"] and args.camera_obs:
                for name in args.camera_names:
                    frames = np.stack([value[f"{name}_image"] for value in recorder.observations])
                    if len(suspicious_camera_frames(frames)):
                        info = {"success": False, "stage": "camera_corruption", "camera": name}
                        break

            if not info["success"]:
                reason = (
                    info.get("exception_type", "exception")
                    if info.get("stage") == "exception"
                    else failure_reason(info)
                )
                append_jsonl(
                    attempts_path,
                    {
                        "attempt": attempt_index,
                        "session_attempt": attempts,
                        "task": args.task,
                        "seed": args.seed,
                        "environment_seed": environment_seed,
                        **counts,
                        "success": False,
                        "failure_reason": reason,
                        "saved_episode": None,
                        "steps": len(recorder.actions),
                        "metadata": metadata,
                        "initial_scene": initial_scene,
                        "sampling_stats": getattr(env, "last_sampling_stats", None),
                        "result": info,
                    },
                )
                print(f"Attempt {attempts}: rejected ({reason})")
                continue

            episode_index = start_index + saved
            episode_path = task_dir / f"episode_{episode_index:06d}.npz"
            summary = recorder.save(episode_path, metadata, info)
            manifest_entry = {
                "episode": episode_index,
                "task": args.task,
                "instruction": metadata["instruction"],
                "steps": summary["steps"],
                "phase_counts": summary["phase_counts"],
                "path": episode_path.name,
                "metadata": metadata,
                **counts,
                "camera_names": list(args.camera_names) if args.camera_obs else [],
            }
            append_jsonl(manifest_path, manifest_entry)
            append_jsonl(
                attempts_path,
                {
                    "attempt": attempt_index,
                    "session_attempt": attempts,
                    "task": args.task,
                    "seed": args.seed,
                    "environment_seed": environment_seed,
                    **counts,
                    "success": True,
                    "failure_reason": None,
                    "saved_episode": episode_index,
                    "steps": summary["steps"],
                    "metadata": metadata,
                    "camera_names": list(args.camera_names) if args.camera_obs else [],
                    "initial_scene": initial_scene,
                    "sampling_stats": getattr(env, "last_sampling_stats", None),
                    "result": info,
                },
            )
            saved += 1
            saved_configurations.append(configuration)
            print(
                f"Saved {saved}/{args.episodes}: {episode_path} "
                f"({summary['steps']} steps)"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("Collection interrupted; preparing completed demonstrations")
    finally:
        if env is not None:
            env.close()

    collection_summary = {
        "task": args.task, "seed": args.seed, "requested": args.episodes,
        "saved": saved, "attempted": attempts, "start_episode": start_index,
        "interrupted": interrupted,
        "rotation_mode": args.rotation_mode,
        "yaw_range_deg": list(args.yaw_range_deg) if args.task == "pick" else None,
        "target_count_distribution": count_distribution(plan),
        "saved_count_distribution": count_distribution(saved_configurations),
    }
    summary_path = task_dir / f"collection_{existing_attempts:06d}.json"
    summary_path.write_text(json.dumps(collection_summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print("Saved scene counts:", collection_summary["saved_count_distribution"])

    if args.camera_obs and len(existing_entries) + saved >= 2:
        prepare_bc_training(args.output, args.training_dir)
        print(f"Training artifacts ready: {args.training_dir or default_training_dir(args.output)}")
    elif args.camera_obs:
        print("Collect at least two successful demos to create train/validation artifacts")

    if saved < args.episodes:
        raise RuntimeError(
            f"Only recorded {saved}/{args.episodes} successful demos "
            f"after {attempts} attempts"
        )
    print(
        f"Recorded {saved} successful {args.task} demonstrations "
        f"from {attempts} attempts"
    )


if __name__ == "__main__":
    main()
