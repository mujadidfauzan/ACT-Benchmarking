import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from robosuite import load_composite_controller_config

from controllers.panda_controller import PandaController
from environments.pick_env import PickEnv
from environments.place_env import PlaceEnv
from environments.stack_env import StackEnv
from environments.scene_counts import (
    add_count_arguments, balanced_count_plan, count_label, count_options,
)
from experts.pick_expert import PickExpert
from experts.place_expert import PlaceExpert
from experts.stack_expert import StackExpert


TASK_DEFAULTS = {
    "pick": {"seed_start": 1000, "gate": 0.95},
    "place": {"seed_start": 2000, "gate": 0.90},
    "stack": {"seed_start": 3000, "gate": 0.85},
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark scripted experts")
    parser.add_argument(
        "--task", choices=("all", "pick", "place", "stack"), default="all"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to each task's fixed seed range",
    )
    parser.add_argument("--output", type=Path, default=Path("results/expert_reliability"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--instruction-split", choices=("train", "held_out"), default="train")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    add_count_arguments(parser)
    parser.add_argument("--fixed-target-color", choices=tuple(PickEnv.COLOR_PALETTE))
    return parser.parse_args()


def build_task(task, seed, instruction_split, render, num_objects=3,
               num_receptacles=3, fixed_target_color=None):
    config = load_composite_controller_config(controller="BASIC")
    common = {
        "robots": "Panda",
        "controller_configs": config,
        "instruction_split": instruction_split,
        "has_renderer": render,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "use_object_obs": True,
        "control_freq": 20,
        "hard_reset": False,
        "seed": seed,
    }
    if task == "pick":
        env = PickEnv(num_objects=num_objects, fixed_target_color=fixed_target_color,
                      horizon=1500, **common)
    elif task == "place":
        env = PlaceEnv(num_objects=num_objects, num_receptacles=num_receptacles,
                       horizon=3000, **common)
    else:
        env = StackEnv(num_cubes=num_objects, horizon=6000, **common)

    controller = PandaController(env)
    pick_expert = PickExpert(controller)
    if task == "pick":
        expert = pick_expert
    elif task == "place":
        expert = PlaceExpert(controller, pick_expert)
    else:
        expert = StackExpert(controller, pick_expert)
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


def snapshot_scene(observation):
    return {
        key: np.asarray(value).copy()
        for key, value in observation.items()
        if key.endswith("_pos") or key.endswith("_quat")
    }


def target_snapshot(task, env, metadata):
    if task == "pick":
        object_id = metadata["target_object"]
        return {"target_object_position": env.get_object_position(object_id)}
    if task == "place":
        object_id = metadata["target_object"]
        receptacle_id = metadata["target_receptacle"]
        return {
            "target_object_position": env.get_object_position(object_id),
            "target_receptacle_position": env.get_receptacle_position(receptacle_id),
        }
    return {
        "stack_order": metadata["stack_order"],
        "stack_positions": {
            color: env.get_object_position(color) for color in metadata["stack_order"]
        },
    }


def classify_failure(task, info):
    if info.get("success"):
        return None
    stage = info.get("stage", "unknown")
    detail = info.get("detail") or {}

    if stage == "exception":
        return info.get("exception_type", "exception")
    if stage == "timeout":
        return "timeout"
    if stage == "pick":
        pick_stage = detail.get("stage", "verification")
        return f"pick_{pick_stage}"
    if task == "pick":
        return f"pick_{'verification' if stage == 'complete' else stage}"
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
    return f"{task}_{stage}"


def benchmark_task(
    task,
    episodes,
    run_dir,
    instruction_split,
    render,
    seed_offset=0,
    object_counts=(3,),
    receptacle_counts=(None,),
    fixed_target_color=None,
):
    attempts_path = run_dir / f"{task}_attempts.jsonl"
    successes = 0
    failure_counts = Counter()
    if task == "place" and receptacle_counts == (None,):
        receptacle_counts = (3,)
    plan = balanced_count_plan(episodes, object_counts, receptacle_counts,
                               TASK_DEFAULTS[task]["seed_start"] + seed_offset)
    groups = {}

    with attempts_path.open("w", encoding="utf-8") as attempts_file:
        for episode_index in range(episodes):
            num_objects, num_receptacles = configuration = plan[episode_index]
            seed = (
                TASK_DEFAULTS[task]["seed_start"]
                + seed_offset
                + episode_index
            )
            env = None
            metadata = {"task": task}
            initial_scene = {}
            target_state = {}
            info = {"success": False, "stage": "exception"}
            steps = 0
            try:
                env, expert = build_task(task, seed, instruction_split, render,
                                         num_objects, num_receptacles, fixed_target_color)
                observation = env.reset()
                metadata = env.get_task_metadata()
                initial_scene = snapshot_scene(observation)
                target_state = target_snapshot(task, env, metadata)
                observation, info = run_expert(
                    task, expert, observation, metadata, render
                )
                steps = int(env.timestep)
            except Exception as error:
                info = {
                    "success": False,
                    "stage": "exception",
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
                if env is not None:
                    steps = int(getattr(env, "timestep", 0))
            finally:
                if env is not None:
                    env.close()

            failure_reason = classify_failure(task, info)
            success = bool(info.get("success", False))
            successes += int(success)
            if failure_reason is not None:
                failure_counts[failure_reason] += 1
            group = groups.setdefault(count_label(configuration), {
                "num_objects": num_objects, "num_receptacles": num_receptacles,
                "attempted": 0, "successful": 0, "failure_counts": Counter(),
            })
            group["attempted"] += 1
            group["successful"] += int(success)
            if failure_reason is not None:
                group["failure_counts"][failure_reason] += 1

            entry = {
                "attempt": episode_index,
                "seed": seed,
                "task": task,
                "num_objects": num_objects,
                "num_receptacles": num_receptacles,
                "success": success,
                "failure_reason": failure_reason,
                "steps": steps,
                "metadata": metadata,
                "initial_scene": initial_scene,
                "target_state": target_state,
                "result": info,
            }
            attempts_file.write(
                json.dumps(entry, default=_json_default, sort_keys=True) + "\n"
            )
            print(
                f"[{task}] {episode_index + 1}/{episodes} seed={seed} "
                f"success={success} failure={failure_reason}"
            )

    success_rate = successes / episodes
    gate = TASK_DEFAULTS[task]["gate"]
    by_scene_counts = {
        label: {**values, "success_rate": values["successful"] / values["attempted"],
                "required_success_rate": gate,
                "gate_passed": values["successful"] / values["attempted"] >= gate,
                "failure_counts": dict(sorted(values["failure_counts"].items()))}
        for label, values in sorted(groups.items())
    }
    return {
        "task": task,
        "attempted": episodes,
        "successful": successes,
        "failed": episodes - successes,
        "success_rate": success_rate,
        "required_success_rate": gate,
        "gate_passed": success_rate >= gate and all(
            group["gate_passed"] for group in by_scene_counts.values()
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "seed_start": TASK_DEFAULTS[task]["seed_start"] + seed_offset,
        "seed_end": (
            TASK_DEFAULTS[task]["seed_start"] + seed_offset + episodes - 1
        ),
        "attempts_log": attempts_path.name,
        "by_scene_counts": by_scene_counts,
    }


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.fixed_target_color is not None and args.task != "pick":
        raise ValueError("--fixed-target-color requires --task pick")
    tasks = ("pick", "place", "stack") if args.task == "all" else (args.task,)
    configurations = {task: count_options(task, args) for task in tasks}
    for task, (objects, receptacles) in configurations.items():
        balanced_count_plan(args.episodes, objects, receptacles)
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = args.output / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    summaries = [
        benchmark_task(
            task,
            args.episodes,
            run_dir,
            args.instruction_split,
            args.render,
            args.seed_offset,
            *configurations[task],
            args.fixed_target_color,
        )
        for task in tasks
    ]
    result = {
        "run_name": run_name,
        "instruction_split": args.instruction_split,
        "episodes_per_task": args.episodes,
        "all_gates_passed": all(item["gate_passed"] for item in summaries),
        "tasks": {item["task"]: item for item in summaries},
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=" * 72)
    for item in summaries:
        print(
            f"{item['task']:<6} {item['successful']}/{item['attempted']} "
            f"({item['success_rate']:.1%}) gate={item['required_success_rate']:.0%} "
            f"passed={item['gate_passed']} failures={item['failure_counts']}"
        )
    print("Summary:", summary_path)

    if args.enforce_gates and not result["all_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
