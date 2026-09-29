"""Audit recorded manipulation demonstrations without modifying the dataset."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from language.instruction_generator import TEMPLATES


TASKS = ("pick", "place", "stack")
REQUIRED_POLICY_KEYS = (
    "obs__agentview_image",
    "obs__robot0_joint_pos",
    "obs__robot0_joint_vel",
    "obs__robot0_gripper_qpos",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit demonstration files, metadata, images, and distributions"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/demos/pilot_v01"),
        help="Dataset root containing pick/place/stack directories",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=list(TASKS)
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="Quick samples RGB frames; full checks every RGB frame",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=None,
        help="Audit only the first N manifest entries per task",
    )
    parser.add_argument(
        "--rgb-samples",
        type=int,
        default=5,
        help="Representative RGB frames checked per episode in quick mode",
    )
    parser.add_argument(
        "--check-duplicates",
        action="store_true",
        help="Hash complete NPZ files to detect byte-identical episodes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_audit/pilot_v01"),
    )
    return parser.parse_args()


def read_jsonl(path):
    entries = []
    errors = []
    if not path.exists():
        return entries, [f"missing manifest: {path}"]
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"manifest line {line_number}: {error}")
    return entries, errors


def scalar_json(value, field):
    try:
        raw = value.item()
    except ValueError as error:
        raise ValueError(f"{field} must be a scalar JSON string") from error
    if not isinstance(raw, str):
        raise ValueError(f"{field} must contain a JSON string")
    return json.loads(raw)


def entity_by_id(entities, entity_id):
    matches = [entity for entity in entities if entity.get("id") == entity_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one entity with id {entity_id!r}")
    return matches[0]


def join_colors(colors):
    if len(colors) == 2:
        return f"{colors[0]} and {colors[1]}"
    return ", ".join(colors[:-1]) + f", and {colors[-1]}"


def position_phrase(colors):
    labels = ["at the bottom"]
    if len(colors) == 3:
        labels += ["in the middle", "on top"]
    else:
        labels += [f"at level {index + 1}" for index in range(1, len(colors) - 1)]
        labels += ["on top"]
    return ", ".join(
        f"{color} {label}" for color, label in zip(colors, labels)
    )


def relation_phrase(colors):
    return ", then ".join(
        f"the {top} cube is on the {support} cube"
        for support, top in zip(colors[:-1], colors[1:])
    )


def expected_instruction(metadata):
    task = metadata["task"]
    split = metadata["instruction_split"]
    template_id = metadata["instruction_template_id"]
    templates = dict(TEMPLATES.get(task, {}).get(split, []))
    if template_id not in templates:
        raise ValueError(
            f"unknown template {template_id!r} for {task}/{split}"
        )

    if task == "pick":
        target = entity_by_id(metadata["objects"], metadata["target_object"])
        values = {"color": target["color"]}
    elif task == "place":
        target = entity_by_id(metadata["objects"], metadata["target_object"])
        receptacle = entity_by_id(
            metadata["receptacles"], metadata["target_receptacle"]
        )
        values = {
            "object_color": target["color"],
            "target_color": receptacle["color"],
        }
    elif task == "stack":
        colors = list(metadata["stack_order"])
        values = {
            "colors": join_colors(colors),
            "positions": position_phrase(colors),
            "relations": relation_phrase(colors),
        }
    else:
        raise ValueError(f"unknown task {task!r}")
    return templates[template_id].format(**values)


def frame_indices(frame_count, mode, sample_count):
    if mode == "full" or frame_count <= sample_count:
        return np.arange(frame_count)
    return np.unique(np.linspace(0, frame_count - 1, sample_count, dtype=int))


def inspect_rgb(images, mode, sample_count):
    errors = []
    if images.ndim != 4 or images.shape[-1] != 3:
        return [f"RGB shape must be (T+1, H, W, 3), got {images.shape}"], {}
    if images.dtype != np.uint8:
        errors.append(f"RGB dtype must be uint8, got {images.dtype}")

    indices = frame_indices(len(images), mode, sample_count)
    black_frames = 0
    constant_frames = 0
    channel_min = np.full(3, 255, dtype=np.int64)
    channel_max = np.zeros(3, dtype=np.int64)
    means = []
    for index in indices:
        frame = images[index]
        channel_min = np.minimum(channel_min, frame.min(axis=(0, 1)))
        channel_max = np.maximum(channel_max, frame.max(axis=(0, 1)))
        means.append(float(frame.mean()))
        if int(frame.max()) == 0:
            black_frames += 1
        if int(frame.max()) == int(frame.min()):
            constant_frames += 1

    if black_frames:
        errors.append(f"{black_frames}/{len(indices)} checked RGB frames are black")
    if constant_frames:
        errors.append(
            f"{constant_frames}/{len(indices)} checked RGB frames are constant"
        )
    return errors, {
        "checked_frames": int(len(indices)),
        "black_frames": black_frames,
        "constant_frames": constant_frames,
        "channel_min": channel_min.tolist(),
        "channel_max": channel_max.tolist(),
        "mean_intensity_min": min(means) if means else None,
        "mean_intensity_max": max(means) if means else None,
    }


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_semantics(stats, task, metadata):
    if task == "pick":
        target = entity_by_id(metadata["objects"], metadata["target_object"])
        stats["target_colors"][target["color"]] += 1
        for entity in metadata["objects"]:
            if entity["id"] != metadata["target_object"]:
                stats["distractor_colors"][entity["color"]] += 1
    elif task == "place":
        target = entity_by_id(metadata["objects"], metadata["target_object"])
        receptacle = entity_by_id(
            metadata["receptacles"], metadata["target_receptacle"]
        )
        source_color = target["color"]
        target_color = receptacle["color"]
        stats["object_to_receptacle"][f"{source_color}->{target_color}"] += 1
        stats["target_object_colors"][source_color] += 1
        stats["target_receptacle_colors"][target_color] += 1
        for entity in metadata["objects"]:
            stats["all_object_colors"][entity["color"]] += 1
        for entity in metadata["receptacles"]:
            stats["all_receptacle_colors"][entity["color"]] += 1
    else:
        order = list(metadata["stack_order"])
        stats["stack_orders"][">".join(order)] += 1
        for color in order:
            stats["stack_color_occurrences"][color] += 1

    stats["instruction_templates"][metadata["instruction_template_id"]] += 1
    stats["instruction_splits"][metadata["instruction_split"]] += 1
    stats["semantic_splits"][metadata["semantic_split"]] += 1


def audit_episode(path, task, manifest_entry, args):
    errors = []
    details = {}
    try:
        with np.load(path, allow_pickle=False) as episode:
            keys = set(episode.files)
            required = {"actions", "metadata_json", "result_json", *REQUIRED_POLICY_KEYS}
            missing = sorted(required - keys)
            if missing:
                errors.append(f"missing keys: {', '.join(missing)}")
                return errors, details, None

            actions = episode["actions"]
            metadata = scalar_json(episode["metadata_json"], "metadata_json")
            result = scalar_json(episode["result_json"], "result_json")
            observation_keys = sorted(key for key in keys if key.startswith("obs__"))

            if actions.ndim != 2:
                errors.append(f"actions must be rank 2, got {actions.shape}")
            elif actions.shape[0] == 0:
                errors.append("trajectory contains no actions")
            if not np.issubdtype(actions.dtype, np.floating):
                errors.append(f"actions must be floating point, got {actions.dtype}")
            if not np.isfinite(actions).all():
                errors.append("actions contain NaN or Inf")

            expected_observations = len(actions) + 1
            for key in observation_keys:
                value = episode[key]
                if value.ndim == 0 or value.shape[0] != expected_observations:
                    errors.append(
                        f"{key} has {value.shape[0] if value.ndim else 0} frames; "
                        f"expected {expected_observations}"
                    )
            for key in REQUIRED_POLICY_KEYS[1:]:
                value = episode[key]
                if not np.issubdtype(value.dtype, np.number):
                    errors.append(f"{key} must be numeric, got {value.dtype}")
                elif not np.isfinite(value).all():
                    errors.append(f"{key} contains NaN or Inf")

            rgb_errors, rgb_stats = inspect_rgb(
                episode["obs__agentview_image"], args.mode, args.rgb_samples
            )
            errors.extend(rgb_errors)

            if metadata.get("task") != task:
                errors.append(f"metadata task is {metadata.get('task')!r}, expected {task!r}")
            if manifest_entry.get("task") != task:
                errors.append("manifest task does not match directory")
            if manifest_entry.get("steps") != len(actions):
                errors.append("manifest steps does not match action count")
            if manifest_entry.get("instruction") != metadata.get("instruction"):
                errors.append("manifest instruction does not match NPZ metadata")
            if manifest_entry.get("metadata") != metadata:
                errors.append("manifest metadata does not match NPZ metadata")
            try:
                generated = expected_instruction(metadata)
                if generated != metadata.get("instruction"):
                    errors.append(
                        "instruction does not match metadata/template: "
                        f"expected {generated!r}"
                    )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"invalid semantic metadata: {error}")

            if result.get("success") is not True:
                errors.append("saved episode result.success is not true")
            if result.get("stage") != "complete":
                errors.append(
                    f"saved episode result stage is {result.get('stage')!r}, expected 'complete'"
                )

            action_min = np.min(actions, axis=0).tolist() if len(actions) else None
            action_max = np.max(actions, axis=0).tolist() if len(actions) else None
            details = {
                "steps": int(len(actions)),
                "action_shape": list(actions.shape),
                "action_min": action_min,
                "action_max": action_max,
                "metadata": metadata,
                "rgb": rgb_stats,
            }
            return errors, details, metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return [f"cannot read episode: {type(error).__name__}: {error}"], details, None


def new_semantic_stats():
    return defaultdict(Counter)


def serializable_semantics(stats):
    return {
        name: dict(sorted(counter.items()))
        for name, counter in sorted(stats.items())
    }


def audit_task(dataset_root, task, args, digest_paths):
    task_dir = dataset_root / task
    manifest_entries, manifest_errors = read_jsonl(task_dir / "manifest.jsonl")
    all_entries = manifest_entries
    if args.limit_per_task is not None:
        manifest_entries = manifest_entries[: args.limit_per_task]

    errors = [{"path": "manifest.jsonl", "error": error} for error in manifest_errors]
    semantics = new_semantic_stats()
    lengths = []
    action_min = None
    action_max = None
    audited_paths = set()

    manifest_paths = [entry.get("path") for entry in all_entries]
    duplicate_manifest_paths = sorted(
        path for path, count in Counter(manifest_paths).items() if path and count > 1
    )
    for duplicate in duplicate_manifest_paths:
        errors.append({"path": duplicate, "error": "duplicate path in manifest"})

    for index, entry in enumerate(manifest_entries):
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            errors.append({"path": f"manifest[{index}]", "error": "missing path"})
            continue
        path = task_dir / relative_path
        audited_paths.add(path.name)
        if not path.is_file():
            errors.append({"path": relative_path, "error": "episode file is missing"})
            continue

        episode_errors, details, metadata = audit_episode(path, task, entry, args)
        errors.extend(
            {"path": relative_path, "error": error} for error in episode_errors
        )
        if details:
            lengths.append(details["steps"])
            if details["action_min"] is not None:
                current_min = np.asarray(details["action_min"], dtype=np.float64)
                current_max = np.asarray(details["action_max"], dtype=np.float64)
                if action_min is None:
                    action_min = current_min
                    action_max = current_max
                elif action_min.shape == current_min.shape:
                    action_min = np.minimum(action_min, current_min)
                    action_max = np.maximum(action_max, current_max)
                else:
                    errors.append(
                        {
                            "path": relative_path,
                            "error": "action dimension differs from earlier episodes",
                        }
                    )
        if metadata is not None:
            try:
                add_semantics(semantics, task, metadata)
            except (KeyError, TypeError, ValueError) as error:
                errors.append({"path": relative_path, "error": f"semantic stats: {error}"})
        if args.check_duplicates:
            digest_paths[file_digest(path)].append(f"{task}/{relative_path}")

    orphan_files = []
    if args.limit_per_task is None:
        manifest_path_set = {path for path in manifest_paths if isinstance(path, str)}
        disk_path_set = {path.name for path in task_dir.glob("episode_*.npz")}
        orphan_files = sorted(disk_path_set - manifest_path_set)
        for orphan in orphan_files:
            errors.append({"path": orphan, "error": "episode is not listed in manifest"})

    trajectory = {
        "count": len(lengths),
        "mean": float(np.mean(lengths)) if lengths else None,
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
    }
    return {
        "manifest_entries": len(all_entries),
        "audited_entries": len(manifest_entries),
        "valid_episodes": len(manifest_entries) - len({item["path"] for item in errors}),
        "error_count": len(errors),
        "errors": errors,
        "trajectory_steps": trajectory,
        "action_min": action_min.tolist() if action_min is not None else None,
        "action_max": action_max.tolist() if action_max is not None else None,
        "semantics": serializable_semantics(semantics),
        "orphan_files": orphan_files,
    }


def markdown_report(report):
    lines = [
        "# Dataset Audit",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Mode: `{report['mode']}`",
        f"- Status: **{report['status'].upper()}**",
        "",
        "## Trajectory Summary",
        "",
        "| Task | Audited | Valid | Mean T | Min T | Max T | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, result in report["tasks"].items():
        trajectory = result["trajectory_steps"]
        mean = f"{trajectory['mean']:.1f}" if trajectory["mean"] is not None else "-"
        lines.append(
            f"| {task.title()} | {result['audited_entries']} | "
            f"{result['valid_episodes']} | {mean} | "
            f"{trajectory['min'] or '-'} | {trajectory['max'] or '-'} | "
            f"{result['error_count']} |"
        )

    lines += ["", "## Semantic Distributions", ""]
    for task, result in report["tasks"].items():
        lines.append(f"### {task.title()}")
        lines.append("")
        for name, distribution in result["semantics"].items():
            values = ", ".join(f"{key}: {value}" for key, value in distribution.items())
            lines.append(f"- `{name}`: {values or '-'}")
        lines.append("")

    lines += ["## Errors", ""]
    any_errors = False
    for task, result in report["tasks"].items():
        for item in result["errors"]:
            any_errors = True
            lines.append(f"- `{task}/{item['path']}`: {item['error']}")
    for group in report["duplicate_files"]:
        any_errors = True
        lines.append(f"- Duplicate files: {', '.join(f'`{path}`' for path in group)}")
    if not any_errors:
        lines.append("No errors found.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        raise ValueError("--limit-per-task must be positive")
    if args.rgb_samples <= 0:
        raise ValueError("--rgb-samples must be positive")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")

    digest_paths = defaultdict(list)
    task_results = {}
    for task in args.tasks:
        print(f"Auditing {task}...")
        task_results[task] = audit_task(args.dataset, task, args, digest_paths)

    duplicates = [paths for paths in digest_paths.values() if len(paths) > 1]
    error_count = sum(result["error_count"] for result in task_results.values())
    error_count += len(duplicates)
    report = {
        "dataset": str(args.dataset),
        "mode": args.mode,
        "limit_per_task": args.limit_per_task,
        "duplicate_check": args.check_duplicates,
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "tasks": task_results,
        "duplicate_files": duplicates,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "audit_report.json"
    markdown_path = args.output / "audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")

    print(f"Audit status: {report['status'].upper()} ({error_count} errors)")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    raise SystemExit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
