"""Audit recorded manipulation demonstrations without modifying the dataset."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from data.episode_io import HDF5_SUFFIXES, open_episode, read_json
from data.bc_preparation import default_training_dir

from environments.instructions import TEMPLATES
from environments.entity_utils import COLOR_PALETTE


TASKS = ("pick", "place", "stack")
REQUIRED_POLICY_KEYS = (
    "obs__agentview_image",
    "obs__robot0_joint_pos",
    "obs__robot0_joint_vel",
    "obs__robot0_gripper_qpos",
    "obs__robot0_eef_pos",
    "obs__robot0_eef_quat",
)
CAMERA_KEYS = {
    "agentview": "obs__agentview_image",
    "robot0_eye_in_hand": "obs__robot0_eye_in_hand_image",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit demonstration files, metadata, images, and distributions"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset root containing pick/place/stack directories",
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=list(TASKS)
    )
    parser.add_argument(
        "--camera-names",
        nargs="+",
        choices=tuple(CAMERA_KEYS),
        default=["agentview"],
        help="Camera streams required and audited in every episode",
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
        help="Hash complete HDF5 files to detect byte-identical episodes",
    )
    parser.add_argument(
        "--expected-pick-target-color",
        choices=tuple(COLOR_PALETTE),
        help="Require every audited Pick episode to target this color",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to results/dataset_audit/<dataset-name>",
    )
    parser.add_argument("--split-dir", type=Path,
                        help="Split manifests; automatically uses the dataset's default training directory when available")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for update estimates only (default: 8)")
    parser.add_argument("--sequence-stride", type=int, default=8,
                        help="A/B/C window stride for estimates only (default: 8); BC-RNN uses stride 1")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="A/B/C batches per update for estimates only (default: 1)")
    args = parser.parse_args()
    args.output = args.output or Path("results/dataset_audit") / args.dataset.name
    return args


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
    return read_json(value, field)


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
    num_objects = len(metadata["objects"])
    stats["object_counts"][str(num_objects)] += 1
    if task == "place":
        num_receptacles = len(metadata["receptacles"])
        stats["receptacle_counts"][str(num_receptacles)] += 1
        stats["scene_count_pairs"][f"objects={num_objects},receptacles={num_receptacles}"] += 1
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
        with open_episode(path) as episode:
            keys = set(episode.keys())
            required_camera_keys = {
                CAMERA_KEYS[name] for name in args.camera_names
            }
            required = {
                "actions",
                "metadata_json",
                "result_json",
                *REQUIRED_POLICY_KEYS,
                *required_camera_keys,
            }
            missing = sorted(required - keys)
            if missing:
                errors.append(f"missing keys: {', '.join(missing)}")
                return errors, details, None
            manifest_cameras = manifest_entry.get("camera_names")
            if manifest_cameras is not None and manifest_cameras != args.camera_names:
                errors.append(
                    "manifest cameras do not match audit cameras: "
                    f"{manifest_cameras!r} != {args.camera_names!r}"
                )

            actions = episode["actions"][:]
            metadata = scalar_json(episode["metadata_json"], "metadata_json")
            result = scalar_json(episode["result_json"], "result_json")
            observation_keys = sorted(key for key in keys if key.startswith("obs__"))
            phase_counts = {}

            if actions.ndim != 2:
                errors.append(f"actions must be rank 2, got {actions.shape}")
            elif actions.shape[0] == 0:
                errors.append("trajectory contains no actions")
            if not np.issubdtype(actions.dtype, np.floating):
                errors.append(f"actions must be floating point, got {actions.dtype}")
            if not np.isfinite(actions).all():
                errors.append("actions contain NaN or Inf")

            if "phases" in keys:
                phases = episode["phases"]
                if phases.ndim != 1:
                    errors.append(f"phases must be rank 1, got {phases.shape}")
                elif len(phases) != len(actions):
                    errors.append(
                        f"phases has {len(phases)} labels; expected {len(actions)}"
                    )
                else:
                    phase_values = phases.asstr()[:].tolist()
                    phase_counts = dict(sorted(Counter(phase_values).items()))
                    invalid_phases = sorted(
                        phase
                        for phase in phase_counts
                        if not phase or phase in {"unassigned", "unlabeled"}
                    )
                    if invalid_phases:
                        errors.append(
                            "invalid phase labels: " + ", ".join(invalid_phases)
                        )
            elif "phase_counts" in manifest_entry:
                errors.append("manifest declares phase counts but HDF5 has no phases")

            if "phase_counts" in manifest_entry:
                expected_phase_counts = manifest_entry["phase_counts"]
                if phase_counts != expected_phase_counts:
                    errors.append("manifest phase counts do not match HDF5 phases")

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
                elif not np.isfinite(value[:]).all():
                    errors.append(f"{key} contains NaN or Inf")

            rgb_stats = {}
            for camera_name in args.camera_names:
                camera_key = CAMERA_KEYS[camera_name]
                camera_errors, camera_stats = inspect_rgb(
                    episode[camera_key], args.mode, args.rgb_samples
                )
                errors.extend(
                    f"{camera_name}: {error}" for error in camera_errors
                )
                rgb_stats[camera_name] = camera_stats

            if metadata.get("task") != task:
                errors.append(f"metadata task is {metadata.get('task')!r}, expected {task!r}")
            count_fields = [
                ("num_objects", metadata.get("objects", []), 2 if task == "stack" else 1)
            ]
            if task == "place":
                count_fields.append(("num_receptacles", metadata.get("receptacles", []), 1))
            for field, entities, minimum in count_fields:
                if not minimum <= len(entities) <= 4:
                    errors.append(f"{field}: invalid entity count {len(entities)} for {task}")
                if field in metadata and metadata[field] != len(entities):
                    errors.append(f"{field} does not match metadata entity list")
                if field in manifest_entry and manifest_entry[field] != len(entities):
                    errors.append(f"manifest {field} does not match metadata entity list")
            if manifest_entry.get("task") != task:
                errors.append("manifest task does not match directory")
            if manifest_entry.get("steps") != len(actions):
                errors.append("manifest steps does not match action count")
            if manifest_entry.get("instruction") != metadata.get("instruction"):
                errors.append("manifest instruction does not match HDF5 metadata")
            if manifest_entry.get("metadata") != metadata:
                errors.append("manifest metadata does not match HDF5 metadata")
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
                "phase_counts": phase_counts,
                "metadata": metadata,
                "rgb": rgb_stats,
            }
            return errors, details, metadata
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return [f"cannot read episode: {type(error).__name__}: {error}"], details, None


def new_semantic_stats():
    return defaultdict(Counter)


def serializable_semantics(stats):
    return {
        name: dict(sorted(counter.items()))
        for name, counter in sorted(stats.items())
    }


def trajectory_statistics(lengths):
    return {
        "count": len(lengths), "total": int(sum(lengths)),
        "mean": float(np.mean(lengths)) if lengths else None,
        "median": float(np.median(lengths)) if lengths else None,
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "p95": float(np.percentile(lengths, 95)) if lengths else None,
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
    episode_lengths = []
    action_min = None
    action_max = None
    phase_counts = Counter()
    audited_paths = set()

    manifest_paths = [entry.get("path") for entry in all_entries]
    duplicate_manifest_paths = sorted(
        path for path, count in Counter(manifest_paths).items() if path and count > 1
    )
    for duplicate in duplicate_manifest_paths:
        errors.append({"path": duplicate, "error": "duplicate path in manifest"})

    for index, entry in enumerate(manifest_entries):
        relative_path = entry.get("path")
        length_row = {"task": task, "episode": entry.get("episode"),
                      "path": relative_path, "manifest_steps": entry.get("steps"), "steps": None}
        episode_lengths.append(length_row)
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
            length_row["steps"] = details["steps"]
            phase_counts.update(details["phase_counts"])
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
            if task == "pick" and args.expected_pick_target_color is not None:
                try:
                    target = entity_by_id(
                        metadata["objects"], metadata["target_object"]
                    )
                    if target["color"] != args.expected_pick_target_color:
                        errors.append(
                            {
                                "path": relative_path,
                                "error": (
                                    f"target color is {target['color']!r}, expected "
                                    f"{args.expected_pick_target_color!r}"
                                ),
                            }
                        )
                    if metadata.get("policy_conditioning") != "vision_only":
                        errors.append(
                            {
                                "path": relative_path,
                                "error": "fixed-color Pick metadata is not vision_only",
                            }
                        )
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(
                        {
                            "path": relative_path,
                            "error": f"fixed target metadata: {error}",
                        }
                    )
            try:
                add_semantics(semantics, task, metadata)
            except (KeyError, TypeError, ValueError) as error:
                errors.append({"path": relative_path, "error": f"semantic stats: {error}"})
        if args.check_duplicates:
            digest_paths[file_digest(path)].append(f"{task}/{relative_path}")

    orphan_files = []
    if args.limit_per_task is None:
        manifest_path_set = {path for path in manifest_paths if isinstance(path, str)}
        disk_path_set = {path.name for path in task_dir.glob("episode_*") if path.suffix in HDF5_SUFFIXES or path.suffix == ".npz"}
        orphan_files = sorted(disk_path_set - manifest_path_set)
        for orphan in orphan_files:
            errors.append({"path": orphan, "error": "episode is not listed in manifest"})

    error_paths = {item["path"] for item in errors}
    for row in episode_lengths:
        row["valid"] = row["steps"] is not None and isinstance(row["path"], str) and row["path"] not in error_paths
    return {
        "manifest_entries": len(all_entries),
        "audited_entries": len(manifest_entries),
        "valid_episodes": len(manifest_entries) - len({item["path"] for item in errors}),
        "error_count": len(errors),
        "errors": errors,
        "trajectory_steps": trajectory_statistics(lengths),
        "episode_lengths": episode_lengths,
        "action_min": action_min.tolist() if action_min is not None else None,
        "action_max": action_max.tolist() if action_max is not None else None,
        "phase_counts": dict(sorted(phase_counts.items())),
        "semantics": serializable_semantics(semantics),
        "orphan_files": orphan_files,
    }


def audit_split_lengths(args, task_results):
    """Count native action timesteps, checking split membership and step metadata.

    Reads only action shapes for episodes outside a limited audit; never scans
    RGB twice. All estimates are training updates, not observation frame counts.
    """
    directory = args.split_dir or default_training_dir(args.dataset)
    result = {"directory": str(directory), "status": "unavailable", "errors": [], "splits": {},
              "assumptions": {"batch_size": args.batch_size, "visual_bc_sequence_stride": args.sequence_stride,
                              "visual_bc_gradient_accumulation_steps": args.gradient_accumulation_steps,
                              "bcrnn_sequence_stride": 1, "bcrnn_drop_last": True}}
    if not directory.exists() and args.split_dir is None:
        return result
    known = set()
    for task in task_results:
        entries, _ = read_jsonl(args.dataset / task / "manifest.jsonl")
        known.update(f"{task}/{entry['path']}" for entry in entries if isinstance(entry.get("path"), str))
    cached = {f"{task}/{row['path']}": row["steps"] for task, value in task_results.items()
              for row in value["episode_lengths"] if row["steps"] is not None}
    seen = set()
    for split in ("train", "val", "test"):
        path = directory / f"{split}.jsonl"
        if split == "test" and not path.exists():
            continue
        entries, errors = read_jsonl(path)
        result["errors"].extend(errors)
        lengths = []
        for entry in entries:
            if entry.get("task") not in task_results:
                continue
            name = entry.get("path")
            if not isinstance(name, str) or name not in known:
                result["errors"].append(f"{split}: path is not in dataset manifests: {name}")
                continue
            if name in seen:
                result["errors"].append(f"{split}: duplicate/overlapping split episode: {name}")
                continue
            seen.add(name)
            try:
                if name not in cached:
                    with open_episode(args.dataset / name) as episode:
                        shape = episode["actions"].shape
                        if len(shape) != 2 or shape[1] != 7 or shape[0] <= 0:
                            raise ValueError(f"Invalid action shape: {shape}")
                        cached[name] = int(shape[0])
                steps = cached[name]
                if entry.get("steps") != steps:
                    result["errors"].append(f"{split}/{name}: split steps differ from HDF5 action count")
                lengths.append(steps)
            except (OSError, KeyError, ValueError, TypeError) as error:
                result["errors"].append(f"{split}/{name}: {error}")
        values = {"trajectory_steps": trajectory_statistics(lengths)}
        if split == "train" and lengths:
            windows = sum((n + args.sequence_stride - 1) // args.sequence_stride for n in lengths)
            batches = (windows + args.batch_size - 1) // args.batch_size
            updates = (batches + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
            values["training_budget"] = {"visual_bc_windows": windows, "visual_bc_batches_per_epoch": batches,
                "visual_bc_updates_per_epoch": updates, "visual_bc_updates_in_100_epochs": 100 * updates,
                "bcrnn_windows": sum(lengths), "bcrnn_updates_per_full_pass": sum(lengths) // args.batch_size,
                "bcrnn_updates_in_100_full_passes": 100 * (sum(lengths) // args.batch_size)}
        result["splits"][split] = values
    missing = known - seen
    if missing:
        result["errors"].append(f"{len(missing)} dataset episodes are not assigned to a split")
    result["status"] = "fail" if result["errors"] else "pass"
    return result


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
        "Step counts are action counts T, not T+1 observations. Statistics cover readable audited episodes.",
        "",
        "| Task | Audited | Valid | Total T | Mean T | Median T | Min T | Max T | P95 T | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, result in report["tasks"].items():
        trajectory = result["trajectory_steps"]
        mean = f"{trajectory['mean']:.1f}" if trajectory["mean"] is not None else "-"
        lines.append(
            f"| {task.title()} | {result['audited_entries']} | "
            f"{result['valid_episodes']} | {trajectory['total']} | {mean} | {trajectory['median']} | "
            f"{trajectory['min'] or '-'} | {trajectory['max'] or '-'} | "
            f"{trajectory['p95']} | {result['error_count']} |"
        )

    lines += ["", f"- Combined audited total: {report['trajectory_steps']['total']} action steps.",
              "- Per-episode lengths: `episode_lengths.csv`.", "", "## Split Lengths and Training Budget", ""]
    split_report = report["split_summary"]
    lines.append(f"- Split directory: `{split_report['directory']}`; status: {split_report['status']}.")
    lines.append(f"- Estimate assumptions: `{json.dumps(split_report['assumptions'], sort_keys=True)}`.")
    for split, values in split_report["splits"].items():
        stats = values["trajectory_steps"]
        lines.append(f"- **{split}**: {stats['count']} episodes, {stats['total']} action steps, mean {stats['mean']}.")
        if "training_budget" in values:
            budget = values["training_budget"]
            lines.extend([
                f"  - A/B/C: {budget['visual_bc_updates_per_epoch']} updates per full epoch; {budget['visual_bc_updates_in_100_epochs']} in 100 epochs.",
                f"  - BC-RNN: {budget['bcrnn_updates_per_full_pass']} updates per full dataset pass; {budget['bcrnn_updates_in_100_full_passes']} in 100 passes.",
                "  - BC-RNN currently uses --max-updates; full passes are a budget estimate, not an --epochs option.",
            ])

    lines += ["", "## Semantic Distributions", ""]
    for task, result in report["tasks"].items():
        lines.append(f"### {task.title()}")
        lines.append("")
        for name, distribution in result["semantics"].items():
            values = ", ".join(f"{key}: {value}" for key, value in distribution.items())
            lines.append(f"- `{name}`: {values or '-'}")
        lines.append("")

    lines += ["## Phase Distributions", ""]
    for task, result in report["tasks"].items():
        values = ", ".join(
            f"{phase}: {count}"
            for phase, count in result["phase_counts"].items()
        )
        lines.append(f"- **{task.title()}**: {values or '-'}")
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
    for error in split_report["errors"]:
        any_errors = True
        lines.append(f"- Split: {error}")
    if not any_errors:
        lines.append("No errors found.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.camera_names[0] != "agentview":
        raise ValueError("--camera-names must start with agentview")
    if len(set(args.camera_names)) != len(args.camera_names):
        raise ValueError("--camera-names cannot contain duplicates")
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        raise ValueError("--limit-per-task must be positive")
    if args.rgb_samples <= 0:
        raise ValueError("--rgb-samples must be positive")
    if min(args.batch_size, args.sequence_stride, args.gradient_accumulation_steps) <= 0:
        raise ValueError("Batch size, sequence stride, and accumulation steps must be positive")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")

    digest_paths = defaultdict(list)
    task_results = {}
    for task in args.tasks:
        print(f"Auditing {task}...")
        task_results[task] = audit_task(args.dataset, task, args, digest_paths)

    duplicates = [paths for paths in digest_paths.values() if len(paths) > 1]
    split_summary = audit_split_lengths(args, task_results)
    error_count = sum(result["error_count"] for result in task_results.values())
    error_count += len(duplicates)
    error_count += len(split_summary["errors"])
    length_rows = [row for value in task_results.values() for row in value["episode_lengths"]]
    report = {
        "dataset": str(args.dataset),
        "mode": args.mode,
        "limit_per_task": args.limit_per_task,
        "duplicate_check": args.check_duplicates,
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "tasks": task_results,
        "duplicate_files": duplicates,
        "trajectory_steps": trajectory_statistics([row["steps"] for row in length_rows if row["steps"] is not None]),
        "split_summary": split_summary,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "audit_report.json"
    markdown_path = args.output / "audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    csv_path = args.output / "episode_lengths.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "episode", "path", "manifest_steps", "steps", "valid"))
        writer.writeheader()
        writer.writerows(length_rows)

    print(f"Audit status: {report['status'].upper()} ({error_count} errors)")
    for task, value in task_results.items():
        stats = value["trajectory_steps"]
        print(f"{task}: {stats['count']} readable episodes; total={stats['total']} action steps; "
              f"min={stats['min']} mean={stats['mean']} median={stats['median']} max={stats['max']} p95={stats['p95']}")
    if split_summary["status"] == "unavailable":
        print("Split manifests unavailable; use --split-dir to obtain training update estimates")
    for split, values in split_summary["splits"].items():
        stats = values["trajectory_steps"]
        print(f"{split}: {stats['count']} episodes; {stats['total']} action steps")
        if "training_budget" in values:
            budget = values["training_budget"]
            print(f"  A/B/C updates/full epoch: {budget['visual_bc_updates_per_epoch']}; "
                  f"100 epochs: {budget['visual_bc_updates_in_100_epochs']}")
            print(f"  BC-RNN updates/full dataset pass: {budget['bcrnn_updates_per_full_pass']}; "
                  f"100 passes: {budget['bcrnn_updates_in_100_full_passes']}")
    print(f"Update estimate assumptions: {split_summary['assumptions']}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Per-episode CSV: {csv_path}")
    raise SystemExit(0 if error_count == 0 else 1)


if __name__ == "__main__":
    main()
