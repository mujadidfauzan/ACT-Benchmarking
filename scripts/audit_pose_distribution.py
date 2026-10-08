"""Compare initial object pose distributions across dataset splits."""

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


SPLITS = ("train", "val", "test")
POSITION_FIELDS = ("x", "y", "z")
ANGLE_FIELDS = ("roll_deg", "pitch_deg", "yaw_deg")
PLOT_COLORS = {
    "red": "#d62728",
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "yellow": "#d4a900",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit initial object positions and orientations by split"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pose_distribution"),
    )
    parser.add_argument(
        "--include-receptacles",
        action="store_true",
        help="Include receptacles in addition to manipulable objects",
    )
    return parser.parse_args()


def read_jsonl(path):
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return entries


def scalar_json(value, field):
    raw = value.item()
    if not isinstance(raw, str):
        raise ValueError(f"{field} must contain a scalar JSON string")
    return json.loads(raw)


def quat_xyzw_to_euler_deg(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((x, y, z, w))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Invalid quaternion: {quaternion}")
    x, y, z, w = (x / norm, y / norm, z / norm, w / norm)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return np.rad2deg((roll, pitch, yaw))


def initial_vector(data, entity_id, suffix):
    key = f"obs__{entity_id}_{suffix}"
    if key not in data:
        raise KeyError(f"Missing observation {key}")
    values = data[key]
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError(f"{key} must have shape (T+1, D), got {values.shape}")
    return values[0]


def episode_rows(dataset, split, entry, include_receptacles):
    episode_path = dataset / entry["path"]
    if not episode_path.is_file():
        raise FileNotFoundError(f"Missing episode: {episode_path}")
    with np.load(episode_path, allow_pickle=False) as data:
        metadata = scalar_json(data["metadata_json"], "metadata_json")
        groups = [("object", metadata.get("objects", []))]
        if include_receptacles:
            groups.append(("receptacle", metadata.get("receptacles", [])))

        rows = []
        for entity_type, entities in groups:
            for entity in entities:
                entity_id = entity["id"]
                position = initial_vector(data, entity_id, "pos")
                quaternion = initial_vector(data, entity_id, "quat")
                roll, pitch, yaw = quat_xyzw_to_euler_deg(quaternion)
                target_ids = {
                    metadata.get("target_object"),
                    metadata.get("target_receptacle"),
                }
                if "is_target" in entity:
                    role = "target" if entity["is_target"] else "distractor"
                elif metadata.get("task") == "stack":
                    role = "participant"
                else:
                    role = "target" if entity_id in target_ids else "distractor"
                rows.append(
                    {
                        "split": split,
                        "task": entry.get("task", metadata.get("task", "unknown")),
                        "episode": int(entry["episode"]),
                        "path": entry["path"],
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "color": entity.get("color", "unknown"),
                        "shape": entity.get("shape", "unknown"),
                        "role": role,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                        "quat_x": float(quaternion[0]),
                        "quat_y": float(quaternion[1]),
                        "quat_z": float(quaternion[2]),
                        "quat_w": float(quaternion[3]),
                        "roll_deg": float(roll),
                        "pitch_deg": float(pitch),
                        "yaw_deg": float(yaw),
                    }
                )
        return rows


def numeric_stats(rows, field):
    values = np.asarray([row[field] for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def circular_stats(rows, field):
    degrees = np.asarray([row[field] for row in rows], dtype=np.float64)
    radians = np.deg2rad(degrees)
    mean_sin = np.sin(radians).mean()
    mean_cos = np.cos(radians).mean()
    resultant = float(np.hypot(mean_sin, mean_cos))
    circular_std = math.degrees(
        math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-12))))
    )
    return {
        "circular_mean": float(math.degrees(math.atan2(mean_sin, mean_cos))),
        "circular_std": circular_std,
        "min": float(degrees.min()),
        "max": float(degrees.max()),
    }


def summarize(rows_by_split, entries_by_split):
    summary = {"splits": {}}
    for split in SPLITS:
        rows = rows_by_split[split]
        if not rows:
            raise ValueError(f"Split {split} contains no object poses")
        summary["splits"][split] = {
            "episodes": len(entries_by_split[split]),
            "entities": len(rows),
            "colors": dict(sorted(Counter(row["color"] for row in rows).items())),
            "roles": dict(sorted(Counter(row["role"] for row in rows).items())),
            "position": {
                field: numeric_stats(rows, field) for field in POSITION_FIELDS
            },
            "orientation": {
                field: circular_stats(rows, field) for field in ANGLE_FIELDS
            },
        }
    return summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(path, summary):
    overview = []
    position_rows = []
    orientation_rows = []
    for split in SPLITS:
        values = summary["splits"][split]
        overview.append(
            [
                split,
                str(values["episodes"]),
                str(values["entities"]),
                ", ".join(f"{key}={value}" for key, value in values["colors"].items()),
                ", ".join(f"{key}={value}" for key, value in values["roles"].items()),
            ]
        )
        for field in POSITION_FIELDS:
            stats = values["position"][field]
            position_rows.append(
                [split, field]
                + [f"{stats[key]:.5f}" for key in ("mean", "std", "min", "max")]
            )
        for field in ANGLE_FIELDS:
            stats = values["orientation"][field]
            orientation_rows.append(
                [split, field]
                + [
                    f"{stats[key]:.2f}"
                    for key in ("circular_mean", "circular_std", "min", "max")
                ]
            )

    content = [
        "# Initial Object Pose Distribution",
        "",
        "All poses are taken from observation frame 0. Angles are in degrees and use quaternion order `xyzw`.",
        "",
        "## Coverage",
        "",
        markdown_table(
            ["Split", "Episodes", "Objects", "Colors", "Roles"], overview
        ),
        "",
        "## Position",
        "",
        markdown_table(
            ["Split", "Axis", "Mean", "Std", "Min", "Max"], position_rows
        ),
        "",
        "## Orientation",
        "",
        markdown_table(
            ["Split", "Angle", "Circular mean", "Circular std", "Min", "Max"],
            orientation_rows,
        ),
        "",
        "## Figures",
        "",
        "![XY position and yaw](position_xy_and_yaw.png)",
        "",
        "![Marginal distributions](pose_marginals.png)",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def plot_distributions(output, rows_by_split):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for column, split in enumerate(SPLITS):
        rows = rows_by_split[split]
        axis = axes[0, column]
        colors = sorted({row["color"] for row in rows})
        for color in colors:
            selected = [row for row in rows if row["color"] == color]
            axis.scatter(
                [row["x"] for row in selected],
                [row["y"] for row in selected],
                s=20,
                alpha=0.7,
                color=PLOT_COLORS.get(color),
                label=color,
            )
        axis.set_title(f"{split}: initial XY")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

        axis = axes[1, column]
        axis.hist(
            [row["yaw_deg"] for row in rows],
            bins=np.linspace(-180, 180, 37),
            color="#4c78a8",
            alpha=0.85,
        )
        axis.set_title(f"{split}: initial yaw")
        axis.set_xlabel("yaw [deg]")
        axis.set_ylabel("object count")
        axis.set_xlim(-180, 180)
        axis.grid(alpha=0.25)
    fig.savefig(output / "position_xy_and_yaw.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fields = ("x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg")
    labels = ("x [m]", "y [m]", "z [m]", "roll [deg]", "pitch [deg]", "yaw [deg]")
    split_colors = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}
    for axis, field, label in zip(axes.flat, fields, labels):
        all_values = np.asarray(
            [row[field] for rows in rows_by_split.values() for row in rows],
            dtype=np.float64,
        )
        low, high = float(all_values.min()), float(all_values.max())
        if math.isclose(low, high):
            low, high = low - 0.5, high + 0.5
        bins = np.linspace(low, high, 25)
        for split in SPLITS:
            axis.hist(
                [row[field] for row in rows_by_split[split]],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=split_colors[split],
                label=split,
            )
        axis.set_xlabel(label)
        axis.set_ylabel("density")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(output / "pose_marginals.png", dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")
    args.output.mkdir(parents=True, exist_ok=True)

    entries_by_split = {}
    rows_by_split = {}
    all_rows = []
    for split in SPLITS:
        split_path = args.split_dir / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {split_path}")
        entries = read_jsonl(split_path)
        rows = []
        for entry in entries:
            rows.extend(
                episode_rows(
                    args.dataset, split, entry, args.include_receptacles
                )
            )
        entries_by_split[split] = entries
        rows_by_split[split] = rows
        all_rows.extend(rows)

    summary = summarize(rows_by_split, entries_by_split)
    write_csv(args.output / "object_poses.csv", all_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(args.output / "report.md", summary)
    plot_distributions(args.output, rows_by_split)

    print("Initial object pose audit complete")
    for split in SPLITS:
        values = summary["splits"][split]
        print(
            f"  {split:5s}: {values['episodes']} episodes, "
            f"{values['entities']} object poses"
        )
    print(f"Report: {args.output / 'report.md'}")
    print(f"Raw table: {args.output / 'object_poses.csv'}")


if __name__ == "__main__":
    main()
