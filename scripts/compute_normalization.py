"""Compute policy normalization statistics from a training split only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data.manipulation_dataset import DEFAULT_POLICY_PROPRIO_KEYS

PROPRIO_KEYS = DEFAULT_POLICY_PROPRIO_KEYS


def parse_args():
    parser = argparse.ArgumentParser(description="Compute train-split statistics")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/demos/pilot_v01"),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/normalization.json"),
    )
    parser.add_argument(
        "--image-convention",
        choices=("opengl", "opencv"),
        default="opengl",
        help="Convention of stored RGB observations; pilot_v01 uses opengl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None
        self.minimum = None
        self.maximum = None

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError(f"Expected non-empty rank-2 values, got {values.shape}")
        batch_count = len(values)
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        batch_min = values.min(axis=0)
        batch_max = values.max(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_min
            self.maximum = batch_max
            return

        if batch_mean.shape != self.mean.shape:
            raise ValueError("Feature dimension changed while computing statistics")
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)
        self.count = total

    def result(self):
        if self.count == 0:
            raise ValueError("No values were accumulated")
        variance = self.m2 / self.count
        std = np.sqrt(np.maximum(variance, 0.0))
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
        }


def equal_task_result(task_stats):
    if not task_stats:
        raise ValueError("No per-task statistics were accumulated")
    stats = list(task_stats.values())
    means = np.stack([stat.mean for stat in stats])
    variances = np.stack([stat.m2 / stat.count for stat in stats])
    mean = means.mean(axis=0)
    variance = np.mean(variances + (means - mean) ** 2, axis=0)
    return {
        "weighting": "equal_task",
        "tasks": sorted(task_stats),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(variance, 0.0)).tolist(),
        "min": np.min(np.stack([stat.minimum for stat in stats]), axis=0).tolist(),
        "max": np.max(np.stack([stat.maximum for stat in stats]), axis=0).tolist(),
    }


def read_entries(path):
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("task", "path", "steps"):
                if key not in entry:
                    raise ValueError(f"{path}:{line_number} is missing {key!r}")
            if entry.get("semantic_split") == "compositional_test":
                raise ValueError(
                    f"Train split contains compositional-test episode: {entry['path']}"
                )
            entries.append(entry)
    if not entries:
        raise ValueError(f"Split contains no episodes: {path}")
    return entries


def split_digest(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")
    if not args.split.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {args.split}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")

    entries = read_entries(args.split)
    proprio_stats = RunningStats()
    action_stats = RunningStats()
    task_proprio_stats = {}
    task_action_stats = {}
    task_episodes = {}
    task_steps = {}

    for index, entry in enumerate(entries, start=1):
        path = args.dataset / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Episode does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = ("actions", *PROPRIO_KEYS)
            missing = [key for key in required if key not in archive]
            if missing:
                raise KeyError(f"{path} is missing keys: {missing}")
            actions = np.asarray(archive["actions"], dtype=np.float64)
            steps = len(actions)
            if steps != int(entry["steps"]):
                raise ValueError(f"Step count mismatch: {path}")
            proprio_parts = []
            for key in PROPRIO_KEYS:
                values = np.asarray(archive[key], dtype=np.float64)
                if len(values) != steps + 1:
                    raise ValueError(f"Observation alignment mismatch: {path}, {key}")
                proprio_parts.append(values[:steps].reshape(steps, -1))
            proprio = np.concatenate(proprio_parts, axis=1)

        if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
            raise ValueError(f"NaN or Inf found in {path}")
        action_stats.update(actions)
        proprio_stats.update(proprio)
        task = entry["task"]
        task_proprio_stats.setdefault(task, RunningStats()).update(proprio)
        task_action_stats.setdefault(task, RunningStats()).update(actions)
        task_episodes[task] = task_episodes.get(task, 0) + 1
        task_steps[task] = task_steps.get(task, 0) + steps
        print(f"[{index:3d}/{len(entries)}] {entry['path']} ({steps} steps)")

    output = {
        "version": "2.0",
        "dataset_root": str(args.dataset.resolve()),
        "split_manifest": str(args.split.resolve()),
        "split_sha256": split_digest(args.split),
        "episodes": len(entries),
        "task_episodes": dict(sorted(task_episodes.items())),
        "task_steps": dict(sorted(task_steps.items())),
        "image": {
            "stored_convention": args.image_convention,
            "vertical_flip_required": args.image_convention == "opengl",
            "scale": 255.0,
            "output_range": [0.0, 1.0],
        },
        "proprio": {
            "keys": list(PROPRIO_KEYS),
            **proprio_stats.result(),
        },
        "action": action_stats.result(),
        "task_balanced_proprio": {
            "keys": list(PROPRIO_KEYS),
            **equal_task_result(task_proprio_stats),
        },
        "task_balanced_action": equal_task_result(task_action_stats),
        "per_task": {
            task: {
                "proprio": task_proprio_stats[task].result(),
                "action": task_action_stats[task].result(),
            }
            for task in sorted(task_proprio_stats)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved normalization statistics: {args.output}")
    print("Proprio samples:", output["proprio"]["count"])
    print("Action samples:", output["action"]["count"])


if __name__ == "__main__":
    main()
