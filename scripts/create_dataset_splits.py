"""Create deterministic episode splits from demonstration manifests."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TASKS = ("pick", "place", "stack")
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset split manifests")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/demos/pilot_v01"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic"),
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=TASKS, default=list(TASKS)
    )
    parser.add_argument(
        "--protocol",
        choices=("semantic", "iid", "language"),
        default="semantic",
        help=(
            "semantic holds out compositional_test; language holds out "
            "held_out instructions; iid randomly splits every episode"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace split files in an existing output directory",
    )
    return parser.parse_args()


def read_manifest(path, dataset_root, task):
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            relative_path = Path(task) / entry["path"]
            episode_path = dataset_root / relative_path
            if not episode_path.is_file():
                raise FileNotFoundError(
                    f"Manifest line {line_number} points to missing file: "
                    f"{episode_path}"
                )
            metadata = entry.get("metadata", {})
            entries.append(
                {
                    "task": task,
                    "episode": entry["episode"],
                    "path": relative_path.as_posix(),
                    "steps": entry["steps"],
                    "instruction": entry["instruction"],
                    "instruction_template_id": metadata.get(
                        "instruction_template_id"
                    ),
                    "instruction_split": metadata.get("instruction_split"),
                    "semantic_split": metadata.get("semantic_split"),
                }
            )
    return entries


def partition_iid(entries, rng, ratios):
    indices = rng.permutation(len(entries))
    train_end = int(np.floor(len(entries) * ratios[0]))
    val_end = train_end + int(np.floor(len(entries) * ratios[1]))
    groups = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    return {
        split: [entries[int(index)] for index in split_indices]
        for split, split_indices in groups.items()
    }


def partition_with_held_out(entries, rng, held_out_key, held_out_value, ratios):
    held_out = [entry for entry in entries if entry.get(held_out_key) == held_out_value]
    eligible = [entry for entry in entries if entry.get(held_out_key) != held_out_value]

    if not held_out:
        return partition_iid(entries, rng, ratios), True

    indices = rng.permutation(len(eligible))
    train_val_total = ratios[0] + ratios[1]
    train_fraction = ratios[0] / train_val_total
    train_count = int(np.floor(len(eligible) * train_fraction))
    result = {
        "train": [eligible[int(index)] for index in indices[:train_count]],
        "val": [eligible[int(index)] for index in indices[train_count:]],
        "test": held_out,
    }
    return result, False


def assign_task(entries, protocol, rng, ratios):
    if protocol == "iid":
        return partition_iid(entries, rng, ratios), False
    if protocol == "semantic":
        return partition_with_held_out(
            entries, rng, "semantic_split", "compositional_test", ratios
        )
    return partition_with_held_out(
        entries, rng, "instruction_split", "held_out", ratios
    )


def semantic_label(entry):
    return entry.get("semantic_split") or "missing"


def summarize(splits, fallback_tasks):
    summary = {
        "total_episodes": sum(len(entries) for entries in splits.values()),
        "fallback_iid_tasks": sorted(fallback_tasks),
        "splits": {},
    }
    for split, entries in splits.items():
        summary["splits"][split] = {
            "episodes": len(entries),
            "steps": sum(entry["steps"] for entry in entries),
            "tasks": dict(sorted(Counter(entry["task"] for entry in entries).items())),
            "semantic_splits": dict(
                sorted(Counter(semantic_label(entry) for entry in entries).items())
            ),
            "instruction_splits": dict(
                sorted(
                    Counter(
                        entry.get("instruction_split") or "missing"
                        for entry in entries
                    ).items()
                )
            ),
        }
    return summary


def write_jsonl(path, entries):
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def main():
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios cannot be negative")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1")
    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("All split ratios must be positive")
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset}")

    output_files = [args.output / f"{name}.jsonl" for name in SPLIT_NAMES]
    output_files.append(args.output / "summary.json")
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Split output already exists: {names}; use --overwrite")

    rng = np.random.default_rng(args.seed)
    splits = defaultdict(list)
    fallback_tasks = []
    seen_paths = set()
    for task in args.tasks:
        entries = read_manifest(
            args.dataset / task / "manifest.jsonl", args.dataset, task
        )
        task_splits, used_fallback = assign_task(entries, args.protocol, rng, ratios)
        if used_fallback:
            fallback_tasks.append(task)
        for split, split_entries in task_splits.items():
            for entry in split_entries:
                if entry["path"] in seen_paths:
                    raise ValueError(f"Episode assigned more than once: {entry['path']}")
                seen_paths.add(entry["path"])
            splits[split].extend(split_entries)

    args.output.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        entries = sorted(splits[split], key=lambda item: (item["task"], item["episode"]))
        write_jsonl(args.output / f"{split}.jsonl", entries)

    summary = {
        "version": "1.0",
        "dataset_root": str(args.dataset.resolve()),
        "protocol": args.protocol,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        **summarize(splits, fallback_tasks),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Created {args.protocol} splits in {args.output}")
    for split in SPLIT_NAMES:
        task_counts = summary["splits"][split]["tasks"]
        print(f"  {split:5s}: {summary['splits'][split]['episodes']:3d} {task_counts}")
    if fallback_tasks:
        print(
            "  IID fallback (no held-out episodes): "
            + ", ".join(sorted(fallback_tasks))
        )


if __name__ == "__main__":
    main()
