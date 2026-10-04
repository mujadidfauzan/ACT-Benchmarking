"""Cache frozen MiniLM embeddings and audit semantic separability."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from huggingface_hub import model_info

from language.frozen_minilm import DEFAULT_MODEL_NAME, FrozenMiniLMEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and audit language embeddings")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/demos/pilot_v01"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/language_embeddings.npz"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/language_audit/pilot_v01_minilm"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_split(path, split_name):
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            for key in ("task", "path", "instruction"):
                if key not in entry:
                    raise ValueError(f"{path}:{line_number} is missing {key!r}")
            entries.append({**entry, "dataset_split": split_name})
    return entries


def episode_metadata(dataset_root, relative_path):
    path = dataset_root / relative_path
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata_json"].item()))


def entity_color(entities, entity_id):
    matches = [entity["color"] for entity in entities if entity["id"] == entity_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one entity with id {entity_id!r}")
    return matches[0]


def semantic_key(task, metadata):
    if task == "pick":
        return entity_color(metadata["objects"], metadata["target_object"])
    if task == "place":
        source = entity_color(metadata["objects"], metadata["target_object"])
        target = entity_color(
            metadata["receptacles"], metadata["target_receptacle"]
        )
        return f"{source}->{target}"
    if task == "stack":
        return ">".join(metadata["stack_order"])
    raise ValueError(f"Unknown task: {task}")


def build_instruction_records(entries, dataset_root):
    records = {}
    for entry in entries:
        metadata = episode_metadata(dataset_root, entry["path"])
        record = {
            "instruction": entry["instruction"],
            "task": entry["task"],
            "semantic_key": semantic_key(entry["task"], metadata),
            "template_id": metadata["instruction_template_id"],
        }
        existing = records.get(entry["instruction"])
        if existing is not None and (
            existing["task"] != record["task"]
            or existing["semantic_key"] != record["semantic_key"]
        ):
            raise ValueError(
                "Identical instruction maps to conflicting semantics: "
                f"{entry['instruction']!r}"
            )
        records[entry["instruction"]] = record
    return records


def closest_pairs(records, similarities, limit=10):
    pairs = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            pairs.append(
                {
                    "cosine": float(similarities[left, right]),
                    "left": records[left]["instruction"],
                    "right": records[right]["instruction"],
                    "left_task": records[left]["task"],
                    "right_task": records[right]["task"],
                    "left_semantic": records[left]["semantic_key"],
                    "right_semantic": records[right]["semantic_key"],
                }
            )
    return sorted(pairs, key=lambda item: item["cosine"], reverse=True)[:limit]


def semantic_centroid_audit(records, embeddings):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["task"], record["semantic_key"])].append(index)
    centroids = {}
    for key, indices in groups.items():
        centroid = embeddings[indices].mean(axis=0)
        centroids[key] = centroid / np.linalg.norm(centroid)

    per_task = {}
    for task in sorted({key[0] for key in centroids}):
        keys = sorted(key for key in centroids if key[0] == task)
        similarities = []
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                similarities.append(
                    {
                        "left": keys[left][1],
                        "right": keys[right][1],
                        "cosine": float(
                            np.dot(centroids[keys[left]], centroids[keys[right]])
                        ),
                    }
                )
        similarities.sort(key=lambda item: item["cosine"], reverse=True)
        per_task[task] = {
            "semantic_classes": len(keys),
            "most_similar_classes": similarities[:10],
        }
    return per_task, centroids


def reversed_stack_audit(centroids):
    stack = {
        key[1]: value for key, value in centroids.items() if key[0] == "stack"
    }
    results = []
    visited = set()
    for order, embedding in sorted(stack.items()):
        reverse = ">".join(reversed(order.split(">")))
        pair = tuple(sorted((order, reverse)))
        if reverse in stack and pair not in visited and order != reverse:
            visited.add(pair)
            results.append(
                {
                    "order": order,
                    "reverse": reverse,
                    "cosine": float(np.dot(embedding, stack[reverse])),
                }
            )
    return sorted(results, key=lambda item: item["cosine"], reverse=True)


def markdown_report(report):
    lines = [
        "# Language Embedding Audit",
        "",
        f"- Model: `{report['model_name']}`",
        f"- Revision: `{report['revision']}`",
        f"- Unique instructions: {report['unique_instructions']}",
        f"- Embedding dimension: {report['embedding_dim']}",
        f"- Token range: {report['tokens']['min']}–{report['tokens']['max']}",
        f"- Embedding norm range: {report['norms']['min']:.6f}–{report['norms']['max']:.6f}",
        f"- Audit status: **{report['audit_status'].upper()}**",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in report["findings"])
    lines += [
        "",
        "## Closest Distinct Instructions",
        "",
        "| Cosine | Left | Right |",
        "| ---: | --- | --- |",
    ]
    for pair in report["closest_distinct_instructions"]:
        lines.append(
            f"| {pair['cosine']:.4f} | {pair['left']} | {pair['right']} |"
        )
    lines += ["", "## Semantic Classes", ""]
    for task, values in report["semantic_audit"].items():
        lines.append(f"### {task.title()}")
        lines.append("")
        lines.append(f"Semantic classes: {values['semantic_classes']}")
        lines.append("")
        for pair in values["most_similar_classes"][:5]:
            lines.append(
                f"- `{pair['left']}` vs `{pair['right']}`: {pair['cosine']:.4f}"
            )
        lines.append("")
    lines += ["## Reversed Stack Orders", ""]
    for pair in report["reversed_stack_orders"]:
        lines.append(
            f"- `{pair['order']}` vs `{pair['reverse']}`: {pair['cosine']:.4f}"
        )
    if not report["reversed_stack_orders"]:
        lines.append("No reversed order pairs were both present.")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")

    split_paths = {
        name: args.split_dir / f"{name}.jsonl"
        for name in ("train", "val", "test")
    }
    entries = []
    for name, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing split: {path}")
        entries.extend(read_split(path, name))
    records_by_instruction = build_instruction_records(entries, args.dataset)
    instructions = sorted(records_by_instruction)
    records = [records_by_instruction[instruction] for instruction in instructions]

    resolved_revision = model_info(args.model, revision=args.revision).sha
    encoder = FrozenMiniLMEncoder(
        model_name=args.model,
        revision=resolved_revision,
        device=args.device,
    )
    tokenized = encoder.tokenize(instructions)
    token_counts = tokenized["attention_mask"].sum(dim=1).cpu().numpy()
    embeddings_tensor = encoder.encode(instructions, batch_size=args.batch_size)
    embeddings = embeddings_tensor.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    similarities = embeddings @ embeddings.T

    split_hashes = {
        name: file_sha256(path) for name, path in split_paths.items()
    }
    metadata = {
        "version": "1.0",
        "model_name": args.model,
        "revision": resolved_revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "normalized": True,
        "embedding_dim": encoder.embedding_dim,
        "split_sha256": split_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        instructions=np.asarray(instructions),
        embeddings=embeddings,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    semantic_audit, centroids = semantic_centroid_audit(records, embeddings)
    closest = closest_pairs(records, similarities, limit=10)
    reversed_orders = reversed_stack_audit(centroids)
    findings = []
    if closest and closest[0]["cosine"] >= 0.995:
        findings.append(
            "Some distinct instructions have cosine similarity >= 0.995; "
            "pooled embeddings weakly separate their semantics."
        )
    if reversed_orders and reversed_orders[0]["cosine"] >= 0.95:
        findings.append(
            "Some reversed Stack orders have cosine similarity >= 0.95; "
            "order information is weak in pooled MiniLM embeddings."
        )
    has_warning = bool(findings)
    if not has_warning:
        findings.append("No configured embedding-separation warning was triggered.")
    report = {
        **metadata,
        "episodes": len(entries),
        "unique_instructions": len(instructions),
        "tokens": {
            "min": int(token_counts.min()),
            "mean": float(token_counts.mean()),
            "max": int(token_counts.max()),
        },
        "norms": {"min": float(norms.min()), "max": float(norms.max())},
        "audit_status": "warning" if has_warning else "pass",
        "findings": findings,
        "closest_distinct_instructions": closest,
        "semantic_audit": semantic_audit,
        "reversed_stack_orders": reversed_orders,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.report_dir / "audit_report.md").write_text(
        markdown_report(report), encoding="utf-8"
    )

    print(f"Cached {len(instructions)} unique instructions: {args.output}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Token range: {token_counts.min()}-{token_counts.max()}")
    print(f"Audit report: {args.report_dir / 'audit_report.md'}")


if __name__ == "__main__":
    main()
