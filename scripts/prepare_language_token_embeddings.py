"""Build a frozen MiniLM token cache and an order-preservation audit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from huggingface_hub import model_info

from language.frozen_minilm import DEFAULT_MODEL_NAME, FrozenMiniLMTokenEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare token-level language cache")
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
        default=Path(
            "data/splits/pilot_v01_semantic/language_token_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/language_audit/pilot_v01_minilm_tokens"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def read_splits(split_dir, dataset_root):
    instructions = set()
    split_hashes = {}
    stack_orders = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split: {path}")
        split_hashes[split] = sha256(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                instruction = entry["instruction"]
                instructions.add(instruction)
                if entry["task"] == "stack":
                    episode_path = dataset_root / entry["path"]
                    with np.load(episode_path, allow_pickle=False) as archive:
                        metadata = json.loads(
                            str(archive["metadata_json"].item())
                        )
                    order = list(metadata["stack_order"])
                    existing = stack_orders.get(instruction)
                    if existing is not None and existing != order:
                        raise ValueError(
                            "Instruction maps to conflicting Stack orders: "
                            f"{instruction!r}"
                        )
                    stack_orders[instruction] = order
    return sorted(instructions), split_hashes, stack_orders


def pad_sequences(sequences, max_length, trailing_shape, dtype, fill_value=0):
    output = np.full(
        (len(sequences), max_length, *trailing_shape),
        fill_value,
        dtype=dtype,
    )
    for index, sequence in enumerate(sequences):
        values = sequence.cpu().numpy()
        output[index, : len(values)] = values
    return output


def order_audit(instructions, tokens, input_ids, masks, stack_orders):
    stack_records = []
    for index, instruction in enumerate(instructions):
        order = stack_orders.get(instruction)
        if order is None:
            continue
        colors = []
        for color in order:
            positions = [
                token_index
                for token_index, token in enumerate(tokens[index])
                if token.lower().replace("##", "") == color
            ]
            colors.append({"color": color, "token_positions": positions})
        stack_records.append(
            {
                "instruction": instruction,
                "stack_order": order,
                "non_padding_tokens": int(masks[index].sum()),
                "tokens": tokens[index],
                "colors": colors,
                "input_ids": input_ids[index, masks[index]].tolist(),
            }
        )

    reversed_pairs = []
    records_by_instruction = {record["instruction"]: record for record in stack_records}
    for left in stack_records:
        for right in stack_records:
            if left["instruction"] >= right["instruction"]:
                continue
            if left["stack_order"] == list(reversed(right["stack_order"])):
                reversed_pairs.append(
                    {
                        "left": left["instruction"],
                        "right": right["instruction"],
                        "input_ids_equal": left["input_ids"] == right["input_ids"],
                        "left_color_tokens": left["colors"],
                        "right_color_tokens": right["colors"],
                    }
                )
    return {
        "stack_instructions": len(records_by_instruction),
        "reversed_color_sequence_pairs": reversed_pairs[:20],
        "all_reversed_pairs_have_distinct_token_sequences": all(
            not pair["input_ids_equal"] for pair in reversed_pairs
        ),
    }


def markdown(report):
    lines = [
        "# Token-Level MiniLM Audit",
        "",
        f"- Model: `{report['model_name']}`",
        f"- Revision: `{report['revision']}`",
        f"- Instructions: {report['instructions']}",
        f"- Cache shape: `{report['cache_shape']}`",
        f"- Token length: {report['token_length']['min']}–{report['token_length']['max']}",
        f"- Truncated instructions: {report['truncated_instructions']}",
        f"- Reversed pairs retain distinct token sequences: "
        f"**{report['order_audit']['all_reversed_pairs_have_distinct_token_sequences']}**",
        "",
        "## Reversed Stack Examples",
        "",
    ]
    for pair in report["order_audit"]["reversed_color_sequence_pairs"][:10]:
        lines.append(f"- `{pair['left']}`")
        lines.append(f"  vs `{pair['right']}`")
        lines.append(f"  input IDs equal: `{pair['input_ids_equal']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")

    instructions, split_hashes, stack_orders = read_splits(
        args.split_dir, args.dataset
    )
    resolved_revision = model_info(args.model, revision=args.revision).sha
    encoder = FrozenMiniLMTokenEncoder(
        model_name=args.model,
        revision=resolved_revision,
        device=args.device,
    )
    encoded = encoder.encode_tokens(instructions, batch_size=args.batch_size)
    lengths = [len(sequence) for sequence in encoded["input_ids"]]
    max_length = max(lengths)
    embeddings = pad_sequences(
        encoded["token_embeddings"],
        max_length,
        (encoder.embedding_dim,),
        np.float32,
    )
    input_ids = pad_sequences(
        encoded["input_ids"], max_length, (), np.int64
    )
    masks = np.zeros((len(instructions), max_length), dtype=np.bool_)
    for index, length in enumerate(lengths):
        masks[index, :length] = True

    metadata = {
        "version": "1.0",
        "model_name": args.model,
        "revision": resolved_revision,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "embedding_dim": encoder.embedding_dim,
        "max_length": max_length,
        "split_sha256": split_hashes,
        "frozen": True,
        "pooling": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        instructions=np.asarray(instructions),
        token_embeddings=embeddings,
        attention_masks=masks,
        input_ids=input_ids,
        tokens_json=np.asarray(json.dumps(encoded["tokens"])),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    audit = order_audit(
        instructions,
        encoded["tokens"],
        input_ids,
        masks,
        stack_orders,
    )
    report = {
        **metadata,
        "instructions": len(instructions),
        "cache_shape": list(embeddings.shape),
        "token_length": {
            "min": min(lengths),
            "mean": float(np.mean(lengths)),
            "max": max(lengths),
        },
        "truncated_instructions": 0,
        "order_audit": audit,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.report_dir / "audit_report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(f"Cached {len(instructions)} instructions: {args.output}")
    print("Token embedding shape:", embeddings.shape)
    print(f"Audit report: {args.report_dir / 'audit_report.md'}")


if __name__ == "__main__":
    main()
