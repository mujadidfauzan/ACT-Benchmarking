"""Smoke-test frozen MiniLM encoding and cached instruction lookup."""

import argparse
import json
from pathlib import Path

import torch

from language.frozen_minilm import CachedLanguageEmbeddings, FrozenMiniLMEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test MiniLM language cache")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/splits/pilot_v01_semantic/language_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic"),
    )
    parser.add_argument(
        "--verify-live-encoder",
        action="store_true",
        help="Also load MiniLM and compare live encoding against the cache",
    )
    return parser.parse_args()


def split_instructions(split_dir):
    instructions = set()
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    instructions.add(json.loads(line)["instruction"])
    return instructions


def main():
    args = parse_args()
    cache = CachedLanguageEmbeddings(args.cache)
    instructions = split_instructions(args.split_dir)
    missing = sorted(instructions - set(cache.instructions))
    extra = sorted(set(cache.instructions) - instructions)
    if missing or extra:
        raise AssertionError(
            f"Cache coverage mismatch; missing={len(missing)}, extra={len(extra)}"
        )
    if cache.embedding_dim != 384:
        raise AssertionError(f"Expected 384D MiniLM embeddings, got {cache.embedding_dim}")

    first = cache.instructions[0]
    single = cache.encode(first)
    batch = cache.encode(cache.instructions[:4])
    if single.shape != (384,) or batch.shape != (4, 384):
        raise AssertionError("Unexpected cached embedding shape")
    if not torch.equal(single, batch[0]):
        raise AssertionError("Single and batched cache lookup differ")
    if not torch.allclose(torch.linalg.vector_norm(batch, dim=-1), torch.ones(4), atol=1e-4):
        raise AssertionError("Cached embeddings are not unit-normalized")

    if args.verify_live_encoder:
        encoder = FrozenMiniLMEncoder(
            model_name=cache.metadata["model_name"],
            revision=cache.metadata["revision"],
            device="cpu",
        )
        if encoder.training:
            raise AssertionError("Frozen encoder must stay in evaluation mode")
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise AssertionError("Frozen encoder contains trainable parameters")
        live = encoder.encode(cache.instructions[:4]).cpu()
        if not torch.allclose(live, batch, atol=1e-5, rtol=1e-5):
            error = torch.max(torch.abs(live - batch)).item()
            raise AssertionError(f"Live and cached embeddings differ: {error}")
        print("Live encoder comparison: PASS")

    print("Cache:", args.cache)
    print("Model:", cache.metadata["model_name"])
    print("Revision:", cache.metadata["revision"])
    print("Unique instructions:", len(cache))
    print("Embedding dimension:", cache.embedding_dim)
    print("Train/val/test coverage: PASS")
    print("Frozen language cache smoke test: PASS")


if __name__ == "__main__":
    main()
