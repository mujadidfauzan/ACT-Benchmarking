"""Smoke-test cached token embeddings and trainable attention pooling."""

import argparse
import io
from pathlib import Path

import torch

from language.frozen_minilm import CachedTokenEmbeddings, TrainableAttentionPooling


def parse_args():
    parser = argparse.ArgumentParser(description="Test token language baseline")
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/splits/pilot_v01_semantic/language_token_embeddings.npz"
        ),
    )
    parser.add_argument("--output-dim", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    cache = CachedTokenEmbeddings(args.cache)
    instructions = cache.instructions[:4]
    batch = cache.lookup(instructions)
    token_embeddings = batch["token_embeddings"].clone().requires_grad_(True)
    mask = batch["attention_mask"]
    pooler = TrainableAttentionPooling(
        input_dim=cache.embedding_dim,
        output_dim=args.output_dim,
        dropout=0.0,
    )
    pooler.train()
    output = pooler(token_embeddings, mask)

    if output["feature"].shape != (4, args.output_dim):
        raise AssertionError("Unexpected pooled feature shape")
    if output["attention_weights"].shape != mask.shape:
        raise AssertionError("Unexpected attention-weight shape")
    if not torch.allclose(
        output["attention_weights"].sum(dim=1), torch.ones(4), atol=1e-6
    ):
        raise AssertionError("Attention weights do not sum to one")
    if torch.count_nonzero(output["attention_weights"].masked_select(~mask)):
        raise AssertionError("Padding positions received attention")

    loss = output["feature"].square().mean()
    loss.backward()
    if not all(parameter.grad is not None for parameter in pooler.parameters()):
        raise AssertionError("Attention pooler did not receive gradients")
    if token_embeddings.grad is None:
        raise AssertionError("Gradient smoke test did not reach token input")
    if cache.token_embeddings.requires_grad:
        raise AssertionError("Cached frozen embeddings must not require gradients")

    pooler.eval()
    with torch.no_grad():
        expected = pooler(batch["token_embeddings"], mask)["feature"]
    buffer = io.BytesIO()
    torch.save(pooler.state_dict(), buffer)
    buffer.seek(0)
    restored = TrainableAttentionPooling(
        input_dim=cache.embedding_dim,
        output_dim=args.output_dim,
        dropout=0.0,
    )
    restored.load_state_dict(torch.load(buffer, map_location="cpu"))
    restored.eval()
    with torch.no_grad():
        actual = restored(batch["token_embeddings"], mask)["feature"]
    if not torch.equal(expected, actual):
        raise AssertionError("Checkpoint round trip changed pooled features")

    print("Cache:", args.cache)
    print("Instructions:", len(cache))
    print("Token cache shape:", tuple(cache.token_embeddings.shape))
    print("Pooled feature shape:", tuple(output["feature"].shape))
    print("Padding attention: PASS")
    print("Trainable pooler gradients: PASS")
    print("Frozen cached embeddings: PASS")
    print("Checkpoint round trip: PASS")
    print("Token language smoke test: PASS")


if __name__ == "__main__":
    main()
