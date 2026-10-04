"""Structural forward, backward, freezing, and checkpoint tests for Language-BC."""

import argparse
import io
from pathlib import Path

import torch
from torch import nn

from data.normalization import PolicyNormalizer
from language.frozen_minilm import CachedLanguageEmbeddings, CachedTokenEmbeddings
from models.language_bc import LANGUAGE_MODES, LanguageBCPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test Language-BC policies")
    parser.add_argument(
        "--pooled-cache",
        type=Path,
        default=Path(
            "data/splits/pilot_v01_semantic/language_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=Path(
            "data/splits/pilot_v01_semantic/language_token_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/normalization.json"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained-visual",
        action="store_true",
        help="Load ImageNet weights; may download them on first use",
    )
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def batch_norm_state(module):
    return {
        name: (
            child.running_mean.detach().clone(),
            child.running_var.detach().clone(),
            child.num_batches_tracked.detach().clone(),
        )
        for name, child in module.named_modules()
        if isinstance(child, nn.BatchNorm2d)
    }


def require_batch_norm_unchanged(before, after):
    require(before.keys() == after.keys(), "BatchNorm module set changed")
    for name in before:
        for previous, current in zip(before[name], after[name]):
            require(
                torch.equal(previous, current),
                f"Frozen BatchNorm state changed: {name}",
            )


def trainable_parameter_names(model):
    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def require_gradient_boundaries(model):
    trainable = trainable_parameter_names(model)
    require(trainable, "Model has no trainable parameters")
    for name, parameter in model.named_parameters():
        if name.startswith("visual_encoder.backbone"):
            require(not parameter.requires_grad, f"Frozen backbone is trainable: {name}")
            require(parameter.grad is None, f"Frozen backbone received gradient: {name}")
        elif parameter.requires_grad:
            require(parameter.grad is not None, f"Trainable parameter has no gradient: {name}")
            require(torch.isfinite(parameter.grad).all().item(), f"Non-finite gradient: {name}")


def make_model(mode, args):
    return LanguageBCPolicy(
        language_mode=mode,
        pretrained_visual=args.pretrained_visual,
        freeze_visual_backbone=True,
        dropout=0.0,
    )


def language_inputs(mode, instructions, pooled_cache, token_cache):
    if mode == "pooled":
        return {"language_embedding": pooled_cache.encode(instructions)}
    token_batch = token_cache.lookup(instructions)
    return {
        "token_embeddings": token_batch["token_embeddings"],
        "attention_mask": token_batch["attention_mask"],
    }


def require_output_contract(output, mode, batch_size, token_length=None):
    expected = {
        "action": (batch_size, 7),
        "visual_feature": (batch_size, 256),
        "proprio_feature": (batch_size, 64),
        "language_feature": (batch_size, 128),
        "fusion_feature": (batch_size, 128),
    }
    for key, shape in expected.items():
        require(tuple(output[key].shape) == shape, f"Unexpected {key}: {output[key].shape}")
        require(torch.isfinite(output[key]).all().item(), f"Non-finite output: {key}")
    if mode == "pooled":
        require(output["attention_weights"] is None, "Pooled mode returned attention")
    else:
        weights = output["attention_weights"]
        require(
            tuple(weights.shape) == (batch_size, token_length),
            f"Unexpected attention shape: {weights.shape}",
        )
        require(
            torch.allclose(weights.sum(dim=1), torch.ones(batch_size), atol=1e-6),
            "Attention weights do not sum to one",
        )


def checkpoint_round_trip(model, model_inputs, expected):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = LanguageBCPolicy(
        language_mode=model.language_mode,
        pretrained_visual=False,
        freeze_visual_backbone=True,
        dropout=0.0,
    )
    restored.load_state_dict(torch.load(buffer, map_location="cpu"))
    restored.eval()
    with torch.no_grad():
        actual = restored(**model_inputs)["action"]
    require(torch.equal(expected, actual), "Checkpoint round trip changed action output")


def test_mode(mode, args, pooled_cache, token_cache, normalizer):
    torch.manual_seed(args.seed)
    model = make_model(mode, args)
    model.train()
    require(not model.visual_encoder.backbone.training, "Frozen backbone entered train mode")
    require(model.visual_encoder.projection.training, "Visual projection is not trainable")

    instructions = pooled_cache.instructions[: args.batch_size]
    image = torch.rand(args.batch_size, 3, args.image_size, args.image_size)
    raw_proprio = normalizer.proprio_mean.unsqueeze(0).repeat(args.batch_size, 1)
    raw_proprio = raw_proprio + 0.1 * normalizer.proprio_std.unsqueeze(0)
    proprio = normalizer.normalize_proprio(raw_proprio)
    raw_action = normalizer.action_mean.unsqueeze(0).repeat(args.batch_size, 1)
    raw_action = raw_action + 0.2 * normalizer.action_std.unsqueeze(0)
    target = normalizer.normalize_action(raw_action)
    lang = language_inputs(mode, instructions, pooled_cache, token_cache)
    model_inputs = {"image": image, "proprio": proprio, **lang}

    batch_norm_before = batch_norm_state(model.visual_encoder.backbone)
    output = model(**model_inputs)
    token_length = lang.get("attention_mask", torch.empty(0, 0)).shape[-1]
    require_output_contract(output, mode, args.batch_size, token_length)
    if mode == "token_attention":
        padding_weights = output["attention_weights"].masked_select(
            ~lang["attention_mask"]
        )
        require(torch.count_nonzero(padding_weights).item() == 0, "Padding received attention")

    loss = nn.functional.mse_loss(output["action"], target)
    require(torch.isfinite(loss).item(), "Loss is not finite")
    loss.backward()
    require_gradient_boundaries(model)
    batch_norm_after = batch_norm_state(model.visual_encoder.backbone)
    require_batch_norm_unchanged(batch_norm_before, batch_norm_after)

    model.eval()
    with torch.no_grad():
        expected = model(**model_inputs)["action"]
    checkpoint_round_trip(model, model_inputs, expected)

    model.visual_encoder.unfreeze_layer4()
    model.train()
    require(model.visual_encoder.backbone.layer4.training, "Layer4 did not enter train mode")
    require(
        all(
            parameter.requires_grad
            for parameter in model.visual_encoder.backbone.layer4.parameters()
        ),
        "Layer4 parameters were not unfrozen",
    )
    require(
        all(
            not parameter.requires_grad
            for name, parameter in model.visual_encoder.backbone.named_parameters()
            if not name.startswith("layer4")
        ),
        "A backbone stage other than layer4 was unfrozen",
    )
    print(f"{mode}: PASS (loss={loss.item():.6f})")


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")

    pooled_cache = CachedLanguageEmbeddings(args.pooled_cache)
    token_cache = CachedTokenEmbeddings(args.token_cache)
    normalizer = PolicyNormalizer.from_json(args.normalization)
    common = set(pooled_cache.instructions) & set(token_cache.instructions)
    require(len(common) >= args.batch_size, "Language caches do not overlap enough")

    for mode in LANGUAGE_MODES:
        test_mode(mode, args, pooled_cache, token_cache, normalizer)
    print("Language-BC model smoke test: PASS")


if __name__ == "__main__":
    main()
