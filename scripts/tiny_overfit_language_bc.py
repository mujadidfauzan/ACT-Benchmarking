"""Overfit one fixed batch to validate Language-BC optimization mechanics."""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate

from data.manipulation_dataset import ManipulationDataset
from data.normalization import PolicyNormalizer
from language.frozen_minilm import CachedLanguageEmbeddings, CachedTokenEmbeddings
from models.language_bc import LANGUAGE_MODES, LanguageBCPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-overfit Language-BC")
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
        "--normalization",
        type=Path,
        default=Path("data/splits/pilot_v01_semantic/normalization.json"),
    )
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
        "--language-mode",
        choices=(*LANGUAGE_MODES, "both"),
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--required-reduction", type=float, default=0.95)
    parser.add_argument("--max-final-loss", type=float, default=1e-2)
    parser.add_argument(
        "--pretrained-visual",
        action="store_true",
        help="Use ImageNet ResNet18; may download weights on first use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/tiny_overfit/language_bc"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_fixed_batch(args):
    dataset = ManipulationDataset(
        dataset_root=args.dataset,
        split_manifest=args.split,
        mode="bc",
        vertical_flip=True,
        cache_size=1,
    )
    first_episode = dataset.episode_sample_ranges[0]
    if len(first_episode) < args.batch_size:
        raise ValueError(
            f"First episode has only {len(first_episode)} samples, "
            f"fewer than batch size {args.batch_size}"
        )
    indices = np.linspace(
        first_episode.start,
        first_episode.stop - 1,
        num=args.batch_size,
        dtype=int,
    )
    batch = default_collate([dataset[int(index)] for index in indices])
    if len(batch["task"]) != args.batch_size:
        raise ValueError(
            f"First batch contains {len(batch['task'])} samples, "
            f"expected {args.batch_size}"
        )
    return batch


def prepare_common_batch(batch, args, normalizer):
    image = batch["image"]
    if image.shape[-2:] != (args.image_size, args.image_size):
        image = F.interpolate(
            image,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    proprio = normalizer.normalize_proprio(batch["proprio"])
    target = normalizer.normalize_action(batch["action"])
    return {
        "image": image,
        "proprio": proprio,
        "target": target,
        "instruction": list(batch["instruction"]),
        "task": list(batch["task"]),
        "episode_path": list(batch["episode_path"]),
        "timestep": batch["timestep"].tolist(),
    }


def language_inputs(mode, instructions, pooled_cache, token_cache):
    if mode == "pooled":
        return {"language_embedding": pooled_cache.encode(instructions)}
    cached = token_cache.lookup(instructions)
    return {
        "token_embeddings": cached["token_embeddings"],
        "attention_mask": cached["attention_mask"],
    }


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def gradient_norm(parameters):
    squared = torch.zeros((), dtype=torch.float32)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum().cpu()
    return float(torch.sqrt(squared))


def checkpoint_round_trip(model, model_inputs, expected):
    state = copy.deepcopy(model.state_dict())
    restored = LanguageBCPolicy(
        language_mode=model.language_mode,
        pretrained_visual=False,
        freeze_visual_backbone=True,
        dropout=0.0,
    )
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        actual = restored(**model_inputs)["action"]
    if not torch.equal(expected, actual):
        error = torch.max(torch.abs(expected - actual)).item()
        raise AssertionError(f"Checkpoint round trip changed output: {error}")


def overfit_mode(mode, common, args, pooled_cache, token_cache):
    set_seed(args.seed)
    model = LanguageBCPolicy(
        language_mode=mode,
        pretrained_visual=args.pretrained_visual,
        freeze_visual_backbone=True,
        dropout=0.0,
    )
    model.train()
    language = language_inputs(
        mode, common["instruction"], pooled_cache, token_cache
    )
    model_inputs = {
        "image": common["image"],
        "proprio": common["proprio"],
        **language,
    }
    target = common["target"]
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    losses = []
    maximum_gradient_norm = 0.0

    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**model_inputs)["action"]
        loss = nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        losses.append(float(loss.detach()))
        if step == args.steps:
            break
        loss.backward()
        current_gradient_norm = gradient_norm(parameters)
        if not np.isfinite(current_gradient_norm):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        maximum_gradient_norm = max(maximum_gradient_norm, current_gradient_norm)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            print(
                f"[{mode}] step {step + 1:4d}/{args.steps}: "
                f"loss={losses[-1]:.8f}, grad_norm={current_gradient_norm:.4f}"
            )

    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = 1.0 - final_loss / max(initial_loss, 1e-12)
    passed = (
        reduction >= args.required_reduction
        and final_loss <= args.max_final_loss
    )

    model.eval()
    with torch.no_grad():
        expected = model(**model_inputs)["action"]
    checkpoint_round_trip(model, model_inputs, expected)

    result = {
        "language_mode": mode,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "learning_rate": args.learning_rate,
        "pretrained_visual": args.pretrained_visual,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": reduction,
        "maximum_gradient_norm": maximum_gradient_norm,
        "required_reduction": args.required_reduction,
        "max_final_loss": args.max_final_loss,
        "passed": passed,
        "losses": losses,
    }
    print(
        f"[{mode}] initial={initial_loss:.8f}, final={final_loss:.8f}, "
        f"reduction={reduction:.2%}, passed={passed}"
    )
    return result


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("--batch-size and --steps must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.image_size < 32:
        raise ValueError("--image-size must be at least 32")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if not 0.0 <= args.required_reduction <= 1.0:
        raise ValueError("--required-reduction must be in [0, 1]")
    if args.max_final_loss < 0:
        raise ValueError("--max-final-loss cannot be negative")

    set_seed(args.seed)
    normalizer = PolicyNormalizer.from_json(args.normalization)
    normalizer.validate_split(args.split)
    pooled_cache = CachedLanguageEmbeddings(args.pooled_cache)
    token_cache = CachedTokenEmbeddings(args.token_cache)
    raw_batch = load_fixed_batch(args)
    common = prepare_common_batch(raw_batch, args, normalizer)
    modes = LANGUAGE_MODES if args.language_mode == "both" else (args.language_mode,)

    print("Fixed batch")
    print("-----------")
    print("Tasks:", common["task"])
    print("Episodes:", common["episode_path"])
    print("Timesteps:", common["timestep"])
    print("Instruction:", common["instruction"][0])
    print("Image shape:", tuple(common["image"].shape))

    results = [
        overfit_mode(mode, common, args, pooled_cache, token_cache)
        for mode in modes
    ]
    report = {
        "seed": args.seed,
        "dataset": str(args.dataset),
        "split": str(args.split),
        "fixed_batch": {
            "tasks": common["task"],
            "episode_paths": common["episode_path"],
            "timesteps": common["timestep"],
            "instruction": common["instruction"],
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Report:", report_path)

    failures = [result["language_mode"] for result in results if not result["passed"]]
    if failures:
        raise SystemExit(
            "Tiny overfit failed for: " + ", ".join(failures)
        )
    print("Language-BC mechanical tiny overfit: PASS")


if __name__ == "__main__":
    main()
