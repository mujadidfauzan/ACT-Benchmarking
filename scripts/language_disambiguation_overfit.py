"""Controlled overfit where identical robot inputs require language-specific actions."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.normalization import PolicyNormalizer
from language.frozen_minilm import CachedLanguageEmbeddings, CachedTokenEmbeddings
from models.language_bc import LANGUAGE_MODES, LanguageBCPolicy


PROPRIO_KEYS = (
    "obs__robot0_joint_pos",
    "obs__robot0_joint_vel",
    "obs__robot0_gripper_qpos",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit reversed Stack instructions with identical observations"
    )
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
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
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
        default=Path("results/language_disambiguation/language_bc"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata_json"].item()))


def stack_records(dataset_root, split_path):
    records = []
    for entry in load_jsonl(split_path):
        if entry["task"] != "stack":
            continue
        episode_path = dataset_root / entry["path"]
        metadata = load_metadata(episode_path)
        records.append(
            {
                "instruction": entry["instruction"],
                "path": entry["path"],
                "episode_path": episode_path,
                "stack_order": tuple(metadata["stack_order"]),
                "template_id": metadata["instruction_template_id"],
            }
        )
    if not records:
        raise ValueError("Train split contains no Stack episodes")
    return records


def select_reversed_pair(records, pooled_cache):
    candidates = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left["stack_order"] != tuple(reversed(right["stack_order"])):
                continue
            if left["template_id"] != right["template_id"]:
                continue
            embeddings = pooled_cache.encode(
                [left["instruction"], right["instruction"]]
            )
            cosine = float(torch.dot(embeddings[0], embeddings[1]))
            candidates.append((cosine, left, right))
    if not candidates:
        raise ValueError(
            "No reversed Stack pair with the same template exists in train split"
        )
    return max(candidates, key=lambda item: item[0])


def load_episode_policy_data(path, normalizer):
    with np.load(path, allow_pickle=False) as archive:
        actions = torch.from_numpy(
            np.asarray(archive["actions"], dtype=np.float32)
        )
        proprio = torch.from_numpy(
            np.concatenate(
                [
                    np.asarray(archive[key][:-1], dtype=np.float32).reshape(
                        len(actions), -1
                    )
                    for key in PROPRIO_KEYS
                ],
                axis=1,
            )
        )
        images = np.asarray(archive["obs__agentview_image"][:-1])
    return {
        "actions": normalizer.normalize_action(actions),
        "proprio": normalizer.normalize_proprio(proprio),
        "images": images,
    }


def most_distinct_action_pair(left_actions, right_actions):
    distances = torch.cdist(left_actions, right_actions)
    flat_index = int(torch.argmax(distances))
    right_count = right_actions.shape[0]
    left_index = flat_index // right_count
    right_index = flat_index % right_count
    return left_index, right_index, float(distances[left_index, right_index])


def controlled_batch(pair, normalizer, image_size):
    _, left, right = pair
    left_data = load_episode_policy_data(left["episode_path"], normalizer)
    right_data = load_episode_policy_data(right["episode_path"], normalizer)
    left_step, right_step, target_distance = most_distinct_action_pair(
        left_data["actions"], right_data["actions"]
    )

    raw_image = np.flip(left_data["images"][left_step], axis=0).copy()
    image = torch.from_numpy(raw_image).permute(2, 0, 1).float().div(255.0)
    image = image.unsqueeze(0)
    if image.shape[-2:] != (image_size, image_size):
        image = F.interpolate(
            image,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    base_proprio = left_data["proprio"][left_step].unsqueeze(0)
    return {
        "image": image.repeat(2, 1, 1, 1),
        "proprio": base_proprio.repeat(2, 1),
        "target": torch.stack(
            [
                left_data["actions"][left_step],
                right_data["actions"][right_step],
            ]
        ),
        "instructions": [left["instruction"], right["instruction"]],
        "paths": [left["path"], right["path"]],
        "steps": [left_step, right_step],
        "orders": [list(left["stack_order"]), list(right["stack_order"])],
        "template_id": left["template_id"],
        "target_distance": target_distance,
    }


def language_inputs(mode, instructions, pooled_cache, token_cache):
    if mode == "pooled":
        return {"language_embedding": pooled_cache.encode(instructions)}
    cached = token_cache.lookup(instructions)
    return {
        "token_embeddings": cached["token_embeddings"],
        "attention_mask": cached["attention_mask"],
    }


def cosine_pair(features):
    return float(
        F.cosine_similarity(features[0].unsqueeze(0), features[1].unsqueeze(0))
    )


def top_attention(cache, instructions, weights, limit=6):
    rows = []
    for row, instruction in enumerate(instructions):
        cache_index = cache._indices[instruction]
        tokens = cache.tokens[cache_index]
        valid_weights = weights[row, : len(tokens)]
        order = torch.argsort(valid_weights, descending=True)[:limit].tolist()
        rows.append(
            [
                {"token": tokens[index], "weight": float(valid_weights[index])}
                for index in order
            ]
        )
    return rows


def overfit_mode(mode, controlled, args, pooled_cache, token_cache):
    set_seed(args.seed)
    model = LanguageBCPolicy(
        language_mode=mode,
        pretrained_visual=args.pretrained_visual,
        freeze_visual_backbone=True,
        dropout=0.0,
    )
    model.train()
    language = language_inputs(
        mode, controlled["instructions"], pooled_cache, token_cache
    )
    inputs = {
        "image": controlled["image"],
        "proprio": controlled["proprio"],
        **language,
    }
    target = controlled["target"]
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    with torch.no_grad():
        initial_output = model(**inputs)
        initial_language_cosine = cosine_pair(initial_output["language_feature"])
        initial_prediction_distance = float(
            torch.linalg.vector_norm(
                initial_output["action"][0] - initial_output["action"][1]
            )
        )

    losses = []
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        loss = nn.functional.mse_loss(output["action"], target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        losses.append(float(loss.detach()))
        if step == args.steps:
            break
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            prediction_distance = float(
                torch.linalg.vector_norm(
                    output["action"][0] - output["action"][1]
                ).detach()
            )
            print(
                f"[{mode}] step {step + 1:4d}/{args.steps}: "
                f"loss={losses[-1]:.8f}, "
                f"prediction_distance={prediction_distance:.4f}"
            )

    model.eval()
    with torch.no_grad():
        final_output = model(**inputs)
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = 1.0 - final_loss / max(initial_loss, 1e-12)
    final_prediction_distance = float(
        torch.linalg.vector_norm(
            final_output["action"][0] - final_output["action"][1]
        )
    )
    final_language_cosine = cosine_pair(final_output["language_feature"])
    passed = (
        reduction >= args.required_reduction
        and final_loss <= args.max_final_loss
        and final_prediction_distance >= 0.9 * controlled["target_distance"]
    )
    result = {
        "language_mode": mode,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": reduction,
        "initial_language_cosine": initial_language_cosine,
        "final_language_cosine": final_language_cosine,
        "target_action_distance": controlled["target_distance"],
        "initial_prediction_distance": initial_prediction_distance,
        "final_prediction_distance": final_prediction_distance,
        "passed": passed,
        "losses": losses,
    }
    if mode == "token_attention":
        result["top_attention_tokens"] = top_attention(
            token_cache,
            controlled["instructions"],
            final_output["attention_weights"].cpu(),
        )
    print(
        f"[{mode}] initial={initial_loss:.8f}, final={final_loss:.8f}, "
        f"reduction={reduction:.2%}, language_cosine="
        f"{initial_language_cosine:.4f}->{final_language_cosine:.4f}, "
        f"passed={passed}"
    )
    return result


def main():
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("--steps and --learning-rate must be positive")
    if args.image_size < 32 or args.log_every <= 0:
        raise ValueError("Invalid image size or log interval")
    if not 0.0 <= args.required_reduction <= 1.0:
        raise ValueError("--required-reduction must be in [0, 1]")

    set_seed(args.seed)
    normalizer = PolicyNormalizer.from_json(args.normalization)
    normalizer.validate_split(args.split)
    pooled_cache = CachedLanguageEmbeddings(args.pooled_cache)
    token_cache = CachedTokenEmbeddings(args.token_cache)
    records = stack_records(args.dataset, args.split)
    pair = select_reversed_pair(records, pooled_cache)
    controlled = controlled_batch(pair, normalizer, args.image_size)

    print("Controlled reversed-order pair")
    print("------------------------------")
    print("Template:", controlled["template_id"])
    print("Order A:", controlled["orders"][0])
    print("Order B:", controlled["orders"][1])
    print("Instruction A:", controlled["instructions"][0])
    print("Instruction B:", controlled["instructions"][1])
    print("Episode steps:", controlled["steps"])
    print("Pooled cosine:", pair[0])
    print("Target normalized-action distance:", controlled["target_distance"])
    print("Identical RGB:", torch.equal(controlled["image"][0], controlled["image"][1]))
    print(
        "Identical proprio:",
        torch.equal(controlled["proprio"][0], controlled["proprio"][1]),
    )

    modes = LANGUAGE_MODES if args.language_mode == "both" else (args.language_mode,)
    results = [
        overfit_mode(mode, controlled, args, pooled_cache, token_cache)
        for mode in modes
    ]
    report = {
        "seed": args.seed,
        "controlled_pair": {
            key: value
            for key, value in controlled.items()
            if key not in {"image", "proprio", "target"}
        },
        "pooled_input_cosine": pair[0],
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
            "Language disambiguation overfit failed for: " + ", ".join(failures)
        )
    print("Language-BC controlled disambiguation overfit: PASS")


if __name__ == "__main__":
    main()
