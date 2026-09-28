import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from language.instruction_generator import generate_instruction


COLOR_PALETTE = {
    "red": np.array([0.90, 0.10, 0.10, 1.0], dtype=np.float32),
    "blue": np.array([0.10, 0.30, 0.90, 1.0], dtype=np.float32),
    "green": np.array([0.10, 0.70, 0.20, 1.0], dtype=np.float32),
    "yellow": np.array([0.95, 0.75, 0.10, 1.0], dtype=np.float32),
}


def sample_unique_colors(rng, count):
    if count > len(COLOR_PALETTE):
        raise ValueError("Not enough unique colors in COLOR_PALETTE")
    names = np.asarray(list(COLOR_PALETTE), dtype=object)
    return [str(value) for value in rng.choice(names, size=count, replace=False)]


@lru_cache(maxsize=1)
def load_split_registry():
    path = Path(__file__).resolve().parents[1] / "configs" / "dataset_splits.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def semantic_split_for(task, **attributes):
    registry = load_split_registry()
    if task == "pick":
        return "train"
    if task == "place":
        combination = [attributes["object_color"], attributes["target_color"]]
        held_out = registry["place"]["held_out_combinations"]
        return "compositional_test" if combination in held_out else "train"
    if task == "stack":
        order = list(attributes["colors"])
        held_out = registry["stack"]["held_out_orders"]
        return "compositional_test" if order in held_out else "train"
    raise KeyError(f"Unknown semantic task: {task}")


# Compatibility helpers for older callers.
def pick_instruction(color):
    rng = np.random.default_rng(0)
    return generate_instruction("pick", rng, color=color)["instruction"]


def place_instruction(object_color, receptacle_color):
    rng = np.random.default_rng(0)
    return generate_instruction(
        "place", rng, object_color=object_color, target_color=receptacle_color
    )["instruction"]


def stack_instruction(colors):
    rng = np.random.default_rng(0)
    return generate_instruction("stack", rng, colors=colors)["instruction"]
