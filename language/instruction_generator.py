import numpy as np


TEMPLATES = {
    "pick": {
        "train": [
            ("pick_train_01", "Pick up the {color} cube."),
            ("pick_train_02", "Grab the {color} cube."),
            ("pick_train_03", "Lift the {color} cube."),
            ("pick_train_04", "Pick the {color} cube up."),
            ("pick_train_05", "Take the {color} cube."),
        ],
        "held_out": [
            ("pick_held_out_01", "Raise the {color} cube from the table."),
            ("pick_held_out_02", "Grasp and lift the {color} cube."),
        ],
    },
    "place": {
        "train": [
            ("place_train_01", "Place the {object_color} cube into the {target_color} tray."),
            ("place_train_02", "Put the {object_color} cube in the {target_color} container."),
            ("place_train_03", "Move the {object_color} cube into the {target_color} tray."),
            ("place_train_04", "Pick up the {object_color} cube and place it in the {target_color} container."),
            ("place_train_05", "Set the {object_color} cube inside the {target_color} tray."),
            ("place_train_06", "Put the {object_color} block into the {target_color} tray."),
        ],
        "held_out": [
            ("place_held_out_01", "Transfer the {object_color} cube to the {target_color} tray."),
            ("place_held_out_02", "Pick up the {object_color} cube and deposit it in the {target_color} container."),
            ("place_held_out_03", "Relocate the {object_color} cube into the {target_color} tray."),
        ],
    },
    "stack": {
        "train": [
            ("stack_train_01", "Stack {colors} from bottom to top."),
            ("stack_train_02", "Build a stack with {positions}."),
            ("stack_train_03", "Arrange the cubes vertically with {positions}."),
            ("stack_train_04", "Make a tower ordered {colors} from bottom to top."),
            ("stack_train_05", "Place the cubes in this bottom-to-top order: {colors}."),
        ],
        "held_out": [
            ("stack_held_out_01", "Construct a tower where {positions}."),
            ("stack_held_out_02", "Pile the cubes so that {relations}."),
        ],
    },
}


def _join_colors(colors):
    if len(colors) == 2:
        return f"{colors[0]} and {colors[1]}"
    return ", ".join(colors[:-1]) + f", and {colors[-1]}"


def _positions(colors):
    labels = ["at the bottom"]
    if len(colors) == 3:
        labels += ["in the middle", "on top"]
    else:
        labels += [f"at level {index + 1}" for index in range(1, len(colors) - 1)]
        labels += ["on top"]
    return ", ".join(
        f"{color} {label}" for color, label in zip(colors, labels)
    )


def _relations(colors):
    return ", then ".join(
        f"the {top} cube is on the {support} cube"
        for support, top in zip(colors[:-1], colors[1:])
    )


def generate_instruction(task, rng, split="train", **attributes):
    if task not in TEMPLATES:
        raise KeyError(f"Unknown instruction task: {task}")
    if split not in TEMPLATES[task]:
        raise KeyError(f"Unknown instruction split for {task}: {split}")

    templates = TEMPLATES[task][split]
    index = int(rng.integers(len(templates)))
    template_id, template = templates[index]
    values = dict(attributes)
    if task == "stack":
        colors = list(attributes["colors"])
        values.update(
            colors=_join_colors(colors),
            positions=_positions(colors),
            relations=_relations(colors),
        )
    return {
        "instruction": template.format(**values),
        "instruction_template_id": template_id,
        "instruction_split": split,
    }
