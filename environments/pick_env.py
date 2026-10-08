import numpy as np

from environments.entity_utils import COLOR_PALETTE, sample_unique_colors, semantic_split_for
from language.instruction_generator import generate_instruction
from environments.stack_env import StackEnv


class PickEnv(StackEnv):
    """Multi-object pick task with one instructed target and distractor cubes."""

    COLOR_PALETTE = COLOR_PALETTE

    def __init__(
        self,
        *args,
        num_objects=3,
        instruction_split="train",
        fixed_target_color=None,
        **kwargs,
    ):
        if num_objects not in (3, 4):
            raise ValueError("num_objects must be either 3 or 4")
        self.target_object_id = None
        self.instruction_split = instruction_split
        if fixed_target_color is not None and fixed_target_color not in COLOR_PALETTE:
            raise ValueError(
                f"Unknown fixed target color: {fixed_target_color!r}"
            )
        self.fixed_target_color = fixed_target_color
        self.instruction_metadata = {}
        # Pick was already reliable at this density. Stack uses a wider
        # clearance because it repeatedly returns near a growing tower.
        kwargs.setdefault("minimum_object_distance", 0.10)
        kwargs.setdefault("x_range", (-0.10, 0.10))
        kwargs.setdefault("y_range", (-0.20, 0.20))
        super().__init__(*args, num_cubes=num_objects, **kwargs)
        self.num_objects = self.num_cubes

    def _sample_task_metadata(self):
        if self.fixed_target_color is None:
            colors = sample_unique_colors(self.rng, self.num_cubes)
        else:
            distractor_palette = [
                color
                for color in COLOR_PALETTE
                if color != self.fixed_target_color
            ]
            distractor_colors = [
                str(color)
                for color in self.rng.choice(
                    distractor_palette,
                    size=self.num_cubes - 1,
                    replace=False,
                )
            ]
            colors = [self.fixed_target_color, *distractor_colors]
            self.rng.shuffle(colors)
        self.object_id_to_color = dict(zip(self.object_ids, colors))
        self.color_to_object_id = {
            color: object_id for object_id, color in self.object_id_to_color.items()
        }
        if self.fixed_target_color is None:
            self.target_object_id = str(self.rng.choice(self.object_ids))
        else:
            self.target_object_id = self.color_to_object_id[
                self.fixed_target_color
            ]
        self.stack_order = []
        target_color = self.object_id_to_color[self.target_object_id]
        self.instruction_metadata = generate_instruction(
            "pick",
            self.rng,
            split=self.instruction_split,
            color=target_color,
        )

    def resolve_object_id(self, identifier):
        if identifier is None:
            return self.target_object_id
        return super().resolve_object_id(identifier)

    def get_task_metadata(self):
        target_color = self.object_id_to_color[self.target_object_id]
        return {
            "task": "pick",
            "objects": [
                {
                    "id": object_id,
                    "shape": "cube",
                    "color": self.object_id_to_color[object_id],
                    "is_target": object_id == self.target_object_id,
                }
                for object_id in self.object_ids
            ],
            "target_object": self.target_object_id,
            "target_color": target_color,
            "policy_conditioning": (
                "vision_only"
                if self.fixed_target_color is not None
                else "language"
            ),
            "distractors": [
                object_id
                for object_id in self.object_ids
                if object_id != self.target_object_id
            ],
            "semantic_split": semantic_split_for("pick", color=target_color),
            **self.instruction_metadata,
        }

    def _check_success(self):
        if self.target_object_id is None:
            return False
        target_z = self.get_object_position(self.target_object_id)[2]
        table_center_z = self.table_offset[2] + self.cube_half_size
        return bool(target_z - table_center_z >= 0.08)

    def reward(self, action=None):
        return float(self._check_success())
