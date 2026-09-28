from collections import OrderedDict

import numpy as np
from robosuite.environments.manipulation.manipulation_env import (
    ManipulationEnv,
)
from robosuite.models.arenas import TableArena
from robosuite.models.objects import (
    BoxObject,
    CompositeObject,
)
from robosuite.models.tasks import (
    ManipulationTask,
)
from robosuite.utils import RandomizationError
from robosuite.utils.mjcf_utils import (
    add_to_dict,
)
from robosuite.utils.observables import (
    Observable,
    sensor,
)
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from robosuite.utils.transform_utils import (
    convert_quat,
)

from language.instruction_generator import generate_instruction

from environments.entity_utils import (
    COLOR_PALETTE,
    sample_unique_colors,
    semantic_split_for,
)

# ============================================================
# CUSTOM TRAY OBJECT
# ============================================================


class TrayObject(CompositeObject):
    """
    Simple shallow rectangular tray.

    Built from:
        1 base
        4 walls

    All dimensions below are HALF-SIZES where appropriate,
    following MuJoCo box geom convention.
    """

    def __init__(
        self,
        name="tray",
        inner_half_size=(0.06, 0.045),
        wall_thickness=0.005,
        wall_height=0.015,
        base_thickness=0.004,
        rgba=(0.2, 0.4, 0.9, 1.0),
    ):
        self._name = name

        self.inner_half_size = np.asarray(
            inner_half_size,
            dtype=np.float32,
        )

        self.wall_thickness = float(wall_thickness)

        self.wall_height = float(wall_height)

        self.base_thickness = float(base_thickness)

        self.rgba = np.asarray(
            rgba,
            dtype=np.float32,
        )

        super().__init__(**self._get_geom_attrs())

    def _get_geom_attrs(self):

        inner_x = self.inner_half_size[0]
        inner_y = self.inner_half_size[1]

        t = self.wall_thickness
        wall_h = self.wall_height
        base_t = self.base_thickness

        outer_x = inner_x + t
        outer_y = inner_y + t

        # Total object half-size estimate
        total_size = np.array(
            [
                outer_x,
                outer_y,
                wall_h + base_t,
            ],
            dtype=np.float32,
        )

        base_args = {
            "total_size": total_size,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }

        obj_args = {}

        # ====================================================
        # BASE
        # ====================================================

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                0.0,
                0.0,
                0.0,
            ),
            geom_quats=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            geom_sizes=np.array(
                [
                    outer_x,
                    outer_y,
                    base_t,
                ]
            ),
            geom_names="base",
            geom_rgbas=self.rgba,
            geom_materials=None,
            geom_frictions=(
                1.0,
                0.005,
                0.0001,
            ),
            density=1000,
        )

        # ====================================================
        # LEFT WALL
        # ====================================================

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                0.0,
                outer_y - t,
                wall_h,
            ),
            geom_quats=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            geom_sizes=np.array(
                [
                    outer_x,
                    t,
                    wall_h,
                ]
            ),
            geom_names="wall_left",
            geom_rgbas=self.rgba,
            geom_materials=None,
            geom_frictions=(
                1.0,
                0.005,
                0.0001,
            ),
            density=1000,
        )

        # ====================================================
        # RIGHT WALL
        # ====================================================

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                0.0,
                -outer_y + t,
                wall_h,
            ),
            geom_quats=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            geom_sizes=np.array(
                [
                    outer_x,
                    t,
                    wall_h,
                ]
            ),
            geom_names="wall_right",
            geom_rgbas=self.rgba,
            geom_materials=None,
            geom_frictions=(
                1.0,
                0.005,
                0.0001,
            ),
            density=1000,
        )

        # ====================================================
        # FRONT WALL
        # ====================================================

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                outer_x - t,
                0.0,
                wall_h,
            ),
            geom_quats=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            geom_sizes=np.array(
                [
                    t,
                    inner_y,
                    wall_h,
                ]
            ),
            geom_names="wall_front",
            geom_rgbas=self.rgba,
            geom_materials=None,
            geom_frictions=(
                1.0,
                0.005,
                0.0001,
            ),
            density=1000,
        )

        # ====================================================
        # BACK WALL
        # ====================================================

        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(
                -outer_x + t,
                0.0,
                wall_h,
            ),
            geom_quats=(
                1.0,
                0.0,
                0.0,
                0.0,
            ),
            geom_sizes=np.array(
                [
                    t,
                    inner_y,
                    wall_h,
                ]
            ),
            geom_names="wall_back",
            geom_rgbas=self.rgba,
            geom_materials=None,
            geom_frictions=(
                1.0,
                0.005,
                0.0001,
            ),
            density=1000,
        )

        obj_args.update(base_args)

        return obj_args


# ============================================================
# PLACE ENVIRONMENT
# ============================================================


class PlaceEnv(ManipulationEnv):
    """Multi-object, multi-receptacle semantic pick-and-place task."""

    COLOR_PALETTE = COLOR_PALETTE

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        num_objects=3,
        num_receptacles=3,
        instruction_split="train",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        minimum_object_distance=0.10,
        minimum_object_receptacle_distance=0.12,
        minimum_receptacle_distance=0.14,
        use_camera_obs=False,
        use_object_obs=True,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=2500,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
    ):
        if num_objects not in (1, 2, 3, 4):
            raise ValueError("num_objects must be between 1 and 4")
        if num_receptacles not in (1, 2, 3, 4):
            raise ValueError("num_receptacles must be between 1 and 4")

        self.num_objects = int(num_objects)
        self.num_receptacles = int(num_receptacles)
        self.object_ids = [f"object_{index}" for index in range(self.num_objects)]
        self.receptacle_ids = [
            f"receptacle_{index}" for index in range(self.num_receptacles)
        ]
        self.table_full_size = np.asarray(table_full_size, dtype=np.float32)
        self.table_friction = table_friction
        self.table_offset = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self.minimum_object_distance = float(minimum_object_distance)
        self.minimum_object_receptacle_distance = float(
            minimum_object_receptacle_distance
        )
        self.minimum_receptacle_distance = float(minimum_receptacle_distance)
        self.minimum_entity_distance = min(
            self.minimum_object_distance,
            self.minimum_object_receptacle_distance,
            self.minimum_receptacle_distance,
        )
        if self.minimum_entity_distance < 0.0:
            raise ValueError("Entity clearances must be non-negative")
        self.use_object_obs = use_object_obs

        self.object_id_to_color = {}
        self.receptacle_id_to_color = {}
        self.object_color_to_id = {}
        self.receptacle_color_to_id = {}
        self.target_object_id = None
        self.target_receptacle_id = None
        self.cube_color = None
        self.receptacle_color = None
        self.instruction_split = instruction_split
        self.instruction_metadata = {}
        self.last_sampling_stats = {
            "attempts": 0,
            "sampler_errors": 0,
            "clearance_rejections": 0,
        }

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    def reward(self, action=None):
        return float(self._check_success())

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0.0, 0.0, 0.0])

        initial_colors = list(self.COLOR_PALETTE.values())
        self.objects = [
            BoxObject(
                name=object_id,
                size=[0.020, 0.020, 0.020],
                rgba=initial_colors[index],
                rng=self.rng,
            )
            for index, object_id in enumerate(self.object_ids)
        ]
        self.receptacles = [
            TrayObject(
                name=receptacle_id,
                inner_half_size=(0.050, 0.040),
                wall_thickness=0.004,
                wall_height=0.015,
                base_thickness=0.004,
                rgba=initial_colors[index],
            )
            for index, receptacle_id in enumerate(self.receptacle_ids)
        ]

        self.cube = self.objects[0]
        self.receptacle = self.receptacles[0]
        self.placement_initializer = UniformRandomSampler(
            name="PlaceEntitySampler",
            mujoco_objects=self.objects + self.receptacles,
            x_range=(-0.16, 0.16),
            y_range=(-0.30, 0.30),
            rotation=(-np.pi / 4, np.pi / 4),
            rotation_axis="z",
            ensure_object_boundary_in_range=True,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.receptacles,
        )

    def _setup_references(self):
        super()._setup_references()
        entities = self.objects + self.receptacles
        self.entity_body_ids = {
            entity.name: self.sim.model.body_name2id(entity.root_body)
            for entity in entities
        }
        self.entity_visual_geom_ids = {
            entity.name: [
                self.sim.model.geom_name2id(name) for name in entity.visual_geoms
            ]
            for entity in entities
        }
        self.cube_body_id = self.entity_body_ids[self.object_ids[0]]
        self.receptacle_body_id = self.entity_body_ids[self.receptacle_ids[0]]
        self.cube_visual_geom_ids = self.entity_visual_geom_ids[self.object_ids[0]]
        self.receptacle_visual_geom_ids = self.entity_visual_geom_ids[
            self.receptacle_ids[0]
        ]

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables

        for entity_id in self.object_ids + self.receptacle_ids:
            body_id = self.entity_body_ids[entity_id]

            @sensor(modality="object")
            def entity_pos(obs_cache, current_body_id=body_id):
                return np.array(self.sim.data.body_xpos[current_body_id])

            @sensor(modality="object")
            def entity_quat(obs_cache, current_body_id=body_id):
                return convert_quat(
                    np.array(self.sim.data.body_xquat[current_body_id]),
                    to="xyzw",
                )

            pos_name = f"{entity_id}_pos"
            quat_name = f"{entity_id}_quat"
            entity_pos.__name__ = pos_name
            entity_quat.__name__ = quat_name
            observables[pos_name] = Observable(
                name=pos_name, sensor=entity_pos, sampling_rate=self.control_freq
            )
            observables[quat_name] = Observable(
                name=quat_name, sensor=entity_quat, sampling_rate=self.control_freq
            )

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return

        placements = self._sample_safe_placements()
        for position, quaternion, entity in placements.values():
            self.sim.data.set_joint_qpos(
                entity.joints[0],
                np.concatenate([np.asarray(position), np.asarray(quaternion)]),
            )

        object_colors = sample_unique_colors(self.rng, self.num_objects)
        receptacle_colors = sample_unique_colors(self.rng, self.num_receptacles)
        self.object_id_to_color = dict(zip(self.object_ids, object_colors))
        self.receptacle_id_to_color = dict(
            zip(self.receptacle_ids, receptacle_colors)
        )
        self.object_color_to_id = {
            color: object_id for object_id, color in self.object_id_to_color.items()
        }
        self.receptacle_color_to_id = {
            color: receptacle_id
            for receptacle_id, color in self.receptacle_id_to_color.items()
        }
        self.target_object_id = str(self.rng.choice(self.object_ids))
        self.target_receptacle_id = str(self.rng.choice(self.receptacle_ids))
        self.cube_color = self.object_id_to_color[self.target_object_id]
        self.receptacle_color = self.receptacle_id_to_color[
            self.target_receptacle_id
        ]
        self.instruction_metadata = generate_instruction(
            "place",
            self.rng,
            split=self.instruction_split,
            object_color=self.cube_color,
            target_color=self.receptacle_color,
        )
        self._apply_colors()

    def _sample_safe_placements(self):
        entity_ids = self.object_ids + self.receptacle_ids
        stats = {
            "attempts": 0,
            "sampler_errors": 0,
            "clearance_rejections": 0,
        }
        for _ in range(500):
            stats["attempts"] += 1
            try:
                placements = self.placement_initializer.sample()
            except RandomizationError:
                stats["sampler_errors"] += 1
                continue

            positions = {
                entity_id: np.asarray(placements[entity_id][0])
                for entity_id in entity_ids
            }
            if all(
                np.linalg.norm(positions[first][:2] - positions[second][:2])
                >= self.required_clearance(first, second)
                for index, first in enumerate(entity_ids)
                for second in entity_ids[index + 1 :]
            ):
                self.last_sampling_stats = stats
                return placements

            stats["clearance_rejections"] += 1

        self.last_sampling_stats = stats
        raise RandomizationError(
            "Could not sample safe multi-place placements after 500 attempts: "
            f"{stats}"
        )

    def required_clearance(self, first_id, second_id):
        first_is_object = first_id in self.object_ids
        second_is_object = second_id in self.object_ids
        if first_is_object and second_is_object:
            return self.minimum_object_distance
        if not first_is_object and not second_is_object:
            return self.minimum_receptacle_distance
        return self.minimum_object_receptacle_distance

    def _apply_colors(self):
        colors = {**self.object_id_to_color, **self.receptacle_id_to_color}
        for entity_id, color in colors.items():
            for geom_id in self.entity_visual_geom_ids[entity_id]:
                self.sim.model.geom_rgba[geom_id] = self.COLOR_PALETTE[color]

    def resolve_object_id(self, identifier):
        if identifier is None:
            return self.target_object_id
        if identifier in self.object_ids:
            return identifier
        if identifier in self.object_color_to_id:
            return self.object_color_to_id[identifier]
        raise KeyError(f"Unknown object identifier: {identifier}")

    def resolve_receptacle_id(self, identifier):
        if identifier is None:
            return self.target_receptacle_id
        if identifier in self.receptacle_ids:
            return identifier
        if identifier in self.receptacle_color_to_id:
            return self.receptacle_color_to_id[identifier]
        raise KeyError(f"Unknown receptacle identifier: {identifier}")

    def get_object_position(self, identifier=None):
        object_id = self.resolve_object_id(identifier)
        return np.array(self.sim.data.body_xpos[self.entity_body_ids[object_id]])

    def get_object_orientation(self, identifier=None):
        object_id = self.resolve_object_id(identifier)
        return convert_quat(
            np.array(self.sim.data.body_xquat[self.entity_body_ids[object_id]]),
            to="xyzw",
        )

    def get_receptacle_position(self, identifier=None):
        receptacle_id = self.resolve_receptacle_id(identifier)
        return np.array(
            self.sim.data.body_xpos[self.entity_body_ids[receptacle_id]]
        )

    def get_receptacle_orientation(self, identifier=None):
        receptacle_id = self.resolve_receptacle_id(identifier)
        return convert_quat(
            np.array(self.sim.data.body_xquat[self.entity_body_ids[receptacle_id]]),
            to="xyzw",
        )

    def get_task_metadata(self):
        object_color = self.object_id_to_color[self.target_object_id]
        receptacle_color = self.receptacle_id_to_color[self.target_receptacle_id]
        return {
            "task": "place",
            "objects": [
                {
                    "id": object_id,
                    "shape": "cube",
                    "color": self.object_id_to_color[object_id],
                    "is_target": object_id == self.target_object_id,
                }
                for object_id in self.object_ids
            ],
            "receptacles": [
                {
                    "id": receptacle_id,
                    "type": "tray",
                    "color": self.receptacle_id_to_color[receptacle_id],
                    "is_target": receptacle_id == self.target_receptacle_id,
                }
                for receptacle_id in self.receptacle_ids
            ],
            "target_object": self.target_object_id,
            "target_receptacle": self.target_receptacle_id,
            "sampling_stats": self.last_sampling_stats.copy(),
            "semantic_split": semantic_split_for(
                "place",
                object_color=object_color,
                target_color=receptacle_color,
            ),
            **self.instruction_metadata,
        }

    def _check_success(self):
        if self.target_object_id is None or self.target_receptacle_id is None:
            return False
        object_position = self.get_object_position(self.target_object_id)
        receptacle_position = self.get_receptacle_position(
            self.target_receptacle_id
        )
        return bool(
            np.linalg.norm(object_position[:2] - receptacle_position[:2]) <= 0.04
        )
