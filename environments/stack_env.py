from itertools import combinations

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import RandomizationError
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from environments.entity_utils import semantic_split_for
from language.instruction_generator import generate_instruction


class StackEnv(ManipulationEnv):
    """Multi-object cube stacking environment with stable object IDs."""

    COLOR_PALETTE = {
        "red": np.array([0.90, 0.10, 0.10, 1.0], dtype=np.float32),
        "blue": np.array([0.10, 0.30, 0.90, 1.0], dtype=np.float32),
        "green": np.array([0.10, 0.70, 0.20, 1.0], dtype=np.float32),
        "yellow": np.array([0.95, 0.75, 0.10, 1.0], dtype=np.float32),
    }

    def __init__(
        self,
        robots="Panda",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        num_cubes=3,
        instruction_split="train",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 0.005, 0.0001),
        cube_half_size=0.02,
        x_range=(-0.13, 0.13),
        y_range=(-0.24, 0.24),
        yaw_range=(-np.pi / 4, np.pi / 4),
        minimum_object_distance=0.13,
        stack_xy_tolerance=0.02,
        stack_z_tolerance=0.01,
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
        horizon=3000,
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
        if num_cubes not in (3, 4):
            raise ValueError("num_cubes must be either 3 or 4")
        if minimum_object_distance < 0.0:
            raise ValueError("minimum_object_distance must be non-negative")

        self.num_cubes = int(num_cubes)
        self.object_ids = [f"cube_{index}" for index in range(self.num_cubes)]
        self.table_full_size = np.asarray(table_full_size, dtype=np.float32)
        self.table_friction = table_friction
        self.table_offset = np.array([0.0, 0.0, 0.8], dtype=np.float32)
        self.cube_half_size = float(cube_half_size)
        self.cube_height = 2.0 * self.cube_half_size
        self.x_range = tuple(float(value) for value in x_range)
        self.y_range = tuple(float(value) for value in y_range)
        self.yaw_range = tuple(float(value) for value in yaw_range)
        self.minimum_object_distance = float(minimum_object_distance)
        self.stack_xy_tolerance = float(stack_xy_tolerance)
        self.stack_z_tolerance = float(stack_z_tolerance)
        self.use_object_obs = use_object_obs

        self.object_id_to_color = {}
        self.color_to_object_id = {}
        self.stack_order = []
        self.instruction_split = instruction_split
        self.instruction_metadata = {}

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

        base_position = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(base_position)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0.0, 0.0, 0.0])

        initial_colors = list(self.COLOR_PALETTE.values())[: self.num_cubes]
        self.cubes = [
            BoxObject(
                name=object_id,
                size=[
                    self.cube_half_size,
                    self.cube_half_size,
                    self.cube_half_size,
                ],
                rgba=initial_colors[index],
                rng=self.rng,
            )
            for index, object_id in enumerate(self.object_ids)
        ]

        self.placement_initializer = UniformRandomSampler(
            name="StackCubeSampler",
            mujoco_objects=self.cubes,
            x_range=self.x_range,
            y_range=self.y_range,
            rotation=self.yaw_range,
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
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        super()._setup_references()

        self.object_body_ids = {
            cube.name: self.sim.model.body_name2id(cube.root_body)
            for cube in self.cubes
        }
        self.object_visual_geom_ids = {
            cube.name: [
                self.sim.model.geom_name2id(geom_name)
                for geom_name in cube.visual_geoms
            ]
            for cube in self.cubes
        }

    def _setup_observables(self):
        observables = super()._setup_observables()

        if not self.use_object_obs:
            return observables

        for object_id in self.object_ids:
            sensors, names = self._create_object_sensors(object_id)
            for name, object_sensor in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=object_sensor,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _create_object_sensors(self, object_id):
        @sensor(modality="object")
        def object_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.object_body_ids[object_id]])

        @sensor(modality="object")
        def object_quat(obs_cache):
            body_quat = np.array(
                self.sim.data.body_xquat[self.object_body_ids[object_id]]
            )
            return convert_quat(body_quat, to="xyzw")

        return [object_pos, object_quat], [
            f"{object_id}_pos",
            f"{object_id}_quat",
        ]

    def _reset_internal(self):
        super()._reset_internal()

        if self.deterministic_reset:
            return

        self._sample_task_metadata()
        placements = self._sample_safe_placements()

        for object_position, object_quat, cube in placements.values():
            self.sim.data.set_joint_qpos(
                cube.joints[0],
                np.concatenate(
                    [
                        np.asarray(object_position),
                        np.asarray(object_quat),
                    ]
                ),
            )

        self._apply_cube_colors()

    def _sample_task_metadata(self):
        color_names = np.asarray(list(self.COLOR_PALETTE.keys()), dtype=object)
        sampled_colors = self.rng.choice(
            color_names,
            size=self.num_cubes,
            replace=False,
        ).tolist()

        self.object_id_to_color = dict(zip(self.object_ids, sampled_colors))
        self.color_to_object_id = {
            color: object_id
            for object_id, color in self.object_id_to_color.items()
        }
        self.stack_order = self.rng.permutation(sampled_colors).tolist()
        self.instruction_metadata = generate_instruction(
            "stack",
            self.rng,
            split=self.instruction_split,
            colors=self.stack_order,
        )

    def _sample_safe_placements(self):
        for _ in range(300):
            placements = self.placement_initializer.sample()
            positions = {
                object_id: np.asarray(placements[object_id][0], dtype=np.float32)
                for object_id in self.object_ids
            }

            if all(
                np.linalg.norm(positions[first][:2] - positions[second][:2])
                >= self.minimum_object_distance
                for first, second in combinations(self.object_ids, 2)
            ):
                return placements

        raise RandomizationError("Could not sample safely separated cube placements")

    def _apply_cube_colors(self):
        for object_id, color_name in self.object_id_to_color.items():
            rgba = self.COLOR_PALETTE[color_name]
            for geom_id in self.object_visual_geom_ids[object_id]:
                self.sim.model.geom_rgba[geom_id] = rgba

    def resolve_object_id(self, identifier):
        if identifier in self.object_ids:
            return identifier
        if identifier in self.color_to_object_id:
            return self.color_to_object_id[identifier]
        raise KeyError(
            f"Unknown object identifier '{identifier}'. "
            f"Available IDs: {self.object_ids}; "
            f"available colors: {list(self.color_to_object_id)}"
        )

    def get_object_position(self, identifier):
        object_id = self.resolve_object_id(identifier)
        return np.array(self.sim.data.body_xpos[self.object_body_ids[object_id]])

    def get_object_orientation(self, identifier):
        object_id = self.resolve_object_id(identifier)
        body_quat = np.array(self.sim.data.body_xquat[self.object_body_ids[object_id]])
        return convert_quat(body_quat, to="xyzw")

    def get_task_metadata(self):
        return {
            "task": "stack",
            "num_objects": self.num_cubes,
            "objects": [
                {
                    "id": object_id,
                    "color": self.object_id_to_color[object_id],
                }
                for object_id in self.object_ids
            ],
            "stack_order": self.stack_order.copy(),
            "semantic_split": semantic_split_for(
                "stack", colors=self.stack_order
            ),
            **self.instruction_metadata,
        }

    def get_stack_errors(self):
        errors = []
        for support_color, top_color in zip(
            self.stack_order[:-1],
            self.stack_order[1:],
        ):
            support_position = self.get_object_position(support_color)
            top_position = self.get_object_position(top_color)
            errors.append(
                {
                    "support": support_color,
                    "top": top_color,
                    "xy_error": float(
                        np.linalg.norm(top_position[:2] - support_position[:2])
                    ),
                    "z_error": float(
                        abs(
                            (top_position[2] - support_position[2])
                            - self.cube_height
                        )
                    ),
                }
            )
        return errors

    def _check_success(self):
        if len(self.stack_order) != self.num_cubes:
            return False

        bottom_position = self.get_object_position(self.stack_order[0])
        expected_bottom_z = self.table_offset[2] + self.cube_half_size
        bottom_on_table = (
            abs(bottom_position[2] - expected_bottom_z) <= self.stack_z_tolerance
        )

        return bottom_on_table and all(
            error["xy_error"] <= self.stack_xy_tolerance
            and error["z_error"] <= self.stack_z_tolerance
            for error in self.get_stack_errors()
        )
