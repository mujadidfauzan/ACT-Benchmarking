import numpy as np
from robosuite.utils import transform_utils as T

from experts.primitives import grasp_object


class PickExpert:

    def __init__(
        self,
        controller,
        approach_height=0.15,
        pre_grasp_height=0.05,
        lift_height=0.20,
        grasp_offset=0.0,
        target_orientation=None,
    ):
        self.controller = controller

        self.approach_height = approach_height
        self.pre_grasp_height = pre_grasp_height
        self.lift_height = lift_height

        self.grasp_offset = grasp_offset

        self.target_orientation = target_orientation

    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================

    def get_target_orientation(self, obs, get_object_orientation=None):
        """
        Keep the gripper's initial roll and pitch,
        but align its yaw with the cube yaw.

        Quaternion convention:
        [x, y, z, w]
        """

        # --------------------------------------------------------
        # 1. Read current EEF orientation
        # --------------------------------------------------------

        eef_quat = self.controller.get_eef_orientation(obs).copy()

        # --------------------------------------------------------
        # 2. Read cube orientation
        # --------------------------------------------------------

        orientation_getter = (
            get_object_orientation or self.controller.get_cube_orientation
        )
        cube_quat = orientation_getter(obs).copy()

        # --------------------------------------------------------
        # 3. Convert both to Euler:
        #    roll, pitch, yaw
        # --------------------------------------------------------

        eef_rotmat = T.quat2mat(eef_quat)
        eef_euler = T.mat2euler(eef_rotmat)

        cube_rotmat = T.quat2mat(cube_quat)
        cube_euler = T.mat2euler(cube_rotmat)

        # --------------------------------------------------------
        # 4. Keep EEF roll + pitch
        #    Use cube yaw
        # --------------------------------------------------------

        target_euler = np.array(
            [
                eef_euler[0],  # roll from gripper
                eef_euler[1],  # pitch from gripper
                cube_euler[2],  # yaw from cube
            ],
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # 5. Convert back to quaternion
        # --------------------------------------------------------

        target_quat = T.mat2quat(T.euler2mat(target_euler)).astype(np.float32)

        print(
            "EEF Euler [r p y]:",
            np.round(
                np.rad2deg(eef_euler),
                2,
            ),
        )

        print(
            "Cube Euler [r p y]:",
            np.round(
                np.rad2deg(cube_euler),
                2,
            ),
        )

        print(
            "Target Euler [r p y]:",
            np.round(
                np.rad2deg(target_euler),
                2,
            ),
        )

        return target_quat

    # ============================================================
    # MAIN EXPERT
    # ============================================================

    def run(
        self,
        obs,
        render=False,
        object_name=None,
        get_object_position=None,
        get_object_orientation=None,
    ):

        if object_name is None and hasattr(self.controller.env, "target_object_id"):
            object_name = self.controller.env.target_object_id

        if object_name is not None:
            object_id = self.controller.env.resolve_object_id(object_name)
            if get_object_position is None:
                get_object_position = lambda current_obs: np.asarray(
                    current_obs[f"{object_id}_pos"], dtype=np.float32
                )
            if get_object_orientation is None:
                get_object_orientation = lambda current_obs: np.asarray(
                    current_obs[f"{object_id}_quat"], dtype=np.float32
                )

        position_getter = get_object_position or self.controller.get_cube_position

        # --------------------------------------------------------
        # 1. Read initial object state
        # --------------------------------------------------------

        initial_cube_pos = position_getter(obs).copy()

        print(
            "Cube position:",
            initial_cube_pos,
        )

        # --------------------------------------------------------
        # 2. Determine expert orientation
        # --------------------------------------------------------

        if self.target_orientation is None:

            target_orientation = self.get_target_orientation(
                obs,
                get_object_orientation=get_object_orientation,
            )

        else:

            target_orientation = np.asarray(
                self.target_orientation,
                dtype=np.float32,
            )

        print(
            "Pick orientation:",
            target_orientation,
        )

        # --------------------------------------------------------
        # 3. Grasp and lift
        # --------------------------------------------------------

        primitive_result = grasp_object(
            obs=obs,
            controller=self.controller,
            object_position=initial_cube_pos,
            target_orientation=target_orientation,
            get_object_position=position_getter,
            approach_height=self.approach_height,
            pre_grasp_height=self.pre_grasp_height,
            grasp_offset=self.grasp_offset,
            lift_height=self.lift_height,
            render=render,
        )

        obs = primitive_result.obs

        if not primitive_result.success:

            failure_labels = {
                "approach": "approach",
                "pre_grasp": "pre-grasp",
                "grasp_pose": "grasp pose",
                "lift": "lift movement",
            }

            print(
                "PickExpert failed:",
                failure_labels[primitive_result.stage],
            )

            return obs, self.failure_info(
                stage=primitive_result.stage,
                initial_cube_pos=initial_cube_pos,
                obs=obs,
                get_object_position=position_getter,
            )

        # --------------------------------------------------------
        # 4. Verify grasp
        # --------------------------------------------------------

        info = self.verify_pick(
            obs,
            initial_cube_pos,
            get_object_position=position_getter,
        )

        if object_name is not None:
            info["object_name"] = object_name

        return obs, info

    # ============================================================
    # SUCCESS VERIFICATION
    # ============================================================

    def verify_pick(
        self,
        obs,
        initial_cube_pos,
        get_object_position=None,
        minimum_lift=0.08,
        max_gripper_distance=0.10,
    ):

        position_getter = get_object_position or self.controller.get_cube_position
        final_cube_pos = position_getter(obs).copy()

        eef_pos = self.controller.get_eef_position(obs).copy()

        lift_distance = final_cube_pos[2] - initial_cube_pos[2]

        gripper_distance = np.linalg.norm(final_cube_pos - eef_pos)

        success = (
            lift_distance >= minimum_lift and gripper_distance <= max_gripper_distance
        )

        print(
            "Final cube position:",
            final_cube_pos,
        )

        print(
            "Lift distance:",
            lift_distance,
        )

        print(
            "Cube-EFF distance:",
            gripper_distance,
        )

        print(
            "Pick success:",
            success,
        )

        return {
            "success": bool(success),
            "stage": "complete",
            "initial_cube_pos": (initial_cube_pos.copy()),
            "final_cube_pos": (final_cube_pos.copy()),
            "lift_distance": float(lift_distance),
            "gripper_distance": float(gripper_distance),
        }

    # ============================================================
    # FAILURE INFO
    # ============================================================

    def failure_info(
        self,
        stage,
        initial_cube_pos,
        obs,
        get_object_position=None,
    ):

        position_getter = get_object_position or self.controller.get_cube_position
        final_cube_pos = position_getter(obs).copy()

        return {
            "success": False,
            "stage": stage,
            "initial_cube_pos": (initial_cube_pos.copy()),
            "final_cube_pos": (final_cube_pos.copy()),
        }
