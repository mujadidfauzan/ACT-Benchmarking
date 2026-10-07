import numpy as np
from robosuite.utils import transform_utils as T
from robosuite.utils.control_utils import orientation_error


class PandaController:

    GRIPPER_OPEN = -1.0
    GRIPPER_CLOSE = 1.0

    def __init__(
        self,
        env,
        position_gain=10.0,
        orientation_gain=2.0,
        position_tolerance=0.005,
        orientation_tolerance=np.deg2rad(1.0),
    ):
        self.env = env

        self.position_gain = position_gain
        self.orientation_gain = orientation_gain

        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance

        self.eef_pos_key = "robot0_eef_pos"
        self.eef_quat_key = "robot0_eef_quat"
        self.gripper_qpos_key = "robot0_gripper_qpos"
        self.current_phase = "unassigned"

    def set_phase(self, phase):
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        self.current_phase = phase

    def get_phase(self):
        return self.current_phase

    # ============================================================
    # STATE GETTERS
    # ============================================================

    def get_eef_position(self, obs):
        return np.asarray(
            obs[self.eef_pos_key],
            dtype=np.float32,
        )

    def get_eef_orientation(self, obs):
        """
        Returns EEF quaternion in robosuite convention:
        [x, y, z, w]
        """
        return np.asarray(
            obs[self.eef_quat_key],
            dtype=np.float32,
        )

    def get_eef_rotation_matrix(self, obs):
        quat = self.get_eef_orientation(obs)

        return T.quat2mat(quat)

    def get_cube_position(self, obs):

        possible_keys = [
            "cube_pos",
            "Cube_pos",
        ]

        for key in possible_keys:
            if key in obs:
                return np.asarray(
                    obs[key],
                    dtype=np.float32,
                )

        raise KeyError(
            "Cube position not found.\n" f"Available keys: {list(obs.keys())}"
        )

    def get_cube_orientation(self, obs):

        possible_keys = [
            "cube_quat",
            "Cube_quat",
        ]

        for key in possible_keys:
            if key in obs:
                return np.asarray(
                    obs[key],
                    dtype=np.float32,
                )

        raise KeyError(
            "Cube orientation not found.\n" f"Available keys: {list(obs.keys())}"
        )

    # ============================================================
    # ERROR CALCULATION
    # ============================================================

    def position_error(
        self,
        obs,
        target_position,
    ):
        current_position = self.get_eef_position(obs)

        target_position = np.asarray(
            target_position,
            dtype=np.float32,
        )

        return target_position - current_position

    def orientation_error_vector(
        self,
        obs,
        target_orientation,
    ):
        """
        Computes orientation error in world-frame axis-angle-like
        representation expected by OSC_POSE.

        target_orientation must be quaternion [x, y, z, w].
        """

        current_quat = self.get_eef_orientation(obs)

        current_rotation = T.quat2mat(current_quat)

        target_rotation = T.quat2mat(
            np.asarray(
                target_orientation,
                dtype=np.float32,
            )
        )

        return orientation_error(
            target_rotation,
            current_rotation,
        )

    def orientation_distance(
        self,
        obs,
        target_orientation,
    ):
        """
        Returns angular difference in radians.
        """

        current_rotation = self.get_eef_rotation_matrix(obs)

        target_rotation = T.quat2mat(
            np.asarray(
                target_orientation,
                dtype=np.float32,
            )
        )

        relative_rotation = target_rotation @ current_rotation.T

        trace_value = np.trace(relative_rotation)

        cosine = (trace_value - 1.0) / 2.0

        cosine = np.clip(
            cosine,
            -1.0,
            1.0,
        )

        return float(np.arccos(cosine))

    # ============================================================
    # ACTION CREATION
    # ============================================================

    def create_action(
        self,
        delta_position=None,
        delta_orientation=None,
        gripper=GRIPPER_OPEN,
    ):

        action = np.zeros(
            self.env.action_dim,
            dtype=np.float32,
        )

        if delta_position is not None:

            delta_position = np.asarray(
                delta_position,
                dtype=np.float32,
            )

            action[:3] = np.clip(
                delta_position,
                -1.0,
                1.0,
            )

        if delta_orientation is not None:

            delta_orientation = np.asarray(
                delta_orientation,
                dtype=np.float32,
            )

            action[3:6] = np.clip(
                delta_orientation,
                -1.0,
                1.0,
            )

        action[-1] = gripper

        return action

    # ============================================================
    # MOVEMENT
    # ============================================================

    def move_to(
        self,
        obs,
        target,
        max_steps=200,
        render=False,
        gripper=GRIPPER_OPEN,
    ):
        """
        Backwards-compatible position-only movement.
        """

        return self.move_to_pose(
            obs=obs,
            target_position=target,
            target_orientation=None,
            max_steps=max_steps,
            render=render,
            gripper=gripper,
        )

    def move_to_pose(
        self,
        obs,
        target_position,
        target_orientation=None,
        max_steps=500,
        render=False,
        gripper=GRIPPER_OPEN,
    ):

        target_position = np.asarray(
            target_position,
            dtype=np.float32,
        )

        if target_orientation is not None:
            target_orientation = np.asarray(
                target_orientation,
                dtype=np.float32,
            )

        for _ in range(max_steps):

            # --------------------------------------------
            # Position control
            # --------------------------------------------

            pos_error = self.position_error(
                obs,
                target_position,
            )

            position_distance = np.linalg.norm(pos_error)

            position_command = self.position_gain * pos_error

            position_command = np.clip(
                position_command,
                -1.0,
                1.0,
            )

            # --------------------------------------------
            # Orientation control
            # --------------------------------------------

            orientation_command = np.zeros(
                3,
                dtype=np.float32,
            )

            orientation_distance = 0.0

            if target_orientation is not None:

                ori_error = self.orientation_error_vector(
                    obs,
                    target_orientation,
                )

                orientation_distance = self.orientation_distance(
                    obs,
                    target_orientation,
                )

                orientation_command = self.orientation_gain * ori_error

                orientation_command = np.clip(
                    orientation_command,
                    -1.0,
                    1.0,
                )

            # --------------------------------------------
            # Check convergence
            # --------------------------------------------

            position_ok = position_distance < self.position_tolerance

            orientation_ok = (
                target_orientation is None
                or orientation_distance < self.orientation_tolerance
            )

            if position_ok and orientation_ok:

                return obs, True

            # --------------------------------------------
            # Send OSC action
            # --------------------------------------------

            action = self.create_action(
                delta_position=position_command,
                delta_orientation=orientation_command,
                gripper=gripper,
            )

            obs, reward, done, info = self.env.step(action)

            if render:
                self.env.render()

        return obs, False

    # ============================================================
    # GRIPPER
    # ============================================================

    def hold_position(
        self,
        obs,
        gripper,
        steps=50,
        render=False,
    ):

        for _ in range(steps):

            action = self.create_action(
                delta_position=np.zeros(3),
                delta_orientation=np.zeros(3),
                gripper=gripper,
            )

            obs, reward, done, info = self.env.step(action)

            if render:
                self.env.render()

        return obs

    def open_gripper(
        self,
        obs,
        steps=20,
        render=False,
    ):

        return self.hold_position(
            obs,
            gripper=self.GRIPPER_OPEN,
            steps=steps,
            render=render,
        )

    def close_gripper(
        self,
        obs,
        steps=40,
        render=False,
    ):

        return self.hold_position(
            obs,
            gripper=self.GRIPPER_CLOSE,
            steps=steps,
            render=render,
        )

    def close_gripper_adaptive(
        self,
        obs,
        min_steps=12,
        max_steps=40,
        stable_steps=3,
        qpos_tolerance=1e-4,
        render=False,
    ):
        """Close until finger motion settles, bounded by safe step limits."""

        if min_steps <= 0:
            raise ValueError("min_steps must be positive")
        if max_steps < min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps")
        if stable_steps <= 0:
            raise ValueError("stable_steps must be positive")
        if qpos_tolerance <= 0:
            raise ValueError("qpos_tolerance must be positive")
        if self.gripper_qpos_key not in obs:
            raise KeyError(
                f"Missing gripper state {self.gripper_qpos_key!r}; "
                f"available keys: {list(obs.keys())}"
            )

        previous_qpos = np.asarray(
            obs[self.gripper_qpos_key], dtype=np.float32
        ).copy()
        consecutive_stable_steps = 0
        steps_taken = 0

        for steps_taken in range(1, max_steps + 1):
            action = self.create_action(
                delta_position=np.zeros(3),
                delta_orientation=np.zeros(3),
                gripper=self.GRIPPER_CLOSE,
            )
            obs, reward, done, info = self.env.step(action)

            if render:
                self.env.render()

            current_qpos = np.asarray(
                obs[self.gripper_qpos_key], dtype=np.float32
            )
            qpos_change = float(np.max(np.abs(current_qpos - previous_qpos)))
            if qpos_change <= qpos_tolerance:
                consecutive_stable_steps += 1
            else:
                consecutive_stable_steps = 0

            previous_qpos = current_qpos.copy()
            if (
                steps_taken >= min_steps
                and consecutive_stable_steps >= stable_steps
            ):
                break

        print(f"Adaptive gripper close: {steps_taken} steps")
        return obs
