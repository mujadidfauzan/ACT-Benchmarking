import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config

from controllers.panda_controller import (
    PandaController,
)

# ============================================================
# QUATERNION UTILITIES
# Convention used here:
# [x, y, z, w]
# ============================================================


def normalize_quat(q):
    q = np.asarray(
        q,
        dtype=np.float32,
    )

    return q / np.linalg.norm(q)


def axis_angle_to_quat(
    axis,
    angle_rad,
):
    """
    Converts axis + angle into quaternion [x, y, z, w].
    """

    axis = np.asarray(
        axis,
        dtype=np.float32,
    )

    axis = axis / np.linalg.norm(axis)

    half_angle = angle_rad / 2.0

    xyz = axis * np.sin(half_angle)

    w = np.cos(half_angle)

    return normalize_quat(
        np.array(
            [
                xyz[0],
                xyz[1],
                xyz[2],
                w,
            ],
            dtype=np.float32,
        )
    )


def quat_multiply(
    q1,
    q2,
):
    """
    Quaternion multiplication for XYZW convention.

    q_result = q1 * q2
    """

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return normalize_quat(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )
    )


def print_pose(
    controller,
    obs,
    label,
):
    position = controller.get_eef_position(obs)

    orientation = controller.get_eef_orientation(obs)

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        "Position:",
        np.round(
            position,
            4,
        ),
    )

    print(
        "Quaternion [x y z w]:",
        np.round(
            orientation,
            4,
        ),
    )


# ============================================================
# MAIN TEST
# ============================================================


def main():

    controller_config = load_composite_controller_config(controller="BASIC")

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=(controller_config),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=2000,
    )

    obs = env.reset()

    controller = PandaController(
        env=env,
        position_gain=10.0,
        # Keep this relatively low initially
        orientation_gain=1.0,
        position_tolerance=0.01,
        orientation_tolerance=(np.deg2rad(5.0)),
    )

    # ========================================================
    # Initial pose
    # ========================================================

    initial_position = controller.get_eef_position(obs).copy()

    initial_orientation = controller.get_eef_orientation(obs).copy()

    print_pose(
        controller,
        obs,
        "INITIAL POSE",
    )

    print(
        "\nInitial orientation norm:",
        np.linalg.norm(initial_orientation),
    )

    # ========================================================
    # TEST 1
    # POSITION ONLY
    # ========================================================

    print()
    print("TEST 1: POSITION ONLY")

    position_target = initial_position.copy()

    position_target[0] += 0.08

    obs, success = controller.move_to_pose(
        obs=obs,
        target_position=(position_target),
        # Explicitly keep initial orientation
        target_orientation=(initial_orientation),
        max_steps=300,
        render=True,
        gripper=(controller.GRIPPER_OPEN),
    )

    print_pose(
        controller,
        obs,
        "AFTER POSITION TEST",
    )

    print(
        "Position test success:",
        success,
    )

    print(
        "Position error:",
        np.linalg.norm(position_target - controller.get_eef_position(obs)),
    )

    print(
        "Orientation error (deg):",
        np.rad2deg(
            controller.orientation_distance(
                obs,
                initial_orientation,
            )
        ),
    )

    input("\nPress Enter to continue to next test...")

    # ========================================================
    # TEST 2
    # ROTATE AROUND WORLD Z
    # ========================================================

    print()
    print("TEST 2: ORIENTATION ONLY")

    current_position = controller.get_eef_position(obs).copy()

    current_orientation = controller.get_eef_orientation(obs).copy()

    # Rotate +30 degrees around world Z
    delta_rotation = axis_angle_to_quat(
        axis=[0.0, 0.0, 1.0],
        angle_rad=np.deg2rad(30.0),
    )

    # World-frame rotation:
    #
    # target = delta * current
    #
    target_orientation = quat_multiply(
        delta_rotation,
        current_orientation,
    )

    print(
        "\nCurrent quaternion:",
        np.round(
            current_orientation,
            4,
        ),
    )

    print(
        "Target quaternion:",
        np.round(
            target_orientation,
            4,
        ),
    )

    print(
        "Requested rotation:",
        "+30 deg around world Z",
    )

    obs, success = controller.move_to_pose(
        obs=obs,
        # Keep XYZ fixed
        target_position=(current_position),
        target_orientation=(target_orientation),
        max_steps=400,
        render=True,
        gripper=(controller.GRIPPER_OPEN),
    )

    print_pose(
        controller,
        obs,
        "AFTER Z ROTATION",
    )

    orientation_error_deg = np.rad2deg(
        controller.orientation_distance(
            obs,
            target_orientation,
        )
    )

    print(
        "Orientation test success:",
        success,
    )

    print(
        "Orientation error (deg):",
        orientation_error_deg,
    )

    input("\nPress Enter to continue to next test...")

    # ========================================================
    # TEST 3
    # POSITION + ORIENTATION
    # ========================================================

    print()
    print("TEST 3: POSITION + ORIENTATION")

    combined_position = controller.get_eef_position(obs).copy()

    combined_position[1] -= 0.3

    # Rotate another 20 degrees around Z
    current_orientation = controller.get_eef_orientation(obs).copy()

    delta_rotation_2 = axis_angle_to_quat(
        axis=[0.0, 0.0, 1.0],
        angle_rad=np.deg2rad(70.0),
    )

    combined_orientation = quat_multiply(
        delta_rotation_2,
        current_orientation,
    )

    obs, success = controller.move_to_pose(
        obs=obs,
        target_position=(combined_position),
        target_orientation=(combined_orientation),
        max_steps=400,
        render=True,
        gripper=(controller.GRIPPER_OPEN),
    )

    print_pose(
        controller,
        obs,
        "AFTER COMBINED TEST",
    )

    print(
        "Combined test success:",
        success,
    )

    print(
        "Position error:",
        np.linalg.norm(combined_position - controller.get_eef_position(obs)),
    )

    print(
        "Orientation error (deg):",
        np.rad2deg(
            controller.orientation_distance(
                obs,
                combined_orientation,
            )
        ),
    )

    input("\nPress Enter to continue to next test...")

    # ========================================================
    # TEST 4
    # RETURN TO INITIAL ORIENTATION
    # ========================================================

    print()
    print("TEST 4: RETURN TO INITIAL ORIENTATION")

    current_position = controller.get_eef_position(obs).copy()

    obs, success = controller.move_to_pose(
        obs=obs,
        target_position=(current_position),
        target_orientation=(initial_orientation),
        max_steps=500,
        render=True,
        gripper=(controller.GRIPPER_OPEN),
    )

    print_pose(
        controller,
        obs,
        "FINAL POSE",
    )

    print(
        "Return orientation success:",
        success,
    )

    print(
        "Final orientation error (deg):",
        np.rad2deg(
            controller.orientation_distance(
                obs,
                initial_orientation,
            )
        ),
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    input("\nPress Enter to close...")

    env.close()


if __name__ == "__main__":
    main()
