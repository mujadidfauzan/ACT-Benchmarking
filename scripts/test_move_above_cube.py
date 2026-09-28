import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.utils import transform_utils as T
from robosuite.utils.placement_samplers import UniformRandomSampler

from controllers.panda_controller import (
    PandaController,
)


def get_pick_orientation(controller, obs):
    """
    Keep the current EEF roll/pitch and align yaw with the cube.
    Quaternion convention: [x, y, z, w].
    """

    eef_quat = controller.get_eef_orientation(obs)
    cube_quat = controller.get_cube_orientation(obs)

    eef_euler = T.mat2euler(T.quat2mat(eef_quat))
    cube_euler = T.mat2euler(T.quat2mat(cube_quat))

    target_euler = np.array(
        [
            eef_euler[0],
            eef_euler[1],
            cube_euler[2],
        ],
        dtype=np.float32,
    )

    return T.mat2quat(T.euler2mat(target_euler)).astype(np.float32)


def quat_to_euler_deg(quat):
    return np.rad2deg(T.mat2euler(T.quat2mat(quat)))


def print_cube_eef_state(label, controller, obs):
    cube_pos = controller.get_cube_position(obs)
    cube_quat = controller.get_cube_orientation(obs)
    eef_pos = controller.get_eef_position(obs)
    eef_quat = controller.get_eef_orientation(obs)

    print(label)
    print("Cube Pos   :", np.round(cube_pos, 4))
    print("Cube Degree:", np.round(quat_to_euler_deg(cube_quat), 2))
    print("EEF Pos    :", np.round(eef_pos, 4))
    print("EEF Degree :", np.round(quat_to_euler_deg(eef_quat), 2))
    print()


config = load_composite_controller_config(controller="BASIC")

placement_initializer = UniformRandomSampler(
    name="uniform_random",
    x_range=[-0.1, 0.1],
    y_range=[-0.2, 0.2],
    rotation=[-np.pi / 4, np.pi / 4],
    rotation_axis="z",
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=True,
    reference_pos=[0.0, 0, 0.8],
    z_offset=0.01,
)

env = suite.make(
    env_name="Lift",
    robots="Panda",
    placement_initializer=placement_initializer,
    controller_configs=config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    control_freq=20,
    horizon=1000,
)

obs = env.reset()

controller = PandaController(env)

for episode in range(2):

    obs = env.reset()

    print("=" * 70)
    print("Episode:", episode)
    print_cube_eef_state("AWAL", controller, obs)

    cube = controller.get_cube_position(obs)

    target = cube.copy()
    target[2] += 0.15

    target_orientation = get_pick_orientation(
        controller,
        obs,
    )

    obs, success = controller.move_to_pose(
        obs=obs,
        target_position=target,
        target_orientation=target_orientation,
        render=False,
    )

    position_error = np.linalg.norm(controller.get_eef_position(obs) - target)

    orientation_error_deg = np.rad2deg(
        controller.orientation_distance(
            obs,
            target_orientation,
        )
    )

    print(
        episode,
        success,
        position_error,
        orientation_error_deg,
    )

    print_cube_eef_state("AKHIR", controller, obs)

env.close()
