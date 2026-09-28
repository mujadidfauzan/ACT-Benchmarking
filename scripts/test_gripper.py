import robosuite as suite
from robosuite import (
    load_composite_controller_config,
)

from controllers.panda_controller import (
    PandaController,
)

config = load_composite_controller_config(controller="BASIC")

env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    horizon=1000,
)

obs = env.reset()

controller = PandaController(env)


print("Closing gripper")

obs = controller.close_gripper(
    obs,
    steps=50,
    render=True,
)


print("Opening gripper")

obs = controller.open_gripper(
    obs,
    steps=50,
    render=True,
)


env.close()
