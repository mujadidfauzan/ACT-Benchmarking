import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()

print("Action dimension:", env.action_dim)
print("Action spec:", env.action_spec)

ACTION_INDEX = 6

for _ in range(100):

    action = np.zeros(env.action_dim)

    action[ACTION_INDEX] = -0.1

    obs, reward, done, info = env.step(action)

    env.render()

env.close()
