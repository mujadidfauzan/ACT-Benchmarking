import numpy as np
import robosuite as suite


def main():
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,  # 20 Hz control frequency
        horizon=500,  # 500 timesteps per episode
    )

    obs = env.reset()

    print("=" * 60)
    print("Environment created successfully!")
    print("=" * 60)

    print("\nAction Dimensions:", env.action_dim)
    print("\nActions Specifications:", env.action_spec)

    print("\nObservation Keys:")

    for key, value in obs.items():
        if hasattr(value, "shape"):
            print(f"{key:35s}: {value.shape}")
        else:
            print(f"{key:35s}: {type(value)}")

    for _ in range(300):
        action = np.zeros(env.action_dim)
        action[0] = 0.1
        obs, reward, done, info = env.step(action)

        env.render()

    env.close()


if __name__ == "__main__":
    main()
