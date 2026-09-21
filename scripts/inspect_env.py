import robosuite as suite


def main():

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )

    obs = env.reset()

    print("=" * 60)
    print("GENERAL")
    print("=" * 60)

    print("Environment:", type(env).__name__)
    print("Action dim :", env.action_dim)
    print("Action spec:")
    print(env.action_spec)

    print()

    print("=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)

    for key, value in obs.items():

        if hasattr(value, "shape"):

            print(f"{key:35s}" f"shape={str(value.shape):20s}" f"dtype={value.dtype}")

        else:

            print(f"{key:35s}" f"type={type(value)}")

    env.close()


if __name__ == "__main__":
    main()
