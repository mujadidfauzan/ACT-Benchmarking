import cv2
import robosuite as suite


def main():

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[256, 256],
        camera_widths=[256, 256],
        control_freq=20,
    )

    obs = env.reset()

    print("=" * 60)
    print("Observation keys")
    print("=" * 60)

    for key, value in obs.items():

        if hasattr(value, "shape"):

            print(f"{key:35s}" f"{str(value.shape):20s}" f"{value.dtype}")

    # Find camera key
    image_keys = [key for key in obs.keys() if "image" in key]

    print("\nImage observations:")
    print(image_keys)

    for image_key in image_keys:

        image = obs[image_key]

        print("\nSelected image:", image_key)
        print("Shape:", image.shape)
        print("Min:", image.min())
        print("Max:", image.max())

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        filename = f"{image_key}.png"

        cv2.imwrite(filename, image_bgr)

        print("Saved:", filename)

    env.close()


if __name__ == "__main__":
    main()
