"""Save camera observations outside the source tree root."""

import argparse
from pathlib import Path

import cv2
import robosuite as suite
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"


def main():
    parser = argparse.ArgumentParser(description="Inspect and save OpenCV-convention RGB cameras")
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/cameras"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
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

    try:
        observation = env.reset()
        for key, value in observation.items():
            if hasattr(value, "shape"):
                print(f"{key:35s} {str(value.shape):20s} {value.dtype}")
            if key.endswith("_image"):
                path = args.output / f"{key}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"Could not write camera image: {path}")
                print(f"Saved: {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
