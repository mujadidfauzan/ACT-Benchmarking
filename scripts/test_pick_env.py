import argparse
from itertools import combinations

import numpy as np
from robosuite import load_composite_controller_config

from environments.pick_env import PickEnv


def main():
    parser = argparse.ArgumentParser(description="Validate multi-object PickEnv")
    parser.add_argument("--resets", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_composite_controller_config(controller="BASIC")
    env = PickEnv(
        robots="Panda",
        controller_configs=config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        hard_reset=False,
        seed=args.seed,
    )
    targets = set()
    try:
        for index in range(args.resets):
            obs = env.reset()
            metadata = env.get_task_metadata()
            targets.add(metadata["target_object"])
            colors = [item["color"] for item in metadata["objects"]]
            assert len(colors) == len(set(colors)) == env.num_objects
            assert metadata["target_object"] in env.object_ids
            assert len(metadata["distractors"]) == env.num_objects - 1
            positions = {
                object_id: obs[f"{object_id}_pos"] for object_id in env.object_ids
            }
            distances = [
                np.linalg.norm(positions[a][:2] - positions[b][:2])
                for a, b in combinations(env.object_ids, 2)
            ]
            assert min(distances) >= env.minimum_object_distance - 1e-3
            print(index + 1, metadata["instruction"], metadata["target_object"])
            if args.render:
                env.render()
        assert len(targets) > 1
    finally:
        env.close()
    print(f"{args.resets} PICK RESETS PASSED")


if __name__ == "__main__":
    main()
