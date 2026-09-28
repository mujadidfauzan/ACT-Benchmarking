import argparse
from itertools import combinations

import numpy as np
from robosuite import load_composite_controller_config

from environments.place_env import PlaceEnv


def main():
    parser = argparse.ArgumentParser(description="Validate multi-object PlaceEnv")
    parser.add_argument("--resets", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = load_composite_controller_config(controller="BASIC")
    env = PlaceEnv(
        robots="Panda",
        controller_configs=config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        hard_reset=False,
        seed=args.seed,
    )
    target_pairs = set()
    try:
        for index in range(args.resets):
            obs = env.reset()
            metadata = env.get_task_metadata()
            pair = (metadata["target_object"], metadata["target_receptacle"])
            target_pairs.add(pair)
            object_colors = [item["color"] for item in metadata["objects"]]
            tray_colors = [item["color"] for item in metadata["receptacles"]]
            assert len(object_colors) == len(set(object_colors)) == env.num_objects
            assert len(tray_colors) == len(set(tray_colors)) == env.num_receptacles
            entity_ids = env.object_ids + env.receptacle_ids
            positions = {entity_id: obs[f"{entity_id}_pos"] for entity_id in entity_ids}
            distances = {
                (a, b): np.linalg.norm(positions[a][:2] - positions[b][:2])
                for a, b in combinations(entity_ids, 2)
            }
            assert all(
                distance >= env.required_clearance(a, b) - 1e-3
                for (a, b), distance in distances.items()
            )
            print(index + 1, metadata["instruction"], pair)
            if args.render:
                env.render()
        assert len(target_pairs) > 1
    finally:
        env.close()
    print(f"{args.resets} PLACE RESETS PASSED")


if __name__ == "__main__":
    main()
