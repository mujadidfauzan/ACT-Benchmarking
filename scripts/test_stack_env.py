import argparse
from itertools import combinations

import numpy as np
from robosuite import load_composite_controller_config
from robosuite.utils import transform_utils as T

from environments.stack_env import StackEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Validate StackEnv resets")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--num-cubes", type=int, choices=(3, 4), default=3)
    parser.add_argument("--resets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def quaternion_to_yaw(quaternion):
    rotation = T.quat2mat(np.asarray(quaternion, dtype=np.float32))
    return float(T.mat2euler(rotation)[2])


def validate_reset(env, obs):
    expected_keys = {
        f"{object_id}_{suffix}"
        for object_id in env.object_ids
        for suffix in ("pos", "quat")
    }
    missing_keys = expected_keys.difference(obs)
    assert not missing_keys, f"Missing observations: {sorted(missing_keys)}"

    metadata = env.get_task_metadata()
    colors = [item["color"] for item in metadata["objects"]]
    assert len(colors) == len(set(colors)) == env.num_cubes
    assert sorted(metadata["stack_order"]) == sorted(colors)

    positions = {}
    for object_id in env.object_ids:
        position = np.asarray(obs[f"{object_id}_pos"])
        quaternion = np.asarray(obs[f"{object_id}_quat"])
        yaw = quaternion_to_yaw(quaternion)

        assert position.shape == (3,)
        assert quaternion.shape == (4,)
        assert np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-4)
        assert env.yaw_range[0] - 1e-3 <= yaw <= env.yaw_range[1] + 1e-3

        cube_radius = env.cubes[0].horizontal_radius
        table_half_size = env.table_full_size[:2] / 2.0
        assert np.all(np.abs(position[:2]) + cube_radius <= table_half_size)
        expected_center_z = env.table_offset[2] + env.cube_half_size
        assert np.isclose(position[2], expected_center_z, atol=1e-3)

        color = env.object_id_to_color[object_id]
        assert env.resolve_object_id(color) == object_id
        assert np.allclose(env.get_object_position(color), position)
        assert np.allclose(env.get_object_orientation(object_id), quaternion)
        for geom_id in env.object_visual_geom_ids[object_id]:
            assert np.allclose(
                env.sim.model.geom_rgba[geom_id],
                env.COLOR_PALETTE[color],
            )

        positions[object_id] = position

    distances = [
        np.linalg.norm(positions[first][:2] - positions[second][:2])
        for first, second in combinations(env.object_ids, 2)
    ]
    assert min(distances) >= env.minimum_object_distance - 1e-3

    return metadata, positions, min(distances)


def main():
    args = parse_args()
    controller_config = load_composite_controller_config(controller="BASIC")

    env = StackEnv(
        robots="Panda",
        controller_configs=controller_config,
        num_cubes=args.num_cubes,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=3000,
        hard_reset=False,
        seed=args.seed,
    )

    position_snapshots = []
    color_assignments = set()
    stack_orders = set()

    print("=" * 80)
    print(f"STACK ENV RANDOM RESET TEST ({args.num_cubes} CUBES)")
    print("=" * 80)

    for reset_index in range(args.resets):
        obs = env.reset()
        zero_action = np.zeros(env.action_dim, dtype=np.float32)

        for _ in range(10):
            obs, _, _, _ = env.step(zero_action)
            if args.render:
                env.render()

        metadata, positions, minimum_distance = validate_reset(env, obs)
        assignment = tuple(
            env.object_id_to_color[object_id] for object_id in env.object_ids
        )
        order = tuple(metadata["stack_order"])

        color_assignments.add(assignment)
        stack_orders.add(order)
        position_snapshots.append(
            np.concatenate([positions[object_id][:2] for object_id in env.object_ids])
        )

        print(f"Reset {reset_index + 1}/{args.resets}")
        print("  colors      :", assignment)
        print("  stack order :", order, "(bottom -> top)")
        print("  min distance:", round(float(minimum_distance), 4), "m")
        for object_id in env.object_ids:
            yaw = np.rad2deg(
                quaternion_to_yaw(obs[f"{object_id}_quat"])
            )
            print(
                f"  {object_id:<7}",
                np.round(positions[object_id], 4),
                f"yaw={yaw:.2f} deg",
            )

    assert len(color_assignments) > 1, "Color assignment did not vary"
    assert len(stack_orders) > 1, "Stack order did not vary"
    assert any(
        not np.allclose(position_snapshots[0], snapshot, atol=1e-4)
        for snapshot in position_snapshots[1:]
    ), "Cube positions did not vary"

    print("=" * 80)
    print(f"{args.resets} RANDOM RESETS PASSED")
    print("=" * 80)

    env.close()


if __name__ == "__main__":
    main()
