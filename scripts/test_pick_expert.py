import argparse

from robosuite import load_composite_controller_config

from controllers.panda_controller import PandaController
from environments.pick_env import PickEnv
from experts.pick_expert import PickExpert


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-object PickExpert")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-objects", type=int, choices=(3, 4), default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_composite_controller_config(controller="BASIC")
    env = PickEnv(
        robots="Panda",
        controller_configs=config,
        num_objects=args.num_objects,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=1200,
        hard_reset=False,
        seed=args.seed,
    )
    expert = PickExpert(PandaController(env))
    success_count = 0
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            metadata = env.get_task_metadata()
            print("Instruction:", metadata["instruction"])
            print("Target ID:", metadata["target_object"])
            obs, info = expert.run(
                obs,
                object_name=metadata["target_object"],
                render=args.render,
            )
            success_count += int(info["success"])
            print(episode + 1, info["success"], info["stage"])
    finally:
        env.close()

    print(f"Success rate: {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()
