import argparse

from robosuite import load_composite_controller_config

from controllers.panda_controller import PandaController
from environments.place_env import PlaceEnv
from experts.pick_expert import PickExpert
from experts.place_expert import PlaceExpert


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-object PlaceExpert")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_composite_controller_config(controller="BASIC")
    env = PlaceEnv(
        robots="Panda",
        controller_configs=config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        control_freq=20,
        horizon=2500,
        hard_reset=False,
        seed=args.seed,
    )
    controller = PandaController(env)
    expert = PlaceExpert(controller, PickExpert(controller))
    success_count = 0
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            metadata = env.get_task_metadata()
            print("Instruction:", metadata["instruction"])
            obs, info = expert.run(
                obs,
                object_name=metadata["target_object"],
                receptacle_name=metadata["target_receptacle"],
                render=args.render,
            )
            success_count += int(info["success"])
            print(episode + 1, info["success"], info["stage"])
    finally:
        env.close()

    print(f"Success rate: {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()
