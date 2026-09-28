import argparse

from robosuite import load_composite_controller_config

from controllers.panda_controller import PandaController
from environments.stack_env import StackEnv
from experts.pick_expert import PickExpert
from experts.stack_expert import StackExpert


def parse_args():
    parser = argparse.ArgumentParser(description="Run StackExpert")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-cubes", type=int, choices=(3, 4), default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


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
        horizon=5000,
        hard_reset=False,
        seed=args.seed,
    )
    controller = PandaController(env=env)
    pick_expert = PickExpert(controller=controller)
    stack_expert = StackExpert(controller=controller, pick_expert=pick_expert)

    success_count = 0
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            print("Task metadata:", env.get_task_metadata())
            obs, info = stack_expert.run(obs=obs, render=args.render)
            success_count += int(info["success"])
            print(
                f"Episode {episode + 1}/{args.episodes}:",
                info["success"],
                info["stage"],
            )
    finally:
        env.close()

    print(f"Success rate: {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()
