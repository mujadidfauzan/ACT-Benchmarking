from experts.primitives.result import PrimitiveResult


def release_object(
    obs,
    controller,
    target_orientation,
    retreat_height=0.15,
    open_steps=50,
    render=False,
    release_ready=None,
    min_open_steps=5,
    ready_hold_steps=3,
):
    """Release and retreat; an optional predicate enables bounded adaptive opening.

    Without a predicate, retain the original fixed-duration behavior for Place.
    With one, retreat requires consecutive ready observations after min_open_steps.
    """

    print("Opening gripper...")

    controller.set_phase("release")
    if release_ready is None:
        obs = controller.open_gripper(obs, steps=open_steps, render=render)
    else:
        if not 0 < min_open_steps <= open_steps or ready_hold_steps <= 0:
            raise ValueError("Invalid adaptive release step bounds")
        consecutive_ready = 0
        for step in range(1, open_steps + 1):
            action = controller.create_action(gripper=controller.GRIPPER_OPEN)
            obs, _, done, _ = controller.env.step(action)
            if render:
                controller.env.render()
            consecutive_ready = consecutive_ready + 1 if release_ready(obs) else 0
            if done:
                return PrimitiveResult(obs=obs, success=False, stage="release")
            if step >= min_open_steps and consecutive_ready >= ready_hold_steps:
                print(f"Adaptive gripper release: {step} steps")
                break
        else:
            return PrimitiveResult(obs=obs, success=False, stage="release")

    retreat_target = controller.get_eef_position(obs).copy()
    retreat_target[2] += retreat_height
    print("Retreat target:", retreat_target)

    controller.set_phase("retreat")
    obs, success = controller.move_to_pose(
        obs,
        target_position=retreat_target,
        target_orientation=target_orientation,
        max_steps=200,
        render=render,
        gripper=controller.GRIPPER_OPEN,
    )

    stage = "complete" if success else "retreat"
    return PrimitiveResult(obs=obs, success=success, stage=stage)
