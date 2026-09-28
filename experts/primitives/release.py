from experts.primitives.result import PrimitiveResult


def release_object(
    obs,
    controller,
    target_orientation,
    retreat_height=0.15,
    open_steps=50,
    render=False,
):
    """Release a grasped object and retreat vertically."""

    print("Opening gripper...")

    obs = controller.open_gripper(
        obs,
        steps=open_steps,
        render=render,
    )

    retreat_target = controller.get_eef_position(obs).copy()
    retreat_target[2] += retreat_height
    print("Retreat target:", retreat_target)

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
