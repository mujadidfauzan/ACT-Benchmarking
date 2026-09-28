import numpy as np

from experts.primitives.result import PrimitiveResult


def _target_with_height(position, height):
    target = np.asarray(position, dtype=np.float32).copy()
    target[2] += height
    return target


def grasp_object(
    obs,
    controller,
    object_position,
    target_orientation,
    get_object_position,
    approach_height=0.15,
    pre_grasp_height=0.05,
    grasp_offset=0.0,
    lift_height=0.20,
    open_steps=20,
    close_steps=40,
    render=False,
):
    """Approach, grasp, and lift an object."""

    object_position = np.asarray(object_position, dtype=np.float32).copy()
    target_orientation = np.asarray(target_orientation, dtype=np.float32)

    obs = controller.open_gripper(
        obs,
        steps=open_steps,
        render=render,
    )

    approach_target = _target_with_height(object_position, approach_height)
    print("Approach target:", approach_target)

    obs, success = controller.move_to_pose(
        obs,
        target_position=approach_target,
        target_orientation=target_orientation,
        max_steps=200,
        render=render,
        gripper=controller.GRIPPER_OPEN,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="approach")

    pre_grasp_target = _target_with_height(object_position, pre_grasp_height)
    print("Pre-grasp target:", pre_grasp_target)

    obs, success = controller.move_to_pose(
        obs,
        target_position=pre_grasp_target,
        target_orientation=target_orientation,
        max_steps=150,
        render=render,
        gripper=controller.GRIPPER_OPEN,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="pre_grasp")

    grasp_target = _target_with_height(object_position, grasp_offset)
    print("Grasp target:", grasp_target)

    obs, success = controller.move_to_pose(
        obs,
        target_position=grasp_target,
        target_orientation=target_orientation,
        max_steps=150,
        render=render,
        gripper=controller.GRIPPER_OPEN,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="grasp_pose")

    obs = controller.close_gripper(
        obs,
        steps=close_steps,
        render=render,
    )

    current_object_position = np.asarray(
        get_object_position(obs),
        dtype=np.float32,
    )
    lift_target = _target_with_height(current_object_position, lift_height)
    print("Lift target:", lift_target)

    obs, success = controller.move_to_pose(
        obs,
        target_position=lift_target,
        target_orientation=target_orientation,
        max_steps=200,
        render=render,
        gripper=controller.GRIPPER_CLOSE,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="lift")

    return PrimitiveResult(obs=obs, success=True, stage="complete")
