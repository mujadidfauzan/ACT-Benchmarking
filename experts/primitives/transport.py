import numpy as np

from experts.primitives.result import PrimitiveResult


def _target_with_height(position, height):
    target = np.asarray(position, dtype=np.float32).copy()
    target[2] += height
    return target


def transport_object(
    obs,
    controller,
    destination_position,
    target_orientation,
    approach_height=0.15,
    pre_place_height=0.15,
    place_offset=0.10,
    render=False,
):
    """Carry a grasped object to a destination and descend to place pose."""

    destination_position = np.asarray(
        destination_position,
        dtype=np.float32,
    ).copy()
    target_orientation = np.asarray(target_orientation, dtype=np.float32)

    approach_target = _target_with_height(destination_position, approach_height)
    print("Approach target:", approach_target)

    controller.set_phase("transport_approach")
    obs, success = controller.move_to_pose(
        obs,
        target_position=approach_target,
        target_orientation=target_orientation,
        max_steps=250,
        render=render,
        gripper=controller.GRIPPER_CLOSE,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="approach")

    pre_place_target = _target_with_height(destination_position, pre_place_height)
    print("Pre-place target:", pre_place_target)

    controller.set_phase("pre_place")
    obs, success = controller.move_to_pose(
        obs,
        target_position=pre_place_target,
        target_orientation=target_orientation,
        max_steps=180,
        render=render,
        gripper=controller.GRIPPER_CLOSE,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="pre_place")

    place_target = _target_with_height(destination_position, place_offset)
    print("Place target:", place_target)

    controller.set_phase("place")
    obs, success = controller.move_to_pose(
        obs,
        target_position=place_target,
        target_orientation=target_orientation,
        max_steps=180,
        render=render,
        gripper=controller.GRIPPER_CLOSE,
    )

    if not success:
        return PrimitiveResult(obs=obs, success=False, stage="place_pose")

    return PrimitiveResult(obs=obs, success=True, stage="complete")
