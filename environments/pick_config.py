"""Pick ablation settings shared by collection and expert benchmarks."""

import argparse

import numpy as np


DEFAULT_YAW_RANGE_DEG = (-45.0, 45.0)
ROTATION_MODES = ("none", "align_yaw")


def add_pick_arguments(parser):
    parser.add_argument("--random-yaw", action=argparse.BooleanOptionalAction, default=True,
                        help="Pick: random cube yaw within +/-45 degrees (default); --no-random-yaw fixes yaw at zero")
    parser.add_argument("--rotation-mode", choices=ROTATION_MODES, default="align_yaw",
                        help="Pick: zero rotational actions, or align gripper yaw to the cube")


def yaw_range_degrees(random_yaw):
    if not isinstance(random_yaw, bool):
        raise ValueError("random_yaw must be a boolean")
    return DEFAULT_YAW_RANGE_DEG if random_yaw else (0.0, 0.0)


def validate_pick_settings(rotation_mode, yaw_range_deg):
    if rotation_mode not in ROTATION_MODES:
        raise ValueError(f"rotation_mode must be one of {ROTATION_MODES}")
    values = np.asarray(yaw_range_deg, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
        raise ValueError("Yaw range must contain two finite values with MIN <= MAX")
    return tuple(float(value) for value in values)


def validate_pick_arguments(args):
    yaw = validate_pick_settings(args.rotation_mode, yaw_range_degrees(args.random_yaw))
    if args.task != "pick" and (args.rotation_mode != "align_yaw" or yaw != DEFAULT_YAW_RANGE_DEG):
        raise ValueError("Custom --rotation-mode and --no-random-yaw require --task pick")
    return yaw
