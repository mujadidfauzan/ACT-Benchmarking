"""Shared training / rollout state contract; no images or expert labels."""

import numpy as np


STATE_FIELDS = (
    ("robot0_eef_pos", 3),
    ("robot0_eef_quat", 4),
    ("robot0_gripper_qpos", 2),
    ("object_0_pos", 3),
    ("object_0_quat", 4),
    ("receptacle_0_pos", 3),
    ("receptacle_0_quat", 4),
)
STATE_DIM = sum(dim for _, dim in STATE_FIELDS)


def pack_state(observation, prefix=""):
    """Concatenate raw world-frame poses, xyzw quaternions and finger qpos.

    Accept either a simulator observation or arrays loaded from HDF5. Prefix
    dimensions must agree; an episode is [T+1,23], one rollout state is [23].
    """
    parts = []
    for name, dim in STATE_FIELDS:
        key = prefix + name
        if key not in observation:
            raise ValueError(f"Missing low-dim observation: {key}")
        value = np.asarray(observation[key], dtype=np.float32)
        if value.ndim < 1 or value.shape[-1] != dim:
            raise ValueError(f"{key}: expected final dimension {dim}, got {value.shape}")
        if parts and value.shape[:-1] != parts[0].shape[:-1]:
            raise ValueError(f"{key}: observation timestep dimensions differ")
        if not np.isfinite(value).all():
            raise ValueError(f"{key}: nonfinite state")
        if name.endswith("_quat") and not np.allclose(np.linalg.norm(value, axis=-1), 1., atol=1e-3):
            raise ValueError(f"{key}: quaternion must have unit norm")
        parts.append(value)
    return np.concatenate(parts, axis=-1)


def validate_scene(config):
    """Fixed IDs are only meaningful for the single-cube single-tray task."""
    if (config.get("tasks") != ["place"] or config.get("scene_counts") != [[1, 1]]
            or config.get("fixed_target_color") is None):
        raise ValueError("Low-dim BC-RNN requires fixed-target Place with 1 cube and 1 tray")
