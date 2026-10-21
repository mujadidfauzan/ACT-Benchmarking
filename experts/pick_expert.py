"""Scripted Pick state machine for demonstration collection and benchmarking."""

import numpy as np

from experts.grasp_expert import GraspExpert
from environments.pick_config import ROTATION_MODES


class PickExpert(GraspExpert):
    def __init__(self, controller, rotation_mode="align_yaw", **kwargs):
        super().__init__(controller, **kwargs)
        if rotation_mode not in ROTATION_MODES:
            raise ValueError(f"rotation_mode must be one of {ROTATION_MODES}")
        self.rotation_mode = rotation_mode

    def run(self, obs, render=False, object_name=None, **kwargs):
        env = self.controller.env
        object_id = env.resolve_object_id(object_name or env.target_object_id)
        position = lambda value: np.asarray(value[f"{object_id}_pos"])
        orientation = lambda value: np.asarray(value[f"{object_id}_quat"])
        initial_z = float(position(obs)[2])
        target_quat = self.get_target_orientation(obs, orientation) if self.rotation_mode == "align_yaw" else None
        stage, stage_steps, held = "approach", 0, 0
        grasp_offset = None
        gains = {"approach": (8., .8), "pre_grasp": (5., .4), "descend": (4., .3),
                 "close": (2., .15), "lift": (6., .5)}
        for step in range(500):
            cube = position(obs)
            eef = self.controller.get_eef_position(obs)
            aligned = target_quat is None or self.controller.orientation_distance(obs, target_quat) < np.deg2rad(3)
            grip = self.controller.GRIPPER_OPEN
            if stage == "approach":
                target = cube + np.array([0, 0, .12])
                if np.linalg.norm(eef - target) < .03:
                    stage, stage_steps = "pre_grasp", 0
            elif stage == "pre_grasp":
                target = cube + np.array([0, 0, .08])
                if np.linalg.norm(eef[:2] - target[:2]) < .012 and abs(eef[2] - target[2]) < .020 and aligned:
                    stage, stage_steps = "descend", 0
            elif stage == "descend":
                target = cube + np.array([0, 0, .005])
                if np.linalg.norm(eef[:2] - target[:2]) < .008 and abs(eef[2] - target[2]) < .012 and aligned:
                    stage, stage_steps = "close", 0
            elif stage == "close":
                target = eef.copy()
                grip = self.controller.GRIPPER_CLOSE
                if stage_steps >= 15:
                    stage, stage_steps = "lift", 0
                    grasp_offset = float(eef[2] - cube[2])
            if stage == "lift":
                grip = self.controller.GRIPPER_CLOSE
                target = np.array([cube[0], cube[1], initial_z + .20 + grasp_offset])
            gain, limit = gains[stage]
            action = self.controller.create_action(
                delta_position=np.clip((target - eef) * gain, -limit, limit),
                delta_orientation=(np.zeros(3) if target_quat is None else np.clip(
                    self.controller.orientation_gain * self.controller.orientation_error_vector(obs, target_quat), -.5, .5)),
                gripper=grip,
            )
            self.controller.set_phase(stage)
            obs, _, done, _ = env.step(action)
            stage_steps += 1
            if render:
                env.render()
            lift = float(position(obs)[2] - initial_z)
            held = held + 1 if stage == "lift" and lift >= .18 and env._check_success() else 0
            if held >= 20:
                return obs, {"success": True, "stage": "complete", "steps": step + 1,
                             "lift_height": lift, "object_name": object_id}
            if done:
                break
        return obs, {"success": False, "stage": stage, "steps": step + 1,
                     "lift_height": float(position(obs)[2] - initial_z), "object_name": object_id}
