import numpy as np
from robosuite.utils import transform_utils as T

from experts.primitives import release_object, transport_object


class PlaceExpert:
    """Pick an instructed object and place it in an instructed receptacle."""

    def __init__(
        self,
        controller,
        pick_expert,
        approach_height=0.15,
        pre_place_height=0.15,
        place_offset=0.10,
        retreat_height=0.15,
        xy_tolerance=0.04,
        distractor_tolerance=0.04,
    ):
        self.controller = controller
        self.pick_expert = pick_expert
        self.env = controller.env
        self.approach_height = approach_height
        self.pre_place_height = pre_place_height
        self.place_offset = place_offset
        self.retreat_height = retreat_height
        self.xy_tolerance = xy_tolerance
        self.distractor_tolerance = distractor_tolerance

    def _object_position(self, obs, identifier):
        object_id = self.env.resolve_object_id(identifier)
        return np.asarray(obs[f"{object_id}_pos"], dtype=np.float32)

    def _receptacle_position(self, obs, identifier):
        receptacle_id = self.env.resolve_receptacle_id(identifier)
        return np.asarray(obs[f"{receptacle_id}_pos"], dtype=np.float32)

    def _receptacle_orientation(self, obs, identifier):
        receptacle_id = self.env.resolve_receptacle_id(identifier)
        return np.asarray(obs[f"{receptacle_id}_quat"], dtype=np.float32)

    def get_receptacle_position(self, obs, receptacle_name=None):
        if hasattr(self.env, "resolve_receptacle_id"):
            return self._receptacle_position(obs, receptacle_name)
        return np.asarray(obs["receptacle_pos"], dtype=np.float32)

    def get_receptacle_orientation(self, obs, receptacle_name=None):
        if hasattr(self.env, "resolve_receptacle_id"):
            return self._receptacle_orientation(obs, receptacle_name)
        return np.asarray(obs["receptacle_quat"], dtype=np.float32)

    def get_place_orientation(self, obs, receptacle_name=None):
        eef_euler = T.mat2euler(
            T.quat2mat(self.controller.get_eef_orientation(obs).copy())
        )
        receptacle_euler = T.mat2euler(
            T.quat2mat(
                self.get_receptacle_orientation(obs, receptacle_name).copy()
            )
        )
        target_euler = np.array(
            [eef_euler[0], eef_euler[1], receptacle_euler[2]],
            dtype=np.float32,
        )
        target_quat = T.mat2quat(T.euler2mat(target_euler)).astype(np.float32)
        print("Current EEF Euler [r p y]:", np.round(np.rad2deg(eef_euler), 2))
        print(
            "Receptacle Euler [r p y]:",
            np.round(np.rad2deg(receptacle_euler), 2),
        )
        print("Place target Euler [r p y]:", np.round(np.rad2deg(target_euler), 2))
        return target_quat

    def run(
        self,
        obs,
        render=False,
        object_name=None,
        receptacle_name=None,
    ):
        if object_name is None:
            object_name = getattr(self.env, "target_object_id", None)
        if receptacle_name is None:
            receptacle_name = getattr(self.env, "target_receptacle_id", None)

        object_id = self.env.resolve_object_id(object_name)
        receptacle_id = self.env.resolve_receptacle_id(receptacle_name)
        initial_distractor_positions = {
            other_id: self.env.get_object_position(other_id).copy()
            for other_id in self.env.object_ids
            if other_id != object_id
        }

        print()
        print("=" * 70)
        print("PLACE EXPERT")
        print("=" * 70)
        print("Target object:", object_id)
        print("Target receptacle:", receptacle_id)

        obs, pick_info = self.pick_expert.run(
            obs,
            render=render,
            object_name=object_id,
        )
        if not pick_info["success"]:
            return obs, self._failure_info(
                "pick", object_id, receptacle_id, detail=pick_info
            )

        receptacle_position = self._receptacle_position(obs, receptacle_id).copy()
        place_orientation = self.get_place_orientation(obs, receptacle_id)
        transport_result = transport_object(
            obs=obs,
            controller=self.controller,
            destination_position=receptacle_position,
            target_orientation=place_orientation,
            approach_height=self.approach_height,
            pre_place_height=self.pre_place_height,
            place_offset=self.place_offset,
            render=render,
        )
        obs = transport_result.obs
        if not transport_result.success:
            return obs, self._failure_info(
                transport_result.stage, object_id, receptacle_id
            )

        release_result = release_object(
            obs=obs,
            controller=self.controller,
            target_orientation=place_orientation,
            retreat_height=self.retreat_height,
            render=render,
        )
        obs = release_result.obs
        if not release_result.success:
            return obs, self._failure_info("retreat", object_id, receptacle_id)

        return obs, self.verify_place(
            obs,
            object_id,
            receptacle_id,
            initial_distractor_positions,
        )

    def verify_place(
        self,
        obs,
        object_name=None,
        receptacle_name=None,
        initial_distractor_positions=None,
    ):
        object_id = self.env.resolve_object_id(object_name)
        receptacle_id = self.env.resolve_receptacle_id(receptacle_name)
        object_position = self._object_position(obs, object_id)
        receptacle_position = self._receptacle_position(obs, receptacle_id)
        eef_position = self.controller.get_eef_position(obs).copy()

        xy_error = float(
            np.linalg.norm(object_position[:2] - receptacle_position[:2])
        )
        gripper_distance = float(np.linalg.norm(object_position - eef_position))
        other_target_distances = {
            other_id: float(
                np.linalg.norm(
                    object_position[:2]
                    - self._receptacle_position(obs, other_id)[:2]
                )
            )
            for other_id in self.env.receptacle_ids
            if other_id != receptacle_id
        }
        wrong_target = any(
            distance <= self.xy_tolerance
            for distance in other_target_distances.values()
        )

        distractor_displacements = {}
        for other_id, initial_position in (initial_distractor_positions or {}).items():
            displacement = np.linalg.norm(
                self._object_position(obs, other_id) - initial_position
            )
            distractor_displacements[other_id] = float(displacement)
        distractors_stable = all(
            displacement <= self.distractor_tolerance
            for displacement in distractor_displacements.values()
        )

        inside_target = xy_error <= self.xy_tolerance
        released = gripper_distance > 0.08
        success = inside_target and released and not wrong_target and distractors_stable

        print("Place verification:")
        print("  XY error:", xy_error)
        print("  released:", released)
        print("  wrong target:", wrong_target)
        print("  distractors stable:", distractors_stable)
        print("  success:", success)

        return {
            "success": bool(success),
            "stage": "complete" if success else "verify_place",
            "target_object": object_id,
            "target_receptacle": receptacle_id,
            "object_pos": object_position.copy(),
            "receptacle_pos": receptacle_position.copy(),
            "xy_error": xy_error,
            "gripper_distance": gripper_distance,
            "wrong_target": bool(wrong_target),
            "other_target_distances": other_target_distances,
            "distractor_displacements": distractor_displacements,
            "distractors_stable": bool(distractors_stable),
        }

    def _failure_info(
        self,
        stage,
        object_name,
        receptacle_name,
        detail=None,
    ):
        return {
            "success": False,
            "stage": stage,
            "target_object": object_name,
            "target_receptacle": receptacle_name,
            "detail": detail,
        }
