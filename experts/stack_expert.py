import numpy as np

from experts.primitives import release_object, transport_object


class StackExpert:
    """Build a color-ordered cube tower from bottom to top."""

    def __init__(
        self,
        controller,
        pick_expert,
        approach_height=0.18,
        pre_place_height=0.10,
        retreat_height=0.12,
        settle_steps=60,
        xy_tolerance=None,
        z_tolerance=None,
        release_distance=0.08,
    ):
        self.controller = controller
        self.pick_expert = pick_expert
        self.env = controller.env
        self.approach_height = approach_height
        self.pre_place_height = pre_place_height
        self.retreat_height = retreat_height
        self.settle_steps = settle_steps
        self.xy_tolerance = (
            self.env.stack_xy_tolerance
            if xy_tolerance is None
            else float(xy_tolerance)
        )
        self.z_tolerance = (
            self.env.stack_z_tolerance
            if z_tolerance is None
            else float(z_tolerance)
        )
        self.release_distance = release_distance

    def _position_getter(self, identifier):
        object_id = self.env.resolve_object_id(identifier)
        return lambda obs: np.asarray(
            obs[f"{object_id}_pos"],
            dtype=np.float32,
        )

    def _orientation_getter(self, identifier):
        object_id = self.env.resolve_object_id(identifier)
        return lambda obs: np.asarray(
            obs[f"{object_id}_quat"],
            dtype=np.float32,
        )

    def _settle(self, obs, render):
        action = self.controller.create_action(
            gripper=self.controller.GRIPPER_OPEN,
        )
        for _ in range(self.settle_steps):
            obs, _, _, _ = self.env.step(action)
            if render:
                self.env.render()
        return obs

    def run(self, obs, render=False):
        stack_order = self.env.stack_order.copy()
        if len(stack_order) < 2:
            raise ValueError("StackExpert requires at least two objects")

        print()
        print("=" * 70)
        print("STACK EXPERT")
        print("=" * 70)
        print("Stack order (bottom -> top):", stack_order)

        completed_order = [stack_order[0]]

        for level, (support, top) in enumerate(
            zip(stack_order[:-1], stack_order[1:]),
            start=1,
        ):
            print()
            print(f"Level {level}: {top} on {support}")

            top_position_getter = self._position_getter(top)
            top_orientation_getter = self._orientation_getter(top)
            obs, pick_info = self.pick_expert.run(
                obs=obs,
                render=render,
                object_name=top,
                get_object_position=top_position_getter,
                get_object_orientation=top_orientation_getter,
            )
            if not pick_info["success"]:
                return obs, self._failure_info(
                    stage="pick",
                    completed_order=completed_order,
                    support=support,
                    top=top,
                    detail=pick_info,
                )

            support_position = self.env.get_object_position(support)
            desired_top_position = support_position.copy()
            desired_top_position[2] += self.env.cube_height

            current_top_position = self.env.get_object_position(top)
            eef_position = self.controller.get_eef_position(obs).copy()
            grasp_offset_z = float(eef_position[2] - current_top_position[2])
            transport_orientation = self.controller.get_eef_orientation(obs).copy()

            print("Desired cube center:", desired_top_position)
            print("EEF-to-cube Z offset:", grasp_offset_z)

            transport_result = transport_object(
                obs=obs,
                controller=self.controller,
                destination_position=desired_top_position,
                target_orientation=transport_orientation,
                approach_height=self.approach_height,
                pre_place_height=self.pre_place_height,
                place_offset=grasp_offset_z,
                render=render,
            )
            obs = transport_result.obs
            if not transport_result.success:
                return obs, self._failure_info(
                    stage=transport_result.stage,
                    completed_order=completed_order,
                    support=support,
                    top=top,
                )

            release_result = release_object(
                obs=obs,
                controller=self.controller,
                target_orientation=transport_orientation,
                retreat_height=self.retreat_height,
                render=render,
            )
            obs = self._settle(release_result.obs, render=render)
            if not release_result.success:
                return obs, self._failure_info(
                    stage="retreat",
                    completed_order=completed_order,
                    support=support,
                    top=top,
                )

            completed_order.append(top)
            verification = self.verify_stack(obs, completed_order)
            if not verification["success"]:
                verification.update(
                    {
                        "stage": "verify_stack",
                        "support": support,
                        "top": top,
                    }
                )
                return obs, verification

        result = self.verify_stack(obs, stack_order)
        result["stage"] = "complete" if result["success"] else "verify_stack"
        return obs, result

    def verify_stack(self, obs, stack_order=None):
        order = list(stack_order or self.env.stack_order)
        eef_position = self.controller.get_eef_position(obs).copy()
        errors = []

        bottom_position = self.env.get_object_position(order[0])
        expected_bottom_z = self.env.table_offset[2] + self.env.cube_half_size
        bottom_z_error = float(abs(bottom_position[2] - expected_bottom_z))

        for support, top in zip(order[:-1], order[1:]):
            support_position = self.env.get_object_position(support)
            top_position = self.env.get_object_position(top)
            xy_error = float(
                np.linalg.norm(top_position[:2] - support_position[:2])
            )
            z_error = float(
                abs(
                    (top_position[2] - support_position[2])
                    - self.env.cube_height
                )
            )
            gripper_distance = float(np.linalg.norm(top_position - eef_position))
            errors.append(
                {
                    "support": support,
                    "top": top,
                    "xy_error": xy_error,
                    "z_error": z_error,
                    "gripper_distance": gripper_distance,
                    "success": bool(
                        xy_error <= self.xy_tolerance
                        and z_error <= self.z_tolerance
                        and gripper_distance > self.release_distance
                    ),
                }
            )

        success = bottom_z_error <= self.z_tolerance and all(
            item["success"] for item in errors
        )

        print("Stack verification:")
        print("  bottom Z error:", bottom_z_error)
        for item in errors:
            print(
                f"  {item['top']} on {item['support']}:",
                f"xy={item['xy_error']:.4f}",
                f"z={item['z_error']:.4f}",
                f"released={item['gripper_distance'] > self.release_distance}",
            )
        print("  success:", success)

        return {
            "success": bool(success),
            "stage": "complete",
            "stack_order": order,
            "bottom_z_error": bottom_z_error,
            "level_errors": errors,
        }

    def _failure_info(
        self,
        stage,
        completed_order,
        support,
        top,
        detail=None,
    ):
        return {
            "success": False,
            "stage": stage,
            "stack_order": self.env.stack_order.copy(),
            "completed_order": completed_order.copy(),
            "support": support,
            "top": top,
            "detail": detail,
        }
