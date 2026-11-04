import numpy as np

from experts.primitives import release_object, transport_object


class StackExpert:
    """Build a color-ordered cube tower from bottom to top."""

    TIMING_RECIPE = "adaptive_release_settle_v1"

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
        self.settle_steps = settle_steps  # Upper bound; stop after stable success hold.
        self.release_min_steps = 5
        self.release_max_steps = 50
        self.release_ready_steps = 3
        self.release_min_aperture = .06
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
        if (not np.isclose(self.xy_tolerance, self.env.stack_xy_tolerance)
                or not np.isclose(self.z_tolerance, self.env.stack_z_tolerance)
                or not np.isclose(self.release_distance, self.env.release_distance)):
            raise ValueError("Expert verification tolerances must match StackEnv")
        if self.settle_steps < self.env.success_hold_steps:
            raise ValueError("settle_steps must cover the environment's stable success hold")

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

    def timing_config(self):
        return dict(recipe=self.TIMING_RECIPE, release_min_steps=self.release_min_steps,
                    release_max_steps=self.release_max_steps,
                    release_ready_steps=self.release_ready_steps,
                    release_min_aperture_m=self.release_min_aperture,
                    settle_max_steps=self.settle_steps,
                    stable_success_steps=self.env.success_hold_steps)

    def _release_ready(self, obs):
        qpos = np.asarray(obs[self.controller.gripper_qpos_key])
        if qpos.shape != (2,) or not np.isfinite(qpos).all():
            return False
        aperture = abs(float(qpos[0] - qpos[1]))
        robot = self.env.robots[0]
        gripper = robot.gripper[robot.arms[0]]
        return bool(aperture >= self.release_min_aperture and not any(
            self.env.check_contact(gripper, cube) for cube in self.env.cubes))

    def _settle(self, obs, render, order):
        self.controller.set_phase("settle")
        action = self.controller.create_action(
            gripper=self.controller.GRIPPER_OPEN,
        )
        self._stable_steps = 0
        for _ in range(self.settle_steps):
            obs, _, done, _ = self.env.step(action)
            self._stable_steps = (self._stable_steps + 1
                                  if self.env.stack_status(order)["success"] else 0)
            if render:
                self.env.render()
            if self._stable_steps >= self.env.success_hold_steps or done:
                break
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
                release_ready=self._release_ready,
                min_open_steps=self.release_min_steps,
                open_steps=self.release_max_steps,
                ready_hold_steps=self.release_ready_steps,
            )
            obs = release_result.obs
            if not release_result.success:
                return obs, self._failure_info(
                    stage=release_result.stage,
                    completed_order=completed_order,
                    support=support,
                    top=top,
                )

            obs = self._settle(obs, render=render, order=completed_order + [top])
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
        order = list(self.env.stack_order if stack_order is None else stack_order)
        status = self.env.stack_status(order)
        held = getattr(self, "_stable_steps", 0)
        success = status["success"] and held >= self.env.success_hold_steps
        if len(order) == self.env.num_cubes:
            success = success and self.env._check_success()
        return {**status, "success": bool(success), "stage": "complete",
                "stable_success_steps": held, "required_stable_steps": self.env.success_hold_steps}

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
