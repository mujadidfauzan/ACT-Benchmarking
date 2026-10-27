"""Initial Place layouts in robot/world coordinates (robot faces +X)."""

import numpy as np

from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler


PLACEMENT_LAYOUTS = ("uniform", "opposite_sides")


class OppositeSidesSampler(SequentialCompositeSampler):
    """Objects and receptacles occupy opposite Y half-planes at episode reset.

    Each sample independently chooses which group uses positive Y. Each entity
    then receives its own random XY and yaw; paired poses are not mirrored.
    Boundary checking keeps each full entity inside its assigned half-table.
    """

    def __init__(self, objects, receptacles, reference_pos, rng):
        super().__init__(name="PlaceOppositeSidesSampler")
        self.rng = rng
        self.objects_sampler = UniformRandomSampler(
            name="PlaceObjects", mujoco_objects=objects, y_range=(-0.30, 0.),
            **self._kwargs(reference_pos, rng),
        )
        self.receptacles_sampler = UniformRandomSampler(
            name="PlaceReceptacles", mujoco_objects=receptacles, y_range=(0., 0.30),
            **self._kwargs(reference_pos, rng),
        )
        self.append_sampler(self.objects_sampler)
        self.append_sampler(self.receptacles_sampler)

    @staticmethod
    def _kwargs(reference_pos, rng):
        return dict(x_range=(-0.16, 0.16), rotation=(-np.pi / 4, np.pi / 4),
                    rotation_axis="z", ensure_object_boundary_in_range=True,
                    ensure_valid_placement=True, reference_pos=reference_pos,
                    z_offset=0.01, rng=rng)

    def sample(self, fixtures=None, reference=None, on_top=True):
        positive_objects = bool(self.rng.integers(2))
        negative, positive = (-0.30, 0.), (0., 0.30)
        self.objects_sampler.y_range = positive if positive_objects else negative
        self.receptacles_sampler.y_range = negative if positive_objects else positive
        return super().sample(fixtures=fixtures, reference=reference, on_top=on_top)
