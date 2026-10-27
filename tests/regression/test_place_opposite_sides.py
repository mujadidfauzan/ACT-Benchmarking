"""Side constraints, seeded sampling and collection/evaluation layout fidelity."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from robosuite.models.objects import BoxObject

from environments.place_env import PlaceEnv, TrayObject
from environments.place_layout import OppositeSidesSampler
from scripts import record_demonstrations as recording, evaluate_visual_bc as evaluation
from tests.regression.test_variable_scene_counts import FakeEnv


class OppositeSidesTests(unittest.TestCase):
    @staticmethod
    def sampler(seed=123, count=1):
        cubes = [BoxObject(name=f"object_{i}", size=[.02] * 3) for i in range(count)]
        trays = [TrayObject(name=f"receptacle_{i}", inner_half_size=(.05, .04),
                            wall_thickness=.004, wall_height=.015, base_thickness=.004)
                 for i in range(count)]
        return OppositeSidesSampler(cubes, trays, np.array([0., 0., .8]), np.random.default_rng(seed))

    def test_both_sides_full_boundaries_independent_positions_and_yaws(self):
        sampler = self.sampler()
        samples = [sampler.sample() for _ in range(200)]
        positive_objects = 0
        for p in samples:
            a, b = p["object_0"], p["receptacle_0"]
            self.assertLess(a[0][1] * b[0][1], 0)
            positive_objects += int(a[0][1] > 0)
            for pos, quat, entity in p.values():
                radius = entity.horizontal_radius
                self.assertLessEqual(abs(pos[0]) + radius, .16 + 1e-8)
                self.assertLessEqual(abs(pos[1]) + radius, .30 + 1e-8)
                self.assertGreaterEqual(abs(pos[1]), radius - 1e-8)
                yaw = 2 * np.arctan2(quat[3], quat[0])
                self.assertLessEqual(abs(yaw), np.pi / 4 + 1e-8)
        self.assertGreater(positive_objects, 70)
        self.assertLess(positive_objects, 130)
        for key in ("object_0", "receptacle_0"):
            self.assertGreater(np.ptp([p[key][0][0] for p in samples]), .10)
            self.assertGreater(np.ptp([abs(p[key][0][1]) for p in samples]), .10)
            self.assertGreater(np.ptp([2 * np.arctan2(p[key][1][3], p[key][1][0]) for p in samples]), 1.)
        self.assertFalse(all(np.isclose(p["object_0"][0][0], p["receptacle_0"][0][0]) for p in samples))
        self.assertFalse(all(np.isclose(p["object_0"][1][3], p["receptacle_0"][1][3]) for p in samples))

    def test_seed_reproducibility_and_multi_entity_clearances(self):
        left, right = self.sampler(8, 3), self.sampler(8, 3)
        env = PlaceEnv.__new__(PlaceEnv)
        env.object_ids = [f"object_{i}" for i in range(3)]
        env.receptacle_ids = [f"receptacle_{i}" for i in range(3)]
        env.minimum_object_distance, env.minimum_receptacle_distance = .10, .14
        env.minimum_object_receptacle_distance = .12
        env.placement_initializer = left
        for _ in range(12):
            # Include the env's unchanged additional clearance rejection loop.
            expected = env._sample_safe_placements()
            other = PlaceEnv.__new__(PlaceEnv)
            other.__dict__.update(env.__dict__)
            other.placement_initializer = right
            actual = other._sample_safe_placements()
            for key in expected:
                np.testing.assert_array_equal(expected[key][0], actual[key][0])
                np.testing.assert_array_equal(expected[key][1], actual[key][1])
            sign = np.sign(expected["object_0"][0][1])
            self.assertTrue(all(np.sign(expected[k][0][1]) == sign for k in env.object_ids))
            self.assertTrue(all(np.sign(expected[k][0][1]) == -sign for k in env.receptacle_ids))
            keys = list(expected)
            for i, first in enumerate(keys):
                for second in keys[i + 1:]:
                    self.assertGreaterEqual(np.linalg.norm(np.asarray(expected[first][0][:2]) - expected[second][0][:2]),
                                            env.required_clearance(first, second))

    def test_invalid_layout_and_pick_option_fail_early(self):
        with self.assertRaisesRegex(ValueError, "placement_layout"):
            PlaceEnv(placement_layout="unknown")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "should_not_exist"
            with patch("sys.argv", ["record", "--task", "pick", "--placement-layout", "opposite_sides",
                                    "--output", str(path)]), self.assertRaisesRegex(ValueError, "requires --task place"):
                recording.main()
            self.assertFalse(path.exists())

    def test_collection_metadata_split_eval_and_append_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "demos"
            created = []

            def build(args, objects, receptacles, seed):
                env = FakeEnv("place", objects, receptacles, seed)
                env.placement_layout = args.placement_layout
                created.append(env)
                expert = SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "approach"))
                return env, expert

            def expert(*args):
                obs, _, _, _ = created[-1].step(np.zeros(7, dtype=np.float32))
                return obs, dict(success=True, stage="complete")

            command = ["record", "--task", "place", "--num-objects", "1", "--num-receptacles", "1",
                       "--placement-layout", "opposite_sides", "--num-demos", "2", "--output", str(root)]
            with patch("sys.argv", command), patch.object(recording, "build_task", side_effect=build), \
                    patch.object(recording, "run_expert", side_effect=expert), redirect_stdout(io.StringIO()):
                recording.main()
            entries = [json.loads(l) for l in (root / "place/manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 2)
            for entry in entries:
                self.assertEqual(entry["metadata"]["collection_config"]["environment_kwargs"]["placement_layout"], "opposite_sides")
            prepared = json.loads((recording.default_training_dir(root) / "training.json").read_text())
            model = SimpleNamespace(camera_names=["agentview"])
            # The evaluator's saved environment_kwargs must take priority over
            # the factory's legacy default layout, for all policy families.
            with patch.object(recording, "PlaceEnv", return_value=SimpleNamespace()) as constructor:
                evaluation.build_environment(2000, 1, False, model, prepared, 300)
            self.assertEqual(constructor.call_args.kwargs["placement_layout"], "opposite_sides")
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            changed = list(command)
            changed[changed.index("opposite_sides")] = "uniform"
            with patch("sys.argv", changed), self.assertRaisesRegex(ValueError, "another placement layout"):
                recording.main()
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def test_legacy_checkpoint_factory_still_uses_uniform(self):
        config = dict(tasks=["place"], camera_sizes=[[128, 128]], fixed_target_color="red",
                      scene_counts=[[1, 1]], environment_configs={"place": {}})
        model = SimpleNamespace(camera_names=["agentview"])
        with patch.object(recording, "PlaceEnv", return_value=SimpleNamespace()) as constructor:
            evaluation.build_environment(2000, 1, False, model, config, 300)
        self.assertEqual(constructor.call_args.kwargs["placement_layout"], "uniform")


if __name__ == "__main__":
    unittest.main(verbosity=2)
