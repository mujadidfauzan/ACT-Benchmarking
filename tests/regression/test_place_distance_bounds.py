"""Distance-bounded sampling and collection-to-evaluation configuration fidelity."""

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

from environments.place_env import PlaceEnv
from scripts import record_demonstrations as recording, evaluate_visual_bc as evaluation
from tests.regression import test_place_opposite_sides as opposite
from tests.regression.test_variable_scene_counts import FakeEnv


class PlaceDistanceBoundsTests(unittest.TestCase):
    @staticmethod
    def sampling_env(seed, maximum):
        env = PlaceEnv.__new__(PlaceEnv)
        env.object_ids, env.receptacle_ids = ["object_0"], ["receptacle_0"]
        env.minimum_object_distance, env.minimum_receptacle_distance = .10, .14
        env.minimum_object_receptacle_distance = .12
        env.maximum_object_receptacle_distance = maximum
        env.placement_initializer = opposite.OppositeSidesTests.sampler(seed)
        return env

    def test_bounded_sampling_both_directions_and_seed_reproducibility(self):
        env, other = self.sampling_env(42, .30), self.sampling_env(42, .30)
        positives, rejected = 0, 0
        for _ in range(200):
            sample, again = env._sample_safe_placements(), other._sample_safe_placements()
            a, b = (np.asarray(sample[key][0]) for key in ("object_0", "receptacle_0"))
            self.assertLess(a[1] * b[1], 0)
            self.assertGreaterEqual(np.linalg.norm(a[:2] - b[:2]), .12)
            self.assertLessEqual(np.linalg.norm(a[:2] - b[:2]), .30)
            positives += int(a[1] > 0)
            rejected += env.last_sampling_stats["maximum_distance_rejections"]
            for key in sample:
                np.testing.assert_array_equal(sample[key][0], again[key][0])
                np.testing.assert_array_equal(sample[key][1], again[key][1])
        self.assertGreater(positives, 65)
        self.assertLess(positives, 135)
        self.assertGreater(rejected, 0)

    def test_unset_limit_preserves_seeded_sampling(self):
        explicit, legacy = self.sampling_env(9, None), self.sampling_env(9, None)
        del legacy.maximum_object_receptacle_distance
        for _ in range(20):
            a, b = explicit._sample_safe_placements(), legacy._sample_safe_placements()
            for key in a:
                np.testing.assert_array_equal(a[key][0], b[key][0])
                np.testing.assert_array_equal(a[key][1], b[key][1])
            self.assertEqual(explicit.last_sampling_stats, legacy.last_sampling_stats)

    def test_invalid_limits_fail_before_environment_or_output_creation(self):
        for maximum in (0., -.3, .12, float("nan"), float("inf")):
            with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError, "maximum_object_receptacle_distance"):
                PlaceEnv(maximum_object_receptacle_distance=maximum)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not_created"
            for task, maximum in (("pick", "0.30"), ("place", "nan"), ("place", "0.12")):
                with patch("sys.argv", ["record", "--task", task, "--max-object-receptacle-distance", maximum,
                                        "--output", str(output)]), self.assertRaisesRegex(ValueError, "--max-object-receptacle-distance"):
                    recording.main()
                self.assertFalse(output.exists())

    def test_metadata_eval_restoration_and_append_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, created = Path(tmp) / "demos", []

            def build(args, objects, receptacles, seed):
                env = FakeEnv("place", objects, receptacles, seed)
                env.placement_layout = args.placement_layout
                env.maximum_object_receptacle_distance = args.maximum_object_receptacle_distance
                created.append(env)
                return env, SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "approach"))

            def expert(*args):
                obs, _, _, _ = created[-1].step(np.zeros(7, dtype=np.float32))
                return obs, dict(success=True, stage="complete")

            command = ["record", "--task", "place", "--num-objects", "1", "--num-receptacles", "1",
                       "--placement-layout", "opposite_sides", "--max-object-receptacle-distance", "0.30",
                       "--num-demos", "2", "--output", str(root)]
            with patch("sys.argv", command), patch.object(recording, "build_task", side_effect=build), \
                    patch.object(recording, "run_expert", side_effect=expert), redirect_stdout(io.StringIO()):
                recording.main()
            entries = [json.loads(line) for line in (root / "place/manifest.jsonl").read_text().splitlines()]
            for entry in entries:
                self.assertEqual(entry["metadata"]["collection_config"]["environment_kwargs"]["maximum_object_receptacle_distance"], .30)
            prepared = json.loads((recording.default_training_dir(root) / "training.json").read_text())
            model = SimpleNamespace(camera_names=["agentview"])
            with patch.object(recording, "PlaceEnv", return_value=SimpleNamespace()) as constructor:
                evaluation.build_environment(2000, 1, False, model, prepared, 300)
            self.assertEqual(constructor.call_args.kwargs["maximum_object_receptacle_distance"], .30)
            self.assertEqual(constructor.call_args.kwargs["placement_layout"], "opposite_sides")
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}
            for value in ("0.25", None):
                changed = list(command)
                index = changed.index("--max-object-receptacle-distance")
                if value is None:
                    del changed[index:index + 2]
                else:
                    changed[index + 1] = value
                with patch("sys.argv", changed), self.assertRaisesRegex(ValueError, "another maximum"):
                    recording.main()
                self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})


if __name__ == "__main__":
    unittest.main(verbosity=2)
