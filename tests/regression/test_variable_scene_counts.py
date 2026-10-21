"""Offline scene-count regression tests; no expert or real simulation runs."""

import copy
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from robosuite.utils import RandomizationError

from environments.pick_env import PickEnv
from environments.place_env import PlaceEnv
from environments.stack_env import StackEnv
from environments.scene_counts import (
    balanced_count_plan, count_distribution, count_options, metadata_counts,
)
from environments.instructions import generate_instruction
from scripts import record_demonstrations as recording
from scripts.audit_dataset import add_semantics, audit_episode, expected_instruction, new_semantic_stats
from data.bc_preparation import dataset_entries


COLORS = ("red", "blue", "green", "yellow")


def metadata(task, num_objects, num_receptacles=None):
    value = {
        "task": task, "num_objects": num_objects, "semantic_split": "train",
        "objects": [{"id": f"object_{i}", "color": COLORS[i], "shape": "cube",
                     "is_target": i == 0} for i in range(num_objects)],
    }
    attributes = {}
    if task == "stack":
        value["stack_order"] = list(COLORS[:num_objects])
        attributes["colors"] = value["stack_order"]
    else:
        value["target_object"] = "object_0"
        if task == "pick":
            value["target_color"] = "red"
            attributes["color"] = "red"
        else:
            value["num_receptacles"] = num_receptacles
            value["receptacles"] = [{"id": f"receptacle_{i}", "color": COLORS[i],
                                    "type": "tray", "is_target": i == 0}
                                   for i in range(num_receptacles)]
            value["target_receptacle"] = "receptacle_0"
            attributes.update(object_color="red", target_color="red")
    value.update(generate_instruction(task, np.random.default_rng(7), **attributes))
    return value


class FakeEnv:
    def __init__(self, task, objects, receptacles, seed):
        self.metadata = metadata(task, objects, receptacles)
        self.rng = np.random.default_rng(seed)
        self.closed = False
        self.timestep = 0
        self.fail_reset_once = False

    def observation(self):
        return {
            "agentview_image": self.rng.integers(20, 40, (32, 32, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": self.rng.integers(20, 40, (32, 32, 3), dtype=np.uint8),
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
            "robot0_joint_vel": np.zeros(7, dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        }

    def reset(self):
        if self.fail_reset_once:
            self.fail_reset_once = False
            raise RandomizationError("injected reset failure")
        self.timestep = 0
        return self.observation()

    def get_task_metadata(self):
        return self.metadata

    def step(self, action):
        self.timestep += 1
        return self.observation(), 0, False, {}

    def close(self):
        self.closed = True


class VariableCountTests(unittest.TestCase):
    def test_balanced_plans_and_validation(self):
        for task, objects, receptacles, episodes in (
            ("pick", (1, 2, 3), (None,), 300),
            ("place", (1, 2, 3), (1, 2, 3), 450),
            ("stack", (2, 3), (None,), 300),
        ):
            plan = balanced_count_plan(episodes, objects, receptacles, 7)
            self.assertEqual(plan, balanced_count_plan(episodes, objects, receptacles, 7))
            self.assertEqual(len(set(Counter(plan).values())), 1)
            self.assertEqual(len(set(plan)), len(objects) * len(receptacles))
        counts = Counter(balanced_count_plan(10, (1, 2, 3)))
        self.assertEqual(max(counts.values()) - min(counts.values()), 1)
        with self.assertRaises(ValueError):
            balanced_count_plan(2, (1, 2, 3))
        with self.assertRaises(ValueError):
            count_options("stack", SimpleNamespace(num_cubes=1))
        with self.assertRaises(ValueError):
            count_options("pick", SimpleNamespace(num_cubes=3, num_receptacles=2))
        with self.assertRaises(ValueError):
            count_options("pick", SimpleNamespace(num_cubes=3, object_counts=[1, 1]))

    def test_environment_metadata_without_physics(self):
        # Keep real task metadata methods, replace only simulator initialization.
        base = "robosuite.environments.manipulation.manipulation_env.ManipulationEnv"
        with patch(f"{base}.__init__", return_value=None):
            for num_objects in (1, 2, 3):
                for fixed in (None, "red"):
                    env = PickEnv(num_objects=num_objects, fixed_target_color=fixed)
                    env.rng = np.random.default_rng(7)
                    for _ in range(3):
                        env._sample_task_metadata()
                        value = env.get_task_metadata()
                        self.assertEqual(value["num_objects"], num_objects)
                        self.assertEqual(len(value["distractors"]), num_objects - 1)
                        self.assertEqual(expected_instruction(value), value["instruction"])
                        if fixed:
                            self.assertEqual(value["target_color"], fixed)
            for num_objects in (2, 3):
                env = StackEnv(num_cubes=num_objects)
                env.rng = np.random.default_rng(7)
                env._sample_task_metadata()
                value = env.get_task_metadata()
                self.assertEqual(len(value["stack_order"]), num_objects)
                self.assertEqual(expected_instruction(value), value["instruction"])
            with self.assertRaises(ValueError):
                StackEnv(num_cubes=1)
            for objects in (1, 2, 3):
                for receptacles in (1, 2, 3):
                    env = PlaceEnv(num_objects=objects, num_receptacles=receptacles)
                    env.rng = np.random.default_rng(7)
                    env.deterministic_reset = False
                    with patch(f"{base}._reset_internal", return_value=None), patch.object(
                        env, "_sample_safe_placements", return_value={}
                    ), patch.object(env, "_apply_colors"):
                        env._reset_internal()
                    value = env.get_task_metadata()
                    self.assertEqual(metadata_counts(value), (objects, receptacles))
                    self.assertEqual(expected_instruction(value), value["instruction"])

    def test_recording_retry_quotas_audit_and_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            created = []
            seeds = []
            calls = []

            def build(args, objects, receptacles, seed):
                env = FakeEnv(args.task, objects, receptacles, seed)
                env.fail_reset_once = not created
                created.append(env)
                seeds.append(seed)
                expert = SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "approach"))
                return env, expert

            def expert(task, policy, observation, value, render):
                calls.append(metadata_counts(value))
                if len(calls) == 1:
                    raise RandomizationError("injected expert failure")
                observation, _, _, _ = created[-1].step(np.zeros(7, dtype=np.float32))
                return observation, {"success": True, "stage": "complete"}

            command = ["record", "--task", "place", "--episodes", "9", "--output", str(root),
                       "--training-dir", str(root / "training"),
                       "--object-counts", "1", "2", "3", "--receptacle-counts", "1", "2", "3",
                       "--camera-obs", "--camera-names", "agentview", "robot0_eye_in_hand"]
            with patch("sys.argv", command), patch.object(recording, "build_task", side_effect=build), patch.object(
                recording, "run_expert", side_effect=expert
            ), redirect_stdout(io.StringIO()):
                recording.main()
                recording.main()
            self.assertEqual(calls[0], calls[1])
            attempts = [json.loads(line) for line in (root / "place/attempts.jsonl").read_text().splitlines()]
            self.assertEqual(attempts[0]["failure_reason"], "scene_sampling_failure")
            self.assertEqual(attempts[0]["num_objects"], attempts[1]["num_objects"])
            self.assertTrue(all(env.closed for env in created))
            self.assertEqual(len(seeds), len(set(seeds)))
            manifests = [json.loads(line) for line in (root / "place/manifest.jsonl").read_text().splitlines()]
            self.assertEqual(len(manifests), 18)
            self.assertEqual(Counter(metadata_counts(entry["metadata"]) for entry in manifests),
                             Counter({(o, r): 2 for o in (1, 2, 3) for r in (1, 2, 3)}))
            self.assertEqual(manifests[-1]["episode"], 17)
            self.assertTrue((root / "training/normalization.json").is_file())
            self.assertTrue((root / "training/training.json").is_file())
            args = SimpleNamespace(camera_names=["agentview", "robot0_eye_in_hand"],
                                   mode="full", rgb_samples=3)
            stats = new_semantic_stats()
            for entry in manifests:
                errors, _, value = audit_episode(root / "place" / entry["path"], "place", entry, args)
                self.assertEqual(errors, [])
                add_semantics(stats, "place", value)
            self.assertEqual(stats["object_counts"], Counter({"1": 6, "2": 6, "3": 6}))
            invalid = copy.deepcopy(manifests[0])
            invalid["num_objects"] = 99
            errors, _, _ = audit_episode(root / "place" / invalid["path"], "place", invalid, args)
            self.assertIn("manifest num_objects does not match metadata entity list", errors)
            summaries = [json.loads(path.read_text()) for path in (root / "place").glob("collection_*.json")]
            self.assertEqual(len(summaries), 2)
            for summary in summaries:
                self.assertEqual(summary["saved_count_distribution"], summary["target_count_distribution"])
            loaded = dataset_entries(root)
            self.assertTrue(all(entry["num_objects"] and entry["num_receptacles"] for entry in loaded))

    def test_attempt_limit_records_missing_quotas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = FakeEnv("pick", 1, None, 7)
            command = ["record", "--task", "pick", "--episodes", "3", "--output", str(root),
                       "--object-counts", "1", "2", "3", "--max-attempts", "1"]
            expert = SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "approach"))
            with patch("sys.argv", command), patch.object(recording, "build_task", return_value=(env, expert)), patch.object(
                recording, "run_expert", return_value=(None, {"success": False, "stage": "complete"})
            ), redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Only recorded 0/3"):
                recording.main()
            self.assertTrue(env.closed)
            summary = json.loads((root / "pick/collection_000000.json").read_text())
            self.assertEqual(summary["saved"], 0)
            self.assertEqual(summary["saved_count_distribution"], {})
            self.assertEqual(summary["target_count_distribution"], count_distribution([(1, None), (2, None), (3, None)]))

    def test_benchmark_count_failure_reporting(self):
        from benchmarks import expert_reliability as benchmark

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def build(task, seed, instruction_split, render, objects, receptacles, fixed, *settings):
                return FakeEnv(task, objects, receptacles, seed), SimpleNamespace()

            def expert(task, policy, observation, value, render):
                return observation, {"success": value["num_objects"] != 2, "stage": "complete"}

            with patch.object(benchmark, "build_task", side_effect=build), patch.object(
                benchmark, "run_expert", side_effect=expert
            ), patch.object(benchmark, "target_snapshot", return_value={}), redirect_stdout(io.StringIO()):
                result = benchmark.benchmark_task("pick", 6, root, "train", False,
                                                  object_counts=(1, 2, 3))
            groups = result["by_scene_counts"]
            self.assertEqual([groups[f"objects={n}"]["attempted"] for n in (1, 2, 3)], [2, 2, 2])
            self.assertEqual(groups["objects=2"]["success_rate"], 0)
            self.assertFalse(groups["objects=2"]["gate_passed"])
            self.assertEqual(result["successful"], 4)

    def test_builder_and_evaluator_arguments(self):
        from scripts import evaluate_visual_bc

        with patch("sys.argv", ["record", "--task", "pick", "--num-objects", "1"]):
            args = recording.parse_args()
        self.assertEqual(count_options("pick", args), ((1,), (None,)))
        with patch.object(recording, "PlaceEnv") as place, patch.object(recording, "PandaController"), patch.object(
            recording, "GraspExpert"
        ), patch.object(recording, "PlaceExpert"):
            args.task = "place"
            recording.build_task(args, 2, 1, seed=123)
            self.assertEqual(place.call_args.kwargs["num_objects"], 2)
            self.assertEqual(place.call_args.kwargs["num_receptacles"], 1)
        with patch("sys.argv", ["evaluate", "--checkpoint", "test.pt", "--normalization", "norm.json",
                                "--num-objects", "1", "--output-dir", "test-output"]):
            args = evaluate_visual_bc.parse_args()
        self.assertEqual(args.num_cubes, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
