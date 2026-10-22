"""Fixed matching-color Place contracts, without a real simulator or GPU job."""

import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

from data.bc_preparation import prepare_bc_training
from environments.entity_utils import sample_unique_colors
from environments.place_env import PlaceEnv
from scripts import evaluate_visual_bc as evaluation, record_demonstrations as recording
from scripts import train_visual_bc as training
from tests.regression import test_visual_bc as fixtures
from tests.regression.test_variable_scene_counts import FakeEnv


class PlaceVisualBCTests(unittest.TestCase):
    def setUp(self):
        fixtures.VisualBCTests.setUp(self)
        (self.dataset / "pick").rename(self.dataset / "place")
        self.manifest = self.dataset / "place/manifest.jsonl"
        for entry in self.entries:
            entry.update(task="place", num_receptacles=1)
            metadata = entry["metadata"]
            metadata.update(task="place", fixed_target_color="red", target_receptacle="tray_0",
                            target_receptacle_color="red",
                            receptacles=[dict(id="tray_0", color="red", type="tray", is_target=True)])
            with h5py.File(self.dataset / "place" / entry["path"], "r+") as data:
                data["metadata_json"][()] = json.dumps(metadata)
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text("".join(json.dumps(v) + "\n" for v in self.entries), encoding="utf-8")

    def test_fixed_color_sampling_and_place_reset(self):
        rng = np.random.default_rng(7)
        for count in (1, 2, 3, 4):
            colors = sample_unique_colors(rng, count, "red")
            self.assertEqual(len(set(colors)), count)
            self.assertIn("red", colors)
        base = "robosuite.environments.manipulation.manipulation_env.ManipulationEnv"
        with patch(f"{base}.__init__", return_value=None):
            for count in (1, 2, 3):
                env = PlaceEnv(num_objects=count, num_receptacles=count, fixed_target_color="red")
                env.rng, env.deterministic_reset = rng, False
                with patch(f"{base}._reset_internal"), patch.object(env, "_sample_safe_placements", return_value={}), patch.object(env, "_apply_colors"):
                    for _ in range(5):
                        env._reset_internal()
                        meta = env.get_task_metadata()
                        self.assertEqual(meta["target_color"], "red")
                        self.assertEqual(meta["target_receptacle_color"], "red")
                        self.assertEqual(env.object_id_to_color[meta["target_object"]], "red")
                        self.assertEqual(env.receptacle_id_to_color[meta["target_receptacle"]], "red")

    def test_prepare_and_train_frame_and_gru(self):
        prepared = prepare_bc_training(self.dataset, self.prepared)
        self.assertEqual(prepared["tasks"], ["place"])
        self.assertEqual(prepared["fixed_target_color"], "red")
        for temporal in (False, True):
            output = self.output / ("gru" if temporal else "frame")
            command = ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                       "--output-dir", str(output), "--epochs", "1", "--device", "cpu"]
            if temporal:
                command += ["--temporal", "--sequence-length", "3", "--sequence-stride", "2", "--batch-size", "4"]
            with patch("sys.argv", command), redirect_stdout(io.StringIO()):
                training.main()
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["data_config"]["tasks"], ["place"])
            args = SimpleNamespace(checkpoint=output / "best.pt", image_size=None, normalization=None)
            model, _, _, _, _ = evaluation.load_policy(args, self.device)
            self.assertEqual(model.temporal, temporal)

    def test_ambiguous_or_mismatched_pair_is_rejected(self):
        self.entries[0]["metadata"]["receptacles"][0]["color"] = "blue"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "matching-color"):
            prepare_bc_training(self.dataset, self.prepared)

    def test_place_geometry_release_wrong_tray_and_distractor(self):
        position = np.array([0., 0., .824])
        trays = {"tray": np.array([0., 0., .8]), "other": np.array([.2, 0., .8])}
        distractor = np.array([-.2, 0., .82])
        env = SimpleNamespace(target_object_id="cube", target_receptacle_id="tray",
                              object_ids=["cube", "other_cube"], receptacle_ids=list(trays),
                              get_object_position=lambda value=None: distractor if value == "other_cube" else position,
                              get_object_orientation=lambda: np.array([0., 0., 0., 1.]),
                              get_receptacle_position=lambda value: trays[value],
                              get_receptacle_orientation=lambda value: np.array([0., 0., 0., 1.]))
        initial = {"other_cube": distractor.copy()}
        status = lambda eef: PlaceEnv.placement_status(env, eef, initial)
        self.assertTrue(status([0., 0., 1.])["success"])
        self.assertFalse(status(position)["released"])
        position[2] = 1.1
        self.assertFalse(status([0., 0., 1.4])["inside_target"])
        position[:] = [.2, 0., .824]
        self.assertTrue(status([0., 0., 1.])["wrong_target"])
        position[:] = [0., 0., .824]
        distractor[0] -= .1
        self.assertFalse(status([0., 0., 1.])["distractors_stable"])

    def test_evaluator_restores_place_factory_and_counts(self):
        prepared = prepare_bc_training(self.dataset, self.prepared)
        with patch.object(recording, "build_task", return_value=(SimpleNamespace(), None)) as factory:
            evaluation.build_environment(123, 2, False, SimpleNamespace(camera_names=["agentview"]), prepared, 3000)
        args = factory.call_args.args[0]
        self.assertEqual((args.task, args.fixed_target_color, args.num_receptacles, args.max_steps), ("place", "red", 1, 3000))
        env = SimpleNamespace(num_receptacles=1, close=lambda: None)
        with patch("sys.argv", ["eval", "--checkpoint", "model.pt", "--episodes", "2", "--num-seeds", "1", "--output-dir", str(self.output)]):
            args = evaluation.parse_args()
        with patch.object(evaluation, "build_environment", return_value=env), patch.object(
            evaluation, "evaluate_episode", return_value=dict(success=True, failure_reason=None, steps=20)
        ), redirect_stdout(io.StringIO()):
            summary = evaluation.evaluate_policy(args, SimpleNamespace(policy_type="visual_bc", camera_names=["agentview"], temporal=False, predict_orientation=True),
                                                  None, dict(data_config=prepared, epoch=1, best_val_loss=.1), self.device)
        self.assertEqual(summary["tasks"][0]["task"], "place")
        self.assertEqual(summary["overall_success_rate"], 1.)
        self.assertEqual(args.max_steps, 3000)
        self.assertTrue((self.output / "place_attempts.jsonl").is_file())

    def test_place_episode_requires_release_not_pick_lift(self):
        prepared = prepare_bc_training(self.dataset, self.prepared)
        from data.normalization import PolicyNormalizer

        normalizer = PolicyNormalizer.from_statistics(prepared["normalization"])
        action = normalizer.normalize_action(torch.tensor([[0., 0., 0., 0., 0., 0., -1.]]))
        model = SimpleNamespace(camera_names=["agentview"], step=lambda **kwargs: {"action": action})
        for expected in (None, "not_released", "wrong_receptacle"):
            env = FakeEnv("place", 1, 1, 7)
            env.object_ids = ["object_0"]
            env.receptacle_ids = ["receptacle_0"]
            env.get_object_position = lambda value=None: np.array([0., 0., .824])
            env.get_receptacle_position = lambda value=None: np.array([0., 0., .8])
            env.action_spec = (-np.ones(7), np.ones(7))
            env.placement_status = lambda *args: dict(
                success=expected is None, inside_target=expected != "wrong_receptacle",
                released=expected != "not_released", wrong_target=expected == "wrong_receptacle",
                distractors_stable=True)
            result = evaluation.evaluate_episode(env, model, normalizer, 128, self.device,
                                                 20, 20, False, 0, None, 20., "red")
            self.assertEqual(result["success"], expected is None)
            self.assertEqual(result["failure_reason"], expected)
            self.assertEqual(result["steps"], 20)
            self.assertEqual(result["target_maximum_lift"], 0.)
