"""Offline contracts for yaw/rotation ablations and causal GRU training."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from controllers.panda_controller import PandaController
from data.bc_preparation import load_bc_training
from data.manipulation_dataset import EpisodeSequenceDataset, CAMERA_OBSERVATION_KEYS
from data.normalization import PolicyNormalizer
from environments.pick_config import validate_pick_arguments, validate_pick_settings
from experts.pick_expert import PickExpert
from models.visual_bc import VisualBCPolicy
from scripts import record_demonstrations as recording, train_visual_bc as training
from scripts import evaluate_visual_bc as evaluation
from benchmarks import expert_reliability as benchmark
from tests.regression import test_visual_bc as fixtures
from tests.regression.test_variable_scene_counts import FakeEnv


class PickExperimentTests(unittest.TestCase):
    setUp = fixtures.VisualBCTests.setUp
    prepare = fixtures.VisualBCTests.prepare
    command = fixtures.VisualBCTests.command

    def test_yaw_cli_validation_and_factory_preserves_position_defaults(self):
        for flag, yaw in (("--no-random-yaw", (0, 0)), ("--random-yaw", (-45, 45))):
            command = ["record", "--task", "pick", flag, "--rotation-mode", "none"]
            with patch("sys.argv", command):
                args = recording.parse_args()
            self.assertEqual(validate_pick_arguments(args), yaw)
            with patch.object(recording, "PickEnv") as env, patch.object(recording, "PandaController"), patch.object(recording, "PickExpert") as expert:
                recording.build_task(args, 3, seed=42)
            np.testing.assert_allclose(env.call_args.kwargs["yaw_range"], np.deg2rad(yaw))
            self.assertNotIn("x_range", env.call_args.kwargs)
            self.assertNotIn("y_range", env.call_args.kwargs)
            self.assertEqual(expert.call_args.kwargs["rotation_mode"], "none")
        for yaw in ((45, -45), (float("nan"), 0), (0, float("inf"))):
            with self.assertRaises(ValueError):
                validate_pick_settings("none", yaw)
        with self.assertRaises(ValueError):
            validate_pick_arguments(SimpleNamespace(task="place", rotation_mode="none", random_yaw=False))

    def test_yaw_toggle_defaults_and_numeric_cli_rejection(self):
        for module in (recording, benchmark):
            for flags, expected in (([], True), (["--random-yaw"], True), (["--no-random-yaw"], False)):
                with patch("sys.argv", ["command", "--task", "pick", *flags]):
                    args = module.parse_args()
                self.assertIs(args.random_yaw, expected)
                self.assertEqual(validate_pick_arguments(args), (-45., 45.) if expected else (0., 0.))
            for flags in (("--yaw-range-deg", "-45", "45"), ("--random-yaw", "1"), ("--no-random-yaw", "0")):
                with patch("sys.argv", ["command", "--task", "pick", *flags]), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        module.parse_args()
                self.assertEqual(error.exception.code, 2)
        for task in ("place", "stack", "all"):
            with self.assertRaises(ValueError):
                validate_pick_arguments(SimpleNamespace(task=task, rotation_mode="align_yaw", random_yaw=False))

    def test_all_six_collection_configs_and_append_guards(self):
        current = []

        def build(args, objects, receptacles, seed):
            env = FakeEnv("pick", objects, receptacles, seed)
            observe = env.observation
            def observation():
                value = observe()
                if len(args.camera_names) == 1:
                    value.pop("robot0_eye_in_hand_image")
                return value
            env.observation = observation
            env.x_range, env.y_range = (-.10, .10), (-.20, .20)
            env.yaw_range = tuple(np.deg2rad(args.yaw_range_deg))
            current.append(env)
            return env, SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "lift"))

        def expert(*unused):
            observation, _, _, _ = current[-1].step(np.zeros(7, np.float32))
            return observation, {"success": True, "stage": "complete"}

        for camera_count in (1, 2):
            for yaw, mode in (((0, 0), "none"), ((-45, 45), "none"), ((-45, 45), "align_yaw")):
                name = f"c{camera_count}_{yaw[0]}_{mode}"
                root, prepared = self.root / name, self.root / f"{name}_bc"
                cameras = ["agentview"] + (["robot0_eye_in_hand"] if camera_count == 2 else [])
                yaw_flag = "--random-yaw" if yaw[0] else "--no-random-yaw"
                command = ["record", "--task", "pick", "--num-demos", "2", "--output", str(root),
                           "--training-dir", str(prepared), "--rotation-mode", mode,
                           yaw_flag, "--camera-names", *cameras]
                with self.subTest(cameras=cameras, yaw=yaw, mode=mode), patch("sys.argv", command), patch.object(
                    recording, "build_task", side_effect=build
                ), patch.object(recording, "run_expert", side_effect=expert), redirect_stdout(io.StringIO()):
                    recording.main()
                config, _ = load_bc_training(root, prepared)
                self.assertEqual(config["camera_names"], cameras)
                self.assertEqual(config["proprio_dim"], 16)
                self.assertEqual(config["predict_orientation"], mode != "none")
                self.assertEqual(config["rotation_mode"], mode)
                collection = config["environment_configs"]["pick"]
                self.assertEqual(collection["yaw_range_deg"], list(yaw))
                self.assertEqual(collection["environment_kwargs"]["x_range"], [-.10, .10])
                manifest = root / "pick/manifest.jsonl"
                before = manifest.read_bytes()
                changed_mode, changed_yaw = command.copy(), command.copy()
                changed_mode[changed_mode.index("--rotation-mode") + 1] = "align_yaw" if mode == "none" else "none"
                changed_yaw[changed_yaw.index(yaw_flag)] = "--no-random-yaw" if yaw[0] else "--random-yaw"
                for changed in (changed_mode, changed_yaw):
                    with patch("sys.argv", changed), self.assertRaisesRegex(ValueError, "yaw/rotation"):
                        recording.main()
                    self.assertEqual(manifest.read_bytes(), before)
                model = VisualBCPolicy(16, cameras, mode != "none")
                with patch.object(recording, "build_task", return_value=(object(), None)) as factory:
                    evaluation.build_environment(1000, 3, False, model, config, 500)
                restored = factory.call_args.args[0]
                self.assertEqual(restored.rotation_mode, mode)
                self.assertEqual(restored.yaw_range_deg, list(yaw))

    def test_none_expert_does_not_wait_for_or_command_orientation(self):
        class Env:
            action_dim = 7
            target_object_id = "cube_0"
            def __init__(self):
                self.eef = np.array([.1, 0, 1.1], np.float32)
                self.cube = np.array([0, 0, .83], np.float32)
                self.held, self.actions = False, []
            def resolve_object_id(self, name):
                return name
            def observation(self):
                return {"robot0_eef_pos": self.eef.copy(), "robot0_eef_quat": np.array([0., 0., 0., 1.]),
                        "cube_0_pos": self.cube.copy(), "cube_0_quat": np.array([0., 0., np.sin(.35), np.cos(.35)])}
            def step(self, action):
                previous = self.eef.copy()
                self.eef += action[:3] * .05
                if action[6] > 0 and abs(self.eef[2] - self.cube[2]) < .025:
                    self.held = True
                if self.held:
                    self.cube += self.eef - previous
                self.actions.append(action.copy())
                return self.observation(), 0, False, {}
            def _check_success(self):
                return self.cube[2] > .95
        env = Env()
        controller = PandaController(env)
        expert = PickExpert(controller, rotation_mode="none")
        with patch.object(controller, "orientation_distance", side_effect=AssertionError("orientation gate used")), patch.object(
            expert, "get_target_orientation", side_effect=AssertionError("yaw target used")
        ):
            _, result = expert.run(env.observation())
        self.assertTrue(result["success"])
        self.assertEqual(np.count_nonzero(np.asarray(env.actions)[:, 3:6]), 0)
        env = Env()
        env.step = lambda action: (env.actions.append(action.copy()) or env.observation(), 0, True, {})
        with redirect_stdout(io.StringIO()):
            PickExpert(PandaController(env), rotation_mode="align_yaw").run(env.observation())
        self.assertGreater(np.linalg.norm(env.actions[0][3:6]), 0)

    def test_preparation_rejects_none_mode_with_rotational_labels(self):
        for entry in self.entries:
            entry["metadata"].update(control_orientation=False, rotation_mode="none")
        self.manifest.write_text("".join(json.dumps(value) + "\n" for value in self.entries))
        with self.assertRaisesRegex(ValueError, "Nonzero rotational action"):
            self.prepare()

    def test_benchmark_factory_uses_the_same_yaw_and_rotation_settings(self):
        for yaw, mode in (((0, 0), "none"), ((-45, 45), "none"), ((-45, 45), "align_yaw")):
            with patch.object(benchmark, "PickEnv") as env, patch.object(benchmark, "PandaController"), patch.object(
                benchmark, "PickExpert"
            ) as expert:
                benchmark.build_task("pick", 1000, "train", False, rotation_mode=mode, yaw_range_deg=yaw)
            np.testing.assert_allclose(env.call_args.kwargs["yaw_range"], np.deg2rad(yaw))
            self.assertNotIn("x_range", env.call_args.kwargs)
            self.assertNotIn("y_range", env.call_args.kwargs)
            self.assertEqual(expert.call_args.kwargs["rotation_mode"], mode)

    def test_invalid_temporal_settings_fail_before_writing_artifacts(self):
        for flags in (("--sequence-length", "8"), ("--temporal", "--sequence-length", "0"),
                      ("--temporal", "--sequence-stride", "0"),
                      ("--temporal", "--sequence-length", "4", "--sequence-stride", "8")):
            with self.subTest(flags=flags), patch("sys.argv", self.command(*flags)), self.assertRaises(ValueError):
                training.main()
            self.assertFalse(self.output.exists())

    def test_temporal_loader_smoke_and_default_recurrent_head(self):
        from tests.smoke import test_dataset_loader as smoke

        self.prepare()
        command = ["loader", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                   "--temporal", "--sequence-length", "8", "--sequence-stride", "4"]
        output = io.StringIO()
        with patch("sys.argv", command), redirect_stdout(output):
            smoke.main()
        self.assertIn("Dataset loader smoke test: PASS", output.getvalue())
        self.assertEqual(VisualBCPolicy(16, temporal=True).get_config()["hidden_dims"], [128])
        self.assertEqual(VisualBCPolicy(16).get_config()["hidden_dims"], [256, 128])

    def test_temporal_windows_are_causal_padded_and_episode_local(self):
        config = self.prepare()
        dataset = EpisodeSequenceDataset(self.dataset, self.prepared / "train.jsonl", sequence_length=4, sequence_stride=2,
                                         proprio_keys=config["proprio_keys"], camera_keys=tuple(CAMERA_OBSERVATION_KEYS.values()))
        for index, (episode_index, start) in enumerate(dataset.windows):
            sample = dataset[index]
            entry = dataset.episodes[episode_index]
            with np.load(self.dataset / entry["path"]) as archive:
                actions = archive["actions"]
            valid = min(4, len(actions) - start)
            self.assertEqual(sample["episode_path"], entry["path"])
            self.assertEqual(sample["valid_mask"].sum(), valid)
            self.assertEqual(sample["image"].shape, (4, 3, 32, 32))
            torch.testing.assert_close(sample["action"][:valid], torch.from_numpy(actions[start:start + valid]))
            for offset in range(valid):
                time = start + offset
                expected = -1. if time == 0 else (1. if actions[time - 1, 6] > 0 else -1.)
                self.assertEqual(float(sample["previous_gripper"][offset, 0]), expected)
            if valid < 4:
                torch.testing.assert_close(sample["action"][valid:], sample["action"][valid - 1:valid].expand(4 - valid, -1))

    def test_temporal_shapes_gradients_causality_and_streaming_equivalence(self):
        for dual in (False, True):
            for orientation in (False, True):
                cameras = ("agentview", "robot0_eye_in_hand") if dual else ("agentview",)
                model = VisualBCPolicy(16, cameras, orientation, temporal=True)
                values = {"image": torch.rand(1, 3, 3, 32, 32), "proprio": torch.randn(1, 3, 16),
                          "previous_gripper": torch.tensor([[[-1.], [1.], [-1.]]])}
                if dual:
                    values["eye_in_hand_image"] = torch.rand_like(values["image"])
                output = model(**values)
                self.assertEqual(output["action"].shape, (1, 3, 7))
                self.assertEqual(output["hidden"].shape, (1, 1, 256))
                (output["motion"].square().mean() + output["gripper_logit"].square().mean()).backward()
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                if not orientation:
                    self.assertEqual(output["action"][..., 3:6].count_nonzero(), 0)
                with torch.no_grad():
                    expected = model(**values)
                    changed = {key: value.clone() for key, value in values.items()}
                    for value in changed.values():
                        value[:, 2] += 5
                    torch.testing.assert_close(model(**changed)["motion"][:, :2], expected["motion"][:, :2], rtol=0, atol=0)
                    hidden = None
                    for time in range(3):
                        step = model.step(values["image"][:, time], values["proprio"][:, time],
                                          eye_in_hand_image=values["eye_in_hand_image"][:, time] if dual else None,
                                          previous_gripper=float(values["previous_gripper"][0, time, 0]), hidden=hidden)
                        torch.testing.assert_close(step["motion"], expected["motion"][:, time], atol=1e-6, rtol=1e-5)
                        hidden = step["hidden"]

    def test_temporal_masked_loss_ignores_padded_targets(self):
        config = self.prepare()
        normalizer = PolicyNormalizer.from_statistics(config["normalization"])
        dataset = EpisodeSequenceDataset(self.dataset, self.prepared / "val.jsonl", sequence_length=8, sequence_stride=8,
                                         proprio_keys=config["proprio_keys"])
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertTrue((~batch["valid_mask"]).any())
        model = VisualBCPolicy(16, temporal=True)
        first = training.run_epoch(model, [batch], normalizer, self.device)
        batch["action"][~batch["valid_mask"]] = 1000
        second = training.run_epoch(model, [batch], normalizer, self.device)
        self.assertEqual(first, second)

    def test_temporal_training_checkpoint_resume_and_eval_load(self):
        command = self.command("--temporal", "--sequence-length", "3", "--sequence-stride", "2", "--batch-size", "2")
        with patch("sys.argv", command), redirect_stdout(io.StringIO()):
            training.main()
        command.remove("--temporal")
        command.extend(("--resume", str(self.output / "latest.pt"), "--epochs", "2"))
        with patch("sys.argv", command), redirect_stdout(io.StringIO()):
            training.main()
        model, _, checkpoint, _, _ = evaluation.load_policy(SimpleNamespace(
            checkpoint=self.output / "latest.pt", normalization=None, image_size=None), self.device)
        self.assertTrue(model.temporal)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["training_config"]["sequence_stride"], 2)
        before = (self.output / "config.json").read_bytes()
        command[command.index("--sequence-length") + 1] = "4"
        with patch("sys.argv", command), self.assertRaisesRegex(ValueError, "sequence-length"):
            training.main()
        self.assertEqual((self.output / "config.json").read_bytes(), before)

    def test_recurrent_evaluation_resets_hidden_and_previous_gripper(self):
        config = self.prepare()
        normalizer = PolicyNormalizer.from_statistics(config["normalization"])
        model = VisualBCPolicy(16, config["camera_names"], temporal=True)
        model.eval()
        class Env:
            object_ids = ("cube_0",)
            action_spec = (np.full(7, -1), np.full(7, 1))
            def __init__(self):
                self.actions = []
            def reset(self):
                return {"agentview_image": np.full((32, 32, 3), 80, np.uint8),
                        "robot0_eye_in_hand_image": np.full((32, 32, 3), 100, np.uint8),
                        "robot0_joint_pos": np.zeros(7), "robot0_eef_pos": np.array([0., 0., 1.]),
                        "robot0_gripper_qpos": np.zeros(2), "robot0_eef_quat": np.array([0., 0., 0., 1.])}
            def get_task_metadata(self):
                return {"target_object": "cube_0", "target_color": "red"}
            def get_object_position(self, identifier):
                return np.array([0., 0., .83])
            def step(self, action):
                self.actions.append(action.copy())
                return self.reset(), 0, False, {}
            def _check_success(self):
                return False
        env = Env()
        with patch.object(model, "step", wraps=model.step) as predict:
            for episode in range(2):
                evaluation.evaluate_episode(env, model, normalizer, 128, self.device, 2, 20,
                                            False, episode, None, 20., "red")
        for index in (0, 2):
            self.assertIsNone(predict.call_args_list[index].kwargs["hidden"])
            self.assertEqual(predict.call_args_list[index].kwargs["previous_gripper"], -1.)
            self.assertIsNotNone(predict.call_args_list[index + 1].kwargs["hidden"])
            self.assertEqual(predict.call_args_list[index + 1].kwargs["previous_gripper"], env.actions[index][6])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
