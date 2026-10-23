"""Visual-BC contract tests; synthetic episodes and no real simulator execution."""

import io
import hashlib
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import h5py
import torch
from torch import nn
from torch.utils.data import DataLoader

from controllers.panda_controller import PandaController
from data.bc_preparation import prepare_bc_training, load_bc_training, suspicious_camera_frames, write_split
from data.manipulation_dataset import ManipulationDataset, CAMERA_OBSERVATION_KEYS
from data.normalization import PolicyNormalizer
from data.episode_io import open_episode, write_episode
from experts.pick_expert import PickExpert
from models.visual_bc import VisualBCPolicy
from models.visual_encoder import SpatialSoftmax, VisualEncoder
from scripts import train_visual_bc as training, record_demonstrations as recording
from tests.regression.test_variable_scene_counts import FakeEnv


class VisualBCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "demos"
        self.prepared = self.root / "prepared"
        self.output = self.root / "run"
        (self.dataset / "pick").mkdir(parents=True)
        self.device = torch.device("cpu")
        self.lengths = [3, 7, 2, 5, 4, 6, 3, 4, 2, 3]
        self.entries = []
        for index, steps in enumerate(self.lengths):
            time = np.arange(steps + 1, dtype=np.float32)[:, None]
            metadata = {"task": "pick", "target_color": "red", "target_object": "cube_0",
                        "control_orientation": True, "objects": [{"id": "cube_0", "color": "red", "is_target": True}]}
            actions = np.arange(steps * 7, dtype=np.float32).reshape(steps, 7) / 100 + index / 10
            actions[:, 6] = np.where(np.arange(steps) % 2, 1, -1)
            payload = {"actions": actions, "metadata_json": np.asarray(json.dumps(metadata)),
                       "obs__robot0_joint_pos": np.repeat(time + index, 7, axis=1),
                       "obs__robot0_eef_pos": np.repeat(time / 100 + index, 3, axis=1),
                       "obs__robot0_gripper_qpos": np.zeros((steps + 1, 2), dtype=np.float32),
                       "obs__robot0_eef_quat": np.tile([0., 0., 0., 1.], (steps + 1, 1)),
                       "obs__cube_0_pos": np.tile([index / 100, 0., .83], (steps + 1, 1)),
                       "obs__cube_0_quat": np.tile([0., 0., 0., 1.], (steps + 1, 1))}
            for key in CAMERA_OBSERVATION_KEYS.values():
                payload[key] = np.full((steps + 1, 32, 32, 3), 80 + index, dtype=np.uint8)
            name = f"episode_{index:06d}.hdf5"
            write_episode(self.dataset / "pick" / name, payload)
            self.entries.append({"episode": index, "task": "pick", "path": name, "steps": steps,
                                 "instruction": "Pick up the red cube.", "metadata": metadata, "num_objects": 1})
        self.manifest = self.dataset / "pick/manifest.jsonl"
        self.manifest.write_text("".join(json.dumps(value) + "\n" for value in self.entries), encoding="utf-8")

    def prepare(self):
        return prepare_bc_training(self.dataset, self.prepared)

    def test_split_hash_is_platform_independent_without_text_newline_translation(self):
        path = self.root / "train.jsonl"
        original_write_text = Path.write_text

        def windows_write_text(target, text, *args, **kwargs):
            return original_write_text(target, text.replace("\n", "\r\n"), *args, **kwargs)

        entries = [{"task": "pick", "instruction": "angkat kubus", "steps": 3}]
        with patch.object(Path, "write_text", windows_write_text):
            digest = write_split(path, entries)
        content = path.read_bytes()
        self.assertEqual(digest, hashlib.sha256(content).hexdigest())
        self.assertEqual(content, (json.dumps(entries[0], sort_keys=True) + "\n").encode("utf-8"))
        self.assertNotIn(b"\r\n", content)

    def command(self, *extra):
        return ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                "--output-dir", str(self.output), "--epochs", "1", "--device", "cpu", *extra]

    def test_periodic_rollout_and_early_stopping_resume(self):
        stats = dict(loss=1., xyz_mse=1., rotation_mse=0., gripper_bce=0.)
        result = dict(epoch=2, success_rate=.5, summary="mock-summary.json")
        command = self.command("--epochs", "10", "--rollout-eval-every", "2",
                               "--early-stopping-patience", "2")
        with patch("sys.argv", command), patch.object(training, "run_epoch", return_value=stats), \
                patch.object(training, "run_rollout", return_value=result) as rollout, redirect_stdout(io.StringIO()):
            training.main()
        rollout.assert_called_once()
        latest = torch.load(self.output / "latest.pt", map_location="cpu", weights_only=False)
        best_rollout = torch.load(self.output / "best_rollout.pt", map_location="cpu", weights_only=False)
        self.assertEqual(latest["epoch"], 3)
        self.assertEqual(latest["epochs_without_improvement"], 2)
        self.assertEqual(latest["best_rollout_success_rate"], .5)
        self.assertEqual(best_rollout["epoch"], 2)
        self.assertEqual(latest["rollout_history"], [result])
        with patch("sys.argv", command + ["--resume", str(self.output / "latest.pt")]), \
                patch.object(training, "run_epoch") as run, redirect_stdout(io.StringIO()):
            training.main()
        run.assert_not_called()

    def test_rollout_subprocess_arguments_and_summary(self):
        with patch("sys.argv", self.command("--rollout-eval-num-seeds", "5", "--rollout-eval-episodes", "20",
                                             "--rollout-eval-max-steps", "300")):
            args = training.parse_args()
        output = self.output / "rollouts" / "epoch_0010"
        output.mkdir(parents=True)
        (output / "summary.json").write_text(json.dumps({"overall_success_rate": .25}), encoding="utf-8")
        with patch.object(training.subprocess, "run") as execute:
            result = training.run_rollout(args, 10, torch.device("cpu"))
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--num-seeds") + 1], "5")
        self.assertEqual(command[command.index("--episodes") + 1], "20")
        self.assertEqual(command[command.index("--seed") + 1], "1000")
        self.assertEqual(command[command.index("--max-steps") + 1], "300")
        self.assertTrue(execute.call_args.kwargs["check"])
        self.assertEqual(result["success_rate"], .25)

    def test_spatial_softmax_coordinates_and_gradients(self):
        pool = SpatialSoftmax()
        torch.testing.assert_close(pool(torch.zeros(1, 2, 3, 5)), torch.zeros(1, 4), atol=1e-7, rtol=0)
        peaks = torch.full((1, 2, 3, 5), -50.)
        peaks[0, 0, 0, 4] = 50.
        peaks[0, 1, 2, 0] = 50.
        torch.testing.assert_close(pool(peaks), torch.tensor([[1., -1., -1., 1.]]))
        features = torch.randn(2, 3, 4, 6, requires_grad=True)
        pool(features).square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                SpatialSoftmax(temperature)

    def test_legacy_average_encoder_config_and_state_dict_remain_compatible(self):
        legacy = {"conv_layers": [[32, 5, 2, 2], [64, 3, 2, 1], [128, 3, 2, 1], [128, 3, 2, 1]],
                  "pool_size": [4, 4], "feature_dim": 256}
        encoder = VisualEncoder(**legacy)
        self.assertEqual(encoder.get_config(), legacy)
        self.assertEqual(encoder.state_dict()["projection.2.weight"].shape, (256, 128 * 4 * 4))
        restored = VisualEncoder(**encoder.get_config())
        restored.load_state_dict(encoder.state_dict(), strict=True)
        image = torch.rand(1, 3, 32, 32)
        torch.testing.assert_close(encoder(image), restored(image), rtol=0, atol=0)

    def test_independent_encoders_and_policy_head(self):
        model = VisualBCPolicy(16, ("agentview", "robot0_eye_in_hand"))
        self.assertEqual([module.out_features for module in model.head if isinstance(module, nn.Linear)], [256, 128, 7])
        self.assertFalse(any(isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.Dropout)) for module in model.modules()))
        for left, right in zip(model.vision.parameters(), model.eye_vision.parameters()):
            self.assertIsNot(left, right)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        output = model(torch.rand(2, 3, 128, 128), torch.randn(2, 16), eye_in_hand_image=torch.rand(2, 3, 128, 128))
        self.assertEqual(output["motion"].shape, (2, 6))
        (output["motion"].square().mean() + output["gripper_logit"].square().mean()).backward()
        self.assertTrue(all(parameter.requires_grad and parameter.grad is not None for parameter in model.parameters()))
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.fill_(0)
            model.head[-1].bias[-1] = -.4
            self.assertEqual(model(torch.zeros(1, 3, 128, 128), torch.zeros(1, 16), eye_in_hand_image=torch.zeros(1, 3, 128, 128))["action"][0, 6], 1)
            model.head[-1].bias[-1] = -.6
            self.assertEqual(model(torch.zeros(1, 3, 128, 128), torch.zeros(1, 16), eye_in_hand_image=torch.zeros(1, 3, 128, 128))["action"][0, 6], -1)

    def test_train_only_statistics_split_and_full_epoch_loss(self):
        config = self.prepare()
        self.assertEqual(config["method"], "visual_bc")
        self.assertEqual(config["normalization"]["normalization_type"], "visual_bc")
        self.assertEqual((config["train_episodes"], config["val_episodes"]), (9, 1))
        normalizer = PolicyNormalizer.from_json(self.prepared / "normalization.json")
        self.assertEqual(normalizer.proprio_dim, 16)
        self.assertTrue(normalizer.validate_split(self.prepared / "train.jsonl"))
        dataset = ManipulationDataset(self.dataset, self.prepared / "train.jsonl",
                                      proprio_keys=config["proprio_keys"], camera_keys=tuple(CAMERA_OBSERVATION_KEYS.values()))
        state, action = [], []
        for entry in dataset.episodes:
            with open_episode(self.dataset / entry["path"]) as archive:
                state.append(np.concatenate([archive[key][:-1] for key in config["proprio_keys"]], axis=1))
                action.append(archive["actions"][:])
        state, action = np.concatenate(state), np.concatenate(action)
        np.testing.assert_allclose(normalizer.proprio_mean, state.mean(0), atol=1e-6)
        np.testing.assert_allclose(normalizer.proprio_std, np.maximum(state.std(0), .001), atol=1e-6)
        np.testing.assert_allclose(normalizer.action_std[:6], np.maximum(action[:, :6].std(0), .01), atol=1e-6)
        self.assertEqual((normalizer.action_mean[6], normalizer.action_std[6]), (0, 1))
        model = VisualBCPolicy(16, config["camera_names"])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        metrics = training.run_epoch(model, DataLoader(dataset, batch_size=128, shuffle=True), normalizer, self.device)
        targets = normalizer.normalize_action(torch.from_numpy(action))
        expected = targets[:, :3].square().mean() + targets[:, 3:6].square().mean() + .2 * math.log(2)
        self.assertAlmostEqual(metrics["loss"], expected.item(), places=5)
        self.assertEqual((metrics["samples"], metrics["batches"]), (len(dataset), 1))

    def test_position_only_mode_and_camera_rejection(self):
        for entry in self.entries:
            path = self.dataset / "pick" / entry["path"]
            with h5py.File(path, "r+") as archive:
                del archive["obs__robot0_eye_in_hand_image"]
                archive["actions"][:, 3:6] = 0
            entry["metadata"]["control_orientation"] = False
        self.manifest.write_text("".join(json.dumps(value) + "\n" for value in self.entries))
        config = self.prepare()
        self.assertEqual(config["proprio_dim"], 16)
        model = VisualBCPolicy(16, config["camera_names"], False)
        output = model(torch.rand(1, 3, 128, 128), torch.zeros(1, 16))
        self.assertEqual(output["motion"].shape, (1, 3))
        self.assertEqual(output["action"][0, 3:6].abs().sum(), 0)
        self.assertEqual(suspicious_camera_frames(np.zeros((2, 32, 32, 3), np.uint8)).tolist(), [0, 1])

    def test_data_mutation_and_split_tampering_are_rejected(self):
        self.prepare()
        load_bc_training(self.dataset, self.prepared)
        split = self.prepared / "train.jsonl"
        original = split.read_bytes()
        split.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "split changed"):
            load_bc_training(self.dataset, self.prepared)
        split.write_bytes(original)
        path = self.dataset / "pick" / self.entries[0]["path"]
        with h5py.File(path, "r+") as archive:
            archive["actions"][0, 0] += .01
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            load_bc_training(self.dataset, self.prepared)

    def test_corrupted_camera_or_normalization_cannot_enter_training(self):
        self.prepare()
        stats = self.prepared / "normalization.json"
        original = stats.read_text()
        value = json.loads(original)
        value["action"]["mean"][0] += 1
        stats.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "normalization file changed"):
            load_bc_training(self.dataset, self.prepared)
        stats.write_text(original)
        path = self.dataset / "pick" / self.entries[0]["path"]
        with h5py.File(path, "r+") as archive:
            archive["obs__agentview_image"][:] = 0
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.prepare()

    def test_pose_audit_accepts_train_val_split(self):
        from scripts import audit_pose_distribution as audit
        self.prepare()
        command = ["audit", "--dataset", str(self.dataset), "--split-dir", str(self.prepared),
                   "--output", str(self.root / "poses")]
        with patch("sys.argv", command), redirect_stdout(io.StringIO()):
            audit.main()
        summary = json.loads((self.root / "poses/summary.json").read_text())
        self.assertEqual(set(summary["splits"]), {"train", "val"})
        self.assertTrue((self.root / "poses/position_xy_and_yaw.png").is_file())

    def test_training_auto_prepare_checkpoint_cpu_load_and_resume(self):
        with patch("sys.argv", self.command()), redirect_stdout(io.StringIO()):
            training.main()
        self.assertTrue((self.prepared / "normalization.json").is_file())
        from scripts.evaluate_visual_bc import load_policy
        loaded, normalizer, checkpoint, size, color = load_policy(SimpleNamespace(
            checkpoint=self.output / "best.pt", normalization=None, image_size=None), self.device)
        self.assertIsInstance(loaded, VisualBCPolicy)
        self.assertEqual((size, color, normalizer.proprio_dim), (128, "red", 16))
        self.assertEqual(checkpoint["history"][0]["train"]["batches"], 1)
        with patch("sys.argv", self.command("--resume", str(self.output / "latest.pt"), "--epochs", "2")), redirect_stdout(io.StringIO()):
            training.main()
        resumed = torch.load(self.output / "latest.pt", map_location="cpu", weights_only=False)
        self.assertEqual((resumed["epoch"], len(resumed["history"])), (2, 2))
        saved_config = (self.output / "config.json").read_bytes()
        with patch("sys.argv", self.command("--resume", str(self.output / "latest.pt"), "--epochs", "2")), self.assertRaises(ValueError):
            training.main()
        self.assertEqual((self.output / "config.json").read_bytes(), saved_config)

    def test_smoke_loader_matches_current_prepared_inputs(self):
        from tests.smoke import test_dataset_loader as smoke

        self.prepare()
        command = ["loader", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                   "--batch-size", "4"]
        output = io.StringIO()
        with patch("sys.argv", command), redirect_stdout(output):
            smoke.main()
        self.assertIn("Dataset loader smoke test: PASS", output.getvalue())
        config, _ = load_bc_training(self.dataset, self.prepared)
        dataset = ManipulationDataset(self.dataset, self.prepared / "train.jsonl",
                                      proprio_keys=config["proprio_keys"])
        sample = dataset[0]
        self.assertEqual(set(sample), {"image", "proprio", "action", "task", "episode_path", "timestep"})
        self.assertNotIn("actions", sample)
        self.assertFalse(any("history" in key for key in sample))
        for removed in ({"mode": "act"}, {"history_size": 8}, {"rgb_history": True}):
            with self.assertRaises(TypeError):
                ManipulationDataset(self.dataset, self.prepared / "train.jsonl", **removed)

    def test_evaluator_rejects_old_visual_policy_before_loading_weights(self):
        from scripts.evaluate_visual_bc import load_policy

        path = self.root / "old.pt"
        for policy_type in ("history_visual_bc", "visual_bc"):
            torch.save({"policy_type": policy_type}, path)
            with self.subTest(policy_type=policy_type), self.assertRaisesRegex(ValueError, "earlier code version"):
                load_policy(SimpleNamespace(checkpoint=path), self.device)

    def test_collection_target_total_and_auto_artifacts(self):
        root = self.root / "collected"
        created = []
        def build(args, objects, receptacles, seed):
            env = FakeEnv("pick", objects, receptacles, seed)
            created.append(env)
            return env, SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "approach"))
        def expert(task, policy, observation, metadata, render):
            observation, _, _, _ = created[-1].step(np.zeros(7, np.float32))
            return observation, {"success": True, "stage": "complete"}
        command = ["record", "--task", "pick", "--num-demos", "4", "--output", str(root),
                   "--training-dir", str(self.root / "collected_training"), "--fixed-target-color", "red"]
        with patch("sys.argv", command), patch.object(recording, "build_task", side_effect=build), patch.object(
            recording, "run_expert", side_effect=expert
        ), redirect_stdout(io.StringIO()):
            recording.main()
            first = len(created)
            recording.main()
            self.assertEqual(len(created), first)
        entries = (root / "pick/manifest.jsonl").read_text().splitlines()
        self.assertEqual(len(entries), 4)
        ready = json.loads((self.root / "collected_training/training.json").read_text())
        self.assertEqual((ready["train_episodes"], ready["val_episodes"]), (3, 1))

    def test_interrupted_collection_prepares_completed_episodes(self):
        root = self.root / "interrupted"
        current = []
        def build(args, objects, receptacles, seed):
            env = FakeEnv("pick", objects, receptacles, seed)
            current.append(env)
            return env, SimpleNamespace(controller=SimpleNamespace(get_phase=lambda: "lift"))
        calls = []
        def expert(task, policy, observation, metadata, render):
            calls.append(1)
            if len(calls) == 4:
                raise KeyboardInterrupt()
            observation, _, _, _ = current[-1].step(np.zeros(7, np.float32))
            return observation, {"success": True, "stage": "complete"}
        command = ["record", "--task", "pick", "--num-demos", "5", "--output", str(root),
                   "--training-dir", str(root / "ready"), "--fixed-target-color", "red"]
        with patch("sys.argv", command), patch.object(recording, "build_task", side_effect=build), patch.object(
            recording, "run_expert", side_effect=expert
        ), redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Only recorded"):
            recording.main()
        config = json.loads((root / "ready/training.json").read_text())
        self.assertEqual(config["train_episodes"] + config["val_episodes"], 3)
        self.assertTrue(current[-1].closed)

    def test_multiseed_evaluation_uses_checkpoint_and_counts_all_attempts(self):
        from scripts.evaluate_visual_bc import evaluate_policy
        config = self.prepare()
        normalizer = PolicyNormalizer.from_statistics(config["normalization"])
        model = VisualBCPolicy(16, config["camera_names"])
        model.eval()
        class Env:
            object_ids = ("cube_0",)
            action_spec = (np.full(7, -1), np.full(7, 1))
            def __init__(self, seed):
                self.seed, self.episode, self.time = seed, -1, 0
                self.closed = False
            def reset(self):
                self.episode += 1
                self.time = 0
                return self.observation()
            def observation(self):
                return {"agentview_image": np.full((32, 32, 3), 100, np.uint8),
                        "robot0_eye_in_hand_image": np.full((32, 32, 3), 120, np.uint8),
                        "robot0_joint_pos": np.zeros(7), "robot0_eef_pos": np.array([0, 0, 1.]),
                        "robot0_gripper_qpos": np.zeros(2), "robot0_eef_quat": np.array([0, 0, 0, 1])}
            def get_task_metadata(self):
                return {"target_object": "cube_0", "target_color": "red"}
            def get_object_position(self, identifier):
                rise = .2 if self.time and not (self.seed == 1000 and self.episode == 0) else 0
                return np.array([0, 0, .83 + rise])
            def step(self, action):
                if self.seed == 1001 and self.episode == 0:
                    raise RuntimeError("injected policy step error")
                self.time += 1
                return self.observation(), 0, False, {}
            def _check_success(self):
                return True
            def close(self):
                self.closed = True
        environments = []
        def build(seed, *unused):
            env = Env(seed)
            environments.append(env)
            return env
        args = SimpleNamespace(checkpoint=Path("best.pt"), max_steps=21, success_hold_steps=None,
                               episodes=2, num_seeds=2, seed_step=1, seed=1000, seed_offset=0,
                               num_cubes=None, render=False, output_dir=self.root / "eval",
                               record_video=False, log_rollout=True, rollout_dir=None, video_fps=20.)
        with patch("scripts.evaluate_visual_bc.build_environment", side_effect=build), redirect_stdout(io.StringIO()):
            summary = evaluate_policy(args, model, normalizer,
                {"data_config": config, "epoch": 1, "best_val_loss": .1}, self.device)
        self.assertEqual(summary["overall_success_rate"], .5)
        self.assertEqual(summary["tasks"][0]["failure_counts"], {"no_lift": 1, "exception": 1})
        self.assertEqual(len((args.output_dir / "pick_attempts.jsonl").read_text().splitlines()), 4)
        self.assertTrue(all(env.closed for env in environments))

    def test_pick_state_machine_without_physics(self):
        class Env:
            action_dim = 7
            target_object_id = "cube_0"
            def __init__(self):
                self.eef = np.array([.1, 0, 1.1], np.float32)
                self.cube = np.array([0, 0, .83], np.float32)
                self.held = False
                self.actions = []
            def resolve_object_id(self, value):
                return value
            def observation(self):
                return {"robot0_eef_pos": self.eef.copy(), "robot0_eef_quat": np.array([0, 0, 0, 1]),
                        "cube_0_pos": self.cube.copy(), "cube_0_quat": np.array([0, 0, 0, 1])}
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
        with redirect_stdout(io.StringIO()):
            _, result = PickExpert(PandaController(env)).run(env.observation())
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["lift_height"], .18)
        self.assertTrue(np.all(np.asarray(env.actions)[-20:, 6] == 1))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
