"""Offline CPU tests for Small CNN BC; no downloads or real simulator rollouts."""

import copy
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

from models.small_cnn_visual_bc import SmallCNNVisualBCPolicy
from models.small_cnn_visual_encoder import SmallCNNVisualEncoder
from scripts import test_spatial_softmax_history as fixtures
from scripts import train_spatial_softmax_visual_bc as training


class SmallCNNTests(unittest.TestCase):
    setUp = fixtures.RGBHistoryTests.setUp

    def args(self, extra=()):
        command = ["trainer", "--dataset", str(self.root), "--train-split", str(self.split),
                   "--val-split", str(self.split), "--normalization", str(self.normalization_path),
                   "--output-dir", str(self.root / "outputs"), "--visual-encoder", "small_cnn",
                   "--no-phase-balanced", "--batches-per-task", "2", "--batch-size", "2",
                   "--image-size", "32", "--dropout", "0", "--no-vertical-flip", *extra]
        with patch("sys.argv", command):
            args = training.parse_args()
        training.validate_spatial_args(args)
        return args

    def test_exact_encoder_and_trainable_single_dual_camera(self):
        encoder = SmallCNNVisualEncoder()
        convolutions = [module for module in encoder.modules() if isinstance(module, nn.Conv2d)]
        self.assertEqual([module.out_channels for module in convolutions], [32, 64, 128, 128])
        self.assertEqual(convolutions[0].kernel_size, (5, 5))
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in encoder.modules()))
        with torch.no_grad():
            self.assertEqual(encoder.convolutions(torch.rand(2, 3, 128, 128)).shape, (2, 128, 8, 8))
            self.assertEqual(encoder(torch.rand(2, 3, 128, 128)).shape, (2, 256))
        for cameras in (("agentview",), ("agentview", "robot0_eye_in_hand")):
            model = SmallCNNVisualBCPolicy(camera_names=cameras, dropout=0)
            self.assertEqual(model.hidden_dims, (1024, 1024))
            self.assertEqual(model.fusion_input_dim, len(cameras) * 256 + 9)
            inputs = {"image": torch.rand(2, 3, 128, 128), "proprio": torch.randn(2, 9)}
            if len(cameras) == 2:
                inputs["eye_in_hand_image"] = torch.rand(2, 3, 128, 128)
            calls = []
            hook = model.visual_encoder.register_forward_hook(lambda *args: calls.append(1))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            before = model.visual_encoder.convolutions[0].weight.detach().clone()
            output = model(**inputs)
            hook.remove()
            self.assertEqual(len(calls), len(cameras))
            self.assertEqual(output["action"].shape, (2, 7))
            output["action"].square().mean().backward()
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.visual_encoder.convolutions[0].weight))

    def test_cli_rejects_resnet_sizes_freeze_and_history(self):
        args = self.args()
        self.assertEqual(args.visual_encoder, "small_cnn")
        for flag in ("--hidden-dim", "--spatial-channels"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args([flag, "64"])
        for flags in (["--pretrained-visual"], ["--unfreeze-layer3-epoch", "1"],
                      ["--history-size", "8"], ["--rgb-history"]):
            with self.assertRaises(ValueError):
                self.args(flags)

    def test_checkpoint_resume_architecture_and_cpu_evaluator(self):
        from models import small_cnn_visual_bc, small_cnn_visual_encoder
        from scripts.evaluate_visual_bc import load_policy

        args = self.args()
        _, _, train, _ = training.build_loaders(args, self.normalizer.proprio_keys)
        inputs, target, _, _ = training.prepare_batch(next(iter(train)), self.normalizer, self.device, 32)
        model = training.build_policy(args, self.normalizer)
        optimizer = training.build_optimizer(args, model)
        optimizer.zero_grad()
        nn.functional.mse_loss(model(**inputs)["action"], target).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(**inputs)["action"]
        payload = training.checkpoint_payload(args, model, optimizer, 2, 4, .2, .5, [], [])
        self.assertFalse(payload["model_config"]["freeze_visual_backbone"])
        path = self.root / "small.pt"
        torch.save(payload, path)
        args.resume = path
        clone = training.build_policy(args, self.normalizer)
        clone_optimizer = training.build_optimizer(args, clone)
        self.assertEqual(training.resume_training(args, clone, clone_optimizer, self.device)[:4],
                         (3, 4, .2, .5))
        self.assertEqual(len(clone_optimizer.state), len(optimizer.state))
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(clone(**inputs)["action"], expected, rtol=0, atol=0)
        # Saved architecture, not current source defaults, reconstructs an old checkpoint.
        with patch.object(small_cnn_visual_bc, "CONTROL_HIDDEN_DIMS", (64,)), patch.object(
            small_cnn_visual_encoder, "FEATURE_DIM", 32
        ), patch("models.visual_encoder.resnet18", side_effect=AssertionError("No ResNet initialization")):
            loaded, _, _, size, color = load_policy(SimpleNamespace(
                checkpoint=path, normalization=self.normalization_path, image_size=None,
            ), self.device)
            self.assertEqual((size, color), (32, "red"))
            with torch.no_grad():
                torch.testing.assert_close(loaded(**inputs)["action"], expected, rtol=0, atol=0)
            changed = training.build_policy(args, self.normalizer)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                training.validate_model_checkpoint(args, changed, payload)
        wrong_type = copy.deepcopy(payload)
        wrong_type["policy_type"] = "spatial_softmax_visual_bc"
        with self.assertRaisesRegex(ValueError, "policy type"):
            training.validate_model_checkpoint(args, clone, wrong_type)

    def test_trainer_periodic_eval_resume_and_weights_only(self):
        args = self.args(["--epochs", "2", "--rollout-eval-every", "2",
                          "--max-train-batches", "1", "--max-val-batches", "1"])
        rollout = {"successful": 1, "attempted": 20, "success_rate": .05,
                   "failure_counts": {"no_lift": 19}}
        with patch.object(training, "parse_args", return_value=args), patch(
            "scripts.evaluate_visual_bc.run_rollout_evaluation", return_value=rollout
        ) as evaluate, redirect_stdout(io.StringIO()):
            training.main()
        evaluate.assert_called_once()
        self.assertIsInstance(evaluate.call_args.args[0], SmallCNNVisualBCPolicy)
        for name in ("best_val.pt", "best_rollout.pt", "latest.pt", "epoch_002.pt"):
            self.assertTrue((args.output_dir / name).is_file(), name)
        args.resume, args.epochs = args.output_dir / "latest.pt", 3
        with patch.object(training, "parse_args", return_value=args), redirect_stdout(io.StringIO()):
            training.main()
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        self.assertEqual((checkpoint["epoch"], checkpoint["global_step"]), (3, 3))
        self.assertEqual(checkpoint["model_config"]["hidden_dims"], [1024, 1024])
        self.assertEqual(checkpoint["history"][-1]["learning_rates"], {"policy": 1e-4})
        saved_config = (args.output_dir / "config.json").read_bytes()
        invalid = copy.copy(args)
        invalid.fixed_target_color = "blue"
        with patch.object(training, "parse_args", return_value=invalid), self.assertRaises(ValueError):
            training.main()
        self.assertEqual((args.output_dir / "config.json").read_bytes(), saved_config)
        fork = copy.copy(args)
        fork.init_checkpoint, fork.resume = args.resume, None
        fork.output_dir, fork.epochs = self.root / "fork", 1
        with patch.object(training, "parse_args", return_value=fork), redirect_stdout(io.StringIO()):
            training.main()
        fresh = torch.load(fork.output_dir / "latest.pt", map_location="cpu", weights_only=False)
        self.assertEqual((fresh["epoch"], fresh["global_step"]), (1, 1))
        self.assertTrue(all(state["step"].item() == 1 for state in fresh["optimizer_state"]["state"].values()))

    def test_evaluator_executes_denormalized_clipped_actions_and_logging(self):
        from scripts.evaluate_visual_bc import evaluate_episode

        model = training.build_policy(self.args(), self.normalizer)
        with torch.no_grad():
            model.action_head.weight.zero_()
            model.action_head.bias.fill_(100)
        model.eval()
        data = self.episodes[0]

        class FakeEnv:
            object_ids = ("cube_0",)
            action_spec = (np.full(7, -1), np.full(7, 1))

            def observation(self):
                return {key.removeprefix("obs__"): value[self.time]
                        for key, value in data.items() if key.startswith("obs__")}

            def reset(self):
                self.time = 0
                return self.observation()

            def get_task_metadata(self):
                return {"target_object": "cube_0", "target_color": "red"}

            def get_object_position(self, object_id):
                return np.array([0, 0, .83])

            def step(self, action):
                np.testing.assert_array_equal(action, np.ones(7))
                self.time += 1
                return self.observation(), 0, False, {}

            def _check_success(self):
                return False

        log = self.root / "rollout.jsonl"
        result = evaluate_episode(FakeEnv(), model, self.normalizer, 32, self.device,
                                  3, 5, False, 0, None, 20, "red", rollout_path=log)
        self.assertEqual((result["steps"], result["failure_reason"]), (3, "no_lift"))
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(sum(row["event"] == "step" for row in rows), 3)
        np.testing.assert_allclose(rows[1]["raw_action"], [200.3] * 7, rtol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
