"""ACT observation/action alignment, posterior isolation, chunks, and resume."""

import copy
import io
import json
import math
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from data.normalization import PolicyNormalizer
from models.act import ACTPolicy
from models.act.policy import AttentionBlock
from models.act.dataset import ACTDataset
from models.act.temporal_ensemble import ChunkState
from scripts import train_act as training, evaluate_act as evaluation
from scripts import evaluate_visual_bc_validation as validation
from tests.regression import test_visual_bc as fixtures


class TinyBackbone(nn.Module):
    """Substitute vision only in orchestration tests; Transformer stays real."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Conv2d(3, 512, 1))

    def forward(self, image):
        return self.net(image)


class ACTTests(unittest.TestCase):
    def setUp(self):
        fixtures.VisualBCTests.setUp(self)
        for path in (self.dataset / "pick").glob("*.hdf5"):
            with h5py.File(path, "r+") as episode:
                episode["actions"][:] = np.clip(episode["actions"][:], -1, 1)

    def prepare(self):
        return fixtures.VisualBCTests.prepare(self)

    def command(self, output=None, *extra):
        return ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                "--output-dir", str(output or self.output), "--device", "cpu", "--batch-size", "2",
                "--hidden-dim", "32", "--num-heads", "4", "--encoder-layers", "1",
                "--decoder-layers", "1", "--feedforward-dim", "64", "--latent-dim", "4",
                "--image-size", "32", "--chunk-size", "3", "--checkpoint-every", "2",
                "--validation-batches", "2", "--rollout-eval-every", "0", *extra]

    def policy(self, **extra):
        config = dict(proprio_dim=16, chunk_size=3, hidden_dim=32, num_heads=4,
                      encoder_layers=1, decoder_layers=1, feedforward_dim=64,
                      latent_dim=4, dropout=0., image_size=32)
        return ACTPolicy(**{**config, **extra})

    def test_native_current_observation_future_targets_and_episode_boundary(self):
        prepared = self.prepare()
        dataset = ACTDataset(self.dataset, self.prepared / "train.jsonl", prepared["camera_names"],
                             prepared["normalization"]["proprio"]["keys"], chunk_size=10)
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), prepared["normalization"]["training_timesteps"])
        episode_index, start = dataset._sample_index[1]
        sample = dataset[1]
        with dataset._episode(episode_index) as episode:
            expected = torch.from_numpy(episode["actions"][start:])
            state = np.concatenate([episode[k][start] for k in dataset.proprio_keys])
            image = torch.from_numpy(episode["obs__agentview_image"][start].transpose(2, 0, 1).copy())
        self.assertEqual(sample["image"].shape, (3, 32, 32))
        self.assertEqual(sample["eye_in_hand_image"].dtype, torch.uint8)
        torch.testing.assert_close(sample["proprio"], torch.tensor(state).float())
        torch.testing.assert_close(sample["image"], image)
        torch.testing.assert_close(sample["action"][:len(expected)], expected)
        self.assertFalse(sample["action"][len(expected):].any())
        self.assertEqual(sample["is_pad"].tolist(), [False] * len(expected) + [True] * (10 - len(expected)))
        self.assertEqual(set(sample), {"image", "eye_in_hand_image", "proprio", "action", "is_pad"})

    def test_masked_loss_denominator_components_and_kl(self):
        prediction = torch.zeros(2, 3, 7, requires_grad=True)
        target = torch.ones_like(prediction)
        padding = torch.tensor([[False, True, True], [False, False, True]])
        target[padding] = float("nan")
        mu, logvar = torch.ones(2, 4), torch.zeros(2, 4)
        losses = training.action_losses(prediction, target, padding, mu, logvar, kl_weight=2.)
        self.assertEqual(losses["l1"].item(), 1.)
        self.assertEqual(losses["kl"].item(), 2.)
        self.assertEqual(losses["loss"].item(), 5.)
        losses["loss"].backward()
        self.assertFalse(prediction.grad[padding].any())
        self.assertAlmostEqual(prediction.grad[~padding].sum().item(), -1., places=6)
        with self.assertRaisesRegex(ValueError, "No valid"):
            training.action_losses(prediction, target, torch.ones_like(padding))

    def test_explicit_attention_matches_sdpa_with_heavy_padding_and_finite_backward(self):
        block = AttentionBlock(32, 4, 64, 0.)
        value = torch.randn(2, 34, 32, requires_grad=True)
        position = torch.randn(1, 34, 32)
        padding = torch.ones(2, 34, dtype=torch.bool)
        padding[:, :3] = False  # CLS + proprio + one real action.
        query = value + position
        attention = block.self_attention(query, query, value, key_padding_mask=padding,
                                         need_weights=False)[0]
        expected = block.norms[0](value + attention)
        expected = block.norms[-1](expected + block.feedforward(expected))
        with patch("torch.nn.functional.scaled_dot_product_attention",
                   side_effect=AssertionError("Fused SDPA must not run")):
            actual = block(value, position, padding=padding)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            actual.square().mean().backward()
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_clipping_recovers_norm_overflow_but_rejects_invalid_components(self):
        layer = nn.Linear(2, 1, bias=False)
        layer.weight.grad = torch.full_like(layer.weight, 1e20)
        norm = training.clip_act_gradients(layer, 1.)
        self.assertTrue(torch.isfinite(norm))
        self.assertGreater(norm.item(), 1e20)
        self.assertAlmostEqual(layer.weight.grad.double().norm().item(), 1., places=5)
        for bad in (float("nan"), float("inf")):
            layer.weight.grad = torch.full_like(layer.weight, bad)
            with self.assertRaisesRegex(FloatingPointError, "weight"):
                training.clip_act_gradients(layer, 1.)

    def test_nonfinite_backward_writes_batch_diagnostic_without_optimizer_step(self):
        prepared = self.prepare()
        normalizer = PolicyNormalizer.from_statistics(prepared["normalization"])
        dataset = ACTDataset(self.dataset, self.prepared / "train.jsonl", prepared["camera_names"],
                             normalizer.proprio_keys, chunk_size=3)
        self.addCleanup(dataset.close)
        with patch("models.act.policy.VisualBackbone", TinyBackbone):
            model = self.policy()
        original = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters())
        hook = model.action_head.weight.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        self.addCleanup(hook.remove)
        with self.assertRaisesRegex(FloatingPointError, "action_head.weight.*update=1051"):
            training.run_batches(model, normalizer, dataset, [[0, 1]], self.device,
                                 kl_weight=10., optimizer=optimizer, start_update=1050,
                                 failure_dir=self.output / "diagnostics")
        report = json.loads((self.output / "diagnostics/nonfinite_update_0001051.json").read_text())
        self.assertEqual(report["nonfinite_gradients"][0]["parameter"], "action_head.weight")
        self.assertFalse(report["optimizer_step_executed"])
        self.assertEqual([s["index"] for s in report["samples"]], [0, 1])
        self.assertTrue(math.isfinite(report["losses"]["loss"]))
        self.assertFalse(optimizer.state)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(original[key], value, atol=0, rtol=0)

    def test_real_resnet_transformer_cvae_backward_and_zero_prior(self):
        model = self.policy().eval()
        image = torch.rand(2, 3, 32, 32, requires_grad=True)
        eye = torch.rand(2, 3, 32, 32, requires_grad=True)
        proprio = torch.rand(2, 16)
        actions = torch.rand(2, 3, 7)
        padding = torch.tensor([[False, False, True], [False, True, True]])
        output = model(image, proprio, eye_in_hand_image=eye, actions=actions, is_pad=padding, sample_posterior=False)
        self.assertEqual(output["action"].shape, (2, 3, 7))
        training.action_losses(output["action"], actions, padding, output["mu"], output["logvar"])["loss"].backward()
        for name in ("backbone", "latent_encoder", "encoder", "decoder", "action_head"):
            gradients = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            self.assertTrue(gradients, name)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), name)
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0., name)
        self.assertGreater(image.grad.abs().sum().item(), 0.)
        self.assertGreater(eye.grad.abs().sum().item(), 0.)
        changed = actions.clone()
        changed[padding] = 10000
        with torch.inference_mode():
            masked = model(image, proprio, eye_in_hand_image=eye, actions=changed, is_pad=padding, sample_posterior=False)
            torch.testing.assert_close(masked["mu"], output["mu"], rtol=0, atol=0)
            torch.testing.assert_close(masked["action"], output["action"], rtol=0, atol=0)
            prior1 = model(image, proprio, eye_in_hand_image=eye)
            torch.manual_seed(82)
            prior2 = model(image, proprio, eye_in_hand_image=eye)
            self.assertIsNone(prior1["mu"])
            torch.testing.assert_close(prior1["action"], prior2["action"], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "history dimension"):
            model(image[:, None], proprio, eye_in_hand_image=eye[:, None])

    def test_temporal_ensemble_time_alignment_zeros_decay_and_bounded_state(self):
        state = ChunkState()
        chunk = torch.tensor([[[0.], [2.], [4.]]])
        state.add(chunk)
        self.assertEqual(state.current(ensemble=True, decay=0).item(), 0.)
        state.add(torch.tensor([[[10.], [20.], [30.]]]))
        self.assertEqual(state.current(ensemble=True, decay=0).item(), 6.)  # (2 + 10) / 2
        state.add(torch.tensor([[[100.], [200.], [300.]]]))
        expected = (4 * math.exp(-2) + 20 * math.exp(-1) + 100) / (math.exp(-2) + math.exp(-1) + 1)
        self.assertAlmostEqual(state.current(ensemble=True, decay=1.).item(), expected, places=5)
        for _ in range(20):
            state.add(torch.zeros(1, 3, 1))
            state.current(ensemble=True)
            self.assertLessEqual(len(state.predictions), 3)

    def test_replanning_and_episode_reset_do_not_leak_chunk_state(self):
        with patch("models.act.policy.VisualBackbone", TinyBackbone):
            model = self.policy(execution_steps=2).eval()
        image, proprio = torch.zeros(1, 3, 32, 32), torch.zeros(1, 16)
        chunks = [torch.full((1, 3, 7), float(i)) + torch.arange(3)[None, :, None] for i in (10, 20, 30)]
        with patch.object(model, "forward", side_effect=[dict(action=c) for c in chunks]) as forward:
            hidden, values = None, []
            for _ in range(5):
                result = model.step(image, proprio, eye_in_hand_image=image, hidden=hidden)
                hidden = result["hidden"]
                values.append(result["action"][0, 0].item())
            self.assertEqual(values, [10, 11, 20, 21, 30])
            self.assertEqual(forward.call_count, 3)
        with patch.object(model, "forward", return_value=dict(action=chunks[0])):
            reset = model.step(image, proprio, eye_in_hand_image=image)
            self.assertEqual(reset["action"][0, 0].item(), 10)
            self.assertEqual(reset["hidden"].time, 1)
        with self.assertRaisesRegex(ValueError, "every step"):
            model.configure_inference(2, True)

    def test_training_resume_is_bitwise_equal_with_dropout_and_posterior_rng(self):
        self.prepare()
        continuous = self.root / "continuous"
        with patch("models.act.policy.VisualBackbone", TinyBackbone), redirect_stdout(io.StringIO()):
            with patch("sys.argv", self.command(continuous, "--max-updates", "4")):
                training.main()
            with patch("sys.argv", self.command(self.output, "--max-updates", "2")):
                training.main()
            resume = ["resume", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                      "--output-dir", str(self.output), "--resume", str(self.output / "latest.pt"),
                      "--max-updates", "4", "--device", "cpu"]
            with patch("sys.argv", resume):
                training.main()
            a = torch.load(continuous / "latest.pt", weights_only=False)
            b = torch.load(self.output / "latest.pt", weights_only=False)
            self.assertEqual(a["history"], b["history"])
            self.assertEqual(b["updates"], 4)
            self.assertEqual(b["best_val_metric"], "prior_l1")
            for key, value in a["model_state"].items():
                torch.testing.assert_close(value, b["model_state"][key], rtol=0, atol=0)
            self.assertTrue(all("prior_l1" in r["val"] and "kl" in r["train"] for r in b["history"]))
            model, normalizer, payload = evaluation.load_policy(self.output / "latest.pt", self.device)
            config = json.loads((self.output / "config.json").read_text())
            self.assertEqual(config["data_config"], b["data_config"])
            self.assertEqual(normalizer.proprio_dim, 16)
            self.assertFalse(model.training)
            loaded = validation.load_validation_policy(self.output / "latest.pt", self.device, 16)
            self.assertEqual(loaded[-1], 0)
            self.assertEqual(loaded[3], 32)
            self.assertEqual(loaded[2]["policy_type"], "act")
            with patch("sys.argv", resume + ["--batch-size", "3"]), self.assertRaisesRegex(ValueError, "differs"):
                training.main()

    def test_periodic_rollout_checkpoint_selection(self):
        self.prepare()
        with patch("models.act.policy.VisualBackbone", TinyBackbone), redirect_stdout(io.StringIO()), \
                patch.object(training, "run_rollout", side_effect=[
                    dict(updates=2, success_rate=.8, summary="one"),
                    dict(updates=4, success_rate=.4, summary="two")]) as rollout, \
                patch("sys.argv", self.command(None, "--max-updates", "4", "--rollout-eval-every", "2")):
            training.main()
        self.assertEqual(rollout.call_count, 2)
        best = torch.load(self.output / "best_rollout.pt", weights_only=False)
        latest = torch.load(self.output / "latest.pt", weights_only=False)
        self.assertEqual(best["updates"], 2)
        self.assertEqual(latest["best_rollout_success_rate"], .8)
        self.assertEqual(len(latest["rollout_history"]), 2)

    def test_resume_recovers_failed_final_rollout_without_retraining(self):
        self.prepare()
        with patch("models.act.policy.VisualBackbone", TinyBackbone), redirect_stdout(io.StringIO()):
            with patch.object(training, "run_rollout", side_effect=RuntimeError("injected evaluator failure")), \
                    patch("sys.argv", self.command(None, "--max-updates", "2", "--rollout-eval-every", "2")), \
                    self.assertRaisesRegex(RuntimeError, "injected evaluator"):
                training.main()
            before = torch.load(self.output / "latest.pt", weights_only=False)
            self.assertEqual(before["updates"], 2)
            self.assertEqual(before["rollout_history"], [])
            self.assertTrue((self.output / "best.pt").is_file())
            resume = ["resume", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                      "--output-dir", str(self.output), "--resume", str(self.output / "latest.pt"),
                      "--max-updates", "2", "--device", "cpu"]
            with patch("sys.argv", resume), patch.object(training, "run_rollout", return_value=
                    dict(updates=2, success_rate=.6, summary="recovered")) as rollout:
                training.main()
            rollout.assert_called_once()
        after = torch.load(self.output / "latest.pt", weights_only=False)
        self.assertEqual(before["history"], after["history"])
        self.assertEqual(after["best_rollout_success_rate"], .6)
        self.assertEqual(after["rollout_history"], [dict(updates=2, success_rate=.6, summary="recovered")])
        for key in before["model_state"]:
            torch.testing.assert_close(before["model_state"][key], after["model_state"][key], rtol=0, atol=0)

    def test_rollout_retry_preserves_partial_artifacts(self):
        partial = self.output / "rollouts/update_0000002/place_attempts.jsonl"
        partial.parent.mkdir(parents=True)
        partial.write_text("original partial output\n")
        args = SimpleNamespace(output_dir=self.output, rollout_eval_episodes=10,
                               rollout_eval_num_seeds=1, rollout_eval_seed=1000, rollout_eval_max_steps=300)

        def evaluate(command, **kwargs):
            directory = self.root / "run/rollouts/update_0000002_retry01"
            self.assertEqual(command[command.index("--output-dir") + 1], str(directory))
            directory.mkdir()
            (directory / "summary.json").write_text(json.dumps(
                dict(overall_success_rate=.7, tasks=[dict(failure_counts={})])))

        with patch.object(training.subprocess, "run", side_effect=evaluate):
            result = training.run_rollout(args, 2, self.device)
        self.assertEqual(result["success_rate"], .7)
        self.assertIn("retry01", result["summary"])
        self.assertEqual(partial.read_text(), "original partial output\n")

    def test_eval_uses_shared_physics_normalization_and_continuous_gripper(self):
        self.prepare()
        with patch("models.act.policy.VisualBackbone", TinyBackbone), redirect_stdout(io.StringIO()), \
                patch("sys.argv", self.command(None, "--max-updates", "2")):
            training.main()
        output = self.root / "evaluation"
        args = ["eval", "--checkpoint", str(self.output / "latest.pt"), "--output-dir", str(output),
                "--device", "cpu", "--episodes", "1", "--num-seeds", "1", "--max-steps", "3", "--log-rollout"]

        class MinimalEnv:
            object_ids = ("cube_0",)
            action_spec = (np.full(7, -1.0), np.full(7, 1.0))

            def reset(self):
                return {"agentview_image": np.full((32, 32, 3), 80, np.uint8),
                        "robot0_eye_in_hand_image": np.full((32, 32, 3), 80, np.uint8),
                        "robot0_joint_pos": np.zeros(7, np.float32),
                        "robot0_eef_pos": np.zeros(3, np.float32),
                        "robot0_gripper_qpos": np.zeros(2, np.float32),
                        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], np.float32)}

            def get_task_metadata(self):
                return {"target_object": "cube_0", "target_color": "red", "task": "pick"}

            def get_object_position(self, identifier):
                return np.array([0.0, 0.0, 0.83])

            def step(self, action):
                return self.reset(), 0, False, {}

            def _check_success(self):
                return False

            def close(self):
                pass

        with patch("models.act.policy.VisualBackbone", TinyBackbone), redirect_stdout(io.StringIO()), \
                patch("scripts.evaluate_visual_bc.build_environment",
                      side_effect=lambda *a, **kw: MinimalEnv()), \
                patch("sys.argv", args):
            summary = evaluation.main()
        self.assertEqual(summary["policy_type"], "act")
        self.assertIsNone(summary["checkpoint_epoch"])
        self.assertEqual(summary["latent_at_inference"], "zero")
        rows = [json.loads(s) for s in next((output / "rollouts").glob("*.jsonl")).read_text().splitlines()]
        self.assertIsNone(rows[1]["gripper_logit"])
        self.assertIsNone(rows[1]["previous_gripper_input"])
        checkpoint = torch.load(self.output / "latest.pt", weights_only=False)
        normalizer = PolicyNormalizer.from_statistics(checkpoint["data_config"]["normalization"])
        expected = normalizer.denormalize_action(torch.tensor(rows[1]["normalized_action"]))
        torch.testing.assert_close(torch.tensor(rows[1]["raw_action"]), expected)



if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
