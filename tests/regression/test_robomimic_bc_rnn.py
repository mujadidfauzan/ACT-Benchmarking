"""Reference-recipe contracts, native episode IO, resume, and shared task evaluation."""

import copy
import hashlib
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn

from models.robomimic_bc_rnn import BCRNNPolicy
from models.robomimic_bc_rnn.dataset import BCRNNSequenceDataset
from models.robomimic_bc_rnn.policy import ImagePreprocessor
from scripts import train_robomimic_bc_rnn as training, evaluate_robomimic_bc_rnn as evaluation
from tests.regression import test_visual_bc as fixtures
from tests.regression.test_variable_scene_counts import FakeEnv


class TinyEncoder(nn.Module):
    """Only training orchestration tests substitute the expensive ResNet."""

    def __init__(self, num_keypoints=32, feature_dim=64, image_size=84, crop_size=76):
        super().__init__()
        self.preprocess = ImagePreprocessor(image_size, crop_size)
        self.projection = nn.Linear(3, feature_dim)

    def forward(self, frames):
        return self.projection(self.preprocess(frames).mean((-2, -1))).relu()


class BCRNNTests(unittest.TestCase):
    def setUp(self):
        fixtures.VisualBCTests.setUp(self)
        # The old generic fixture deliberately allows unbounded synthetic motion;
        # the reference tanh actor's contract is raw actions in [-1,1].
        for path in (self.dataset / "pick").glob("*.hdf5"):
            with h5py.File(path, "r+") as episode:
                episode["actions"][:] = np.clip(episode["actions"][:], -1, 1)

    def command(self, output=None, *extra):
        return ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                "--output-dir", str(output or self.output), "--device", "cpu",
                "--batch-size", "2", "--hidden-dim", "16", "--num-layers", "1",
                "--sequence-length", "3", "--checkpoint-every", "2",
                "--validation-batches", "2", "--rollout-eval-every", "0", *extra]

    def test_native_windows_alignment_padding_and_raw_proprio(self):
        config = fixtures.VisualBCTests.prepare(self)
        dataset = BCRNNSequenceDataset(self.dataset, self.prepared / "train.jsonl", config["camera_names"], 10)
        self.addCleanup(dataset.close)
        self.assertEqual(len(dataset), config["normalization"]["training_timesteps"])
        index = 1
        episode_index, start = dataset._sample_index[index]
        with dataset._episode(episode_index) as episode:
            expected_action = torch.from_numpy(episode["actions"][start:])
            expected_proprio = torch.from_numpy(np.concatenate([episode[key][start:-1] for key in dataset.proprio_keys], -1)).float()
        sample = dataset[index]
        self.assertEqual(set(sample), {"image", "eye_in_hand_image", "proprio", "action"})
        self.assertEqual(sample["proprio"].shape, (10, 9))
        self.assertEqual(sample["image"].dtype, torch.uint8)
        torch.testing.assert_close(sample["action"][:len(expected_action)], expected_action)
        torch.testing.assert_close(sample["proprio"][:len(expected_proprio)], expected_proprio)
        torch.testing.assert_close(sample["action"][len(expected_action):], expected_action[-1:].expand(10 - len(expected_action), 7))
        dataset.close()
        with h5py.File(self.dataset / dataset.episodes[episode_index]["path"], "r+") as episode:
            episode["actions"][start, 0] = 2.
        with self.assertRaisesRegex(ValueError, "controller actions"):
            dataset[index]

    def test_crop_modes_and_uint8_scaling(self):
        preprocess = ImagePreprocessor()
        frames = torch.arange(84 * 84, dtype=torch.float32).reshape(1, 1, 84, 84).expand(8, 3, -1, -1) / (84 * 84)
        preprocess.eval()
        torch.testing.assert_close(preprocess(frames), frames[..., 4:80, 4:80], rtol=0, atol=0)
        pixels = torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8)
        torch.testing.assert_close(preprocess(pixels), preprocess(pixels.float() / 255), rtol=0, atol=0)
        preprocess.train()
        torch.manual_seed(3)
        crops = preprocess(frames)
        self.assertEqual(crops.shape, (8, 3, 76, 76))
        self.assertFalse(torch.equal(crops[0], crops[1]))
        torch.manual_seed(3)
        torch.testing.assert_close(crops, preprocess(frames), rtol=0, atol=0)

    def test_real_resnet_keypoints_lstm_tanh_and_gradients(self):
        model = BCRNNPolicy(hidden_dim=16)
        self.assertIsInstance(model.lstm, nn.LSTM)
        self.assertEqual(model.lstm.num_layers, 2)
        self.assertEqual(model.lstm.input_size, 64 * 2 + 9)
        self.assertEqual(model.encoders["agentview"].keypoints.weight.shape, (32, 512, 1, 1))
        self.assertEqual(model.encoders["agentview"].projection.in_features, 64)
        self.assertIsNot(model.encoders["agentview"], model.encoders["robot0_eye_in_hand"])
        captured = []
        hook = model.lstm.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
        inputs = dict(image=torch.randint(0, 256, (1, 2, 3, 128, 128), dtype=torch.uint8),
                      eye_in_hand_image=torch.rand(1, 2, 3, 128, 128), proprio=torch.randn(1, 2, 9))
        result = model(**inputs)
        hook.remove()
        self.assertEqual(result["action"].shape, (1, 2, 7))
        self.assertTrue((result["action"].abs() <= 1).all())
        self.assertTrue((captured[0][..., :128] >= 0).all())
        torch.testing.assert_close(captured[0][..., 128:], inputs["proprio"])
        result["action"].square().mean().backward()
        for encoder in model.encoders.values():
            self.assertGreater(encoder.keypoints.weight.grad.abs().sum().item(), 0)
            self.assertGreater(encoder.backbone[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.lstm.weight_ih_l0.grad.abs().sum().item(), 0)

    def test_seven_component_mse_and_gripper_gradient(self):
        prediction = torch.zeros(2, 3, 7, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[..., 6] = 1
        loss = training.action_losses(prediction, target)
        self.assertAlmostEqual(loss["loss"].item(), 1 / 7)
        self.assertEqual(loss["gripper_mse"].item(), 1.)
        loss["loss"].backward()
        torch.testing.assert_close(prediction.grad[..., 6], torch.full((2, 3), -2 / 42))
        self.assertEqual(prediction.grad[..., :6].abs().sum().item(), 0)

    def test_streaming_matches_reference_horizon_and_checkpoint_roundtrip(self):
        model = BCRNNPolicy(("agentview",), hidden_dim=16, sequence_length=3).eval()
        image, proprio = torch.rand(1, 8, 3, 84, 84), torch.randn(1, 8, 9)
        with torch.inference_mode():
            expected = torch.cat([model(image[:, start:start + 3], proprio[:, start:start + 3])["action"]
                                  for start in range(0, 8, 3)], dim=1)
            hidden, actions = None, []
            for i in range(8):
                output = model.step(image[:, i], proprio[:, i], hidden=hidden)
                self.assertEqual(output["hidden_state_reset"], i % 3 == 0)
                hidden = output["hidden"]
                actions.append(output["action"])
            torch.testing.assert_close(torch.stack(actions, 1), expected, atol=1e-6, rtol=1e-5)
            opened = model.step(image[:, 0], proprio[:, 0], previous_gripper=-1)["action"]
            closed = model.step(image[:, 0], proprio[:, 0], previous_gripper=1)["action"]
            torch.testing.assert_close(opened, closed, rtol=0, atol=0)
        path = self.root / "policy.pt"
        torch.save(dict(policy_type=model.policy_type, model_config=model.get_config(), model_state=model.state_dict()), path)
        restored, _ = evaluation.load_policy(path, self.device)
        with torch.inference_mode():
            torch.testing.assert_close(restored(image, proprio)["action"], model(image, proprio)["action"], rtol=0, atol=0)
        torch.save(dict(policy_type="visual_bc"), path)
        with self.assertRaisesRegex(ValueError, "robomimic_bc_rnn"):
            evaluation.load_policy(path, self.device)

    def test_sampler_resume_across_shuffle_boundary(self):
        sampler = training.WindowSampler(11, 4, 3)
        first = sampler.next_indices()
        state = copy.deepcopy(sampler.state_dict())
        expected = [sampler.next_indices() for _ in range(5)]
        resumed = training.WindowSampler(11, 4, 999)
        resumed.load_state_dict(state)
        self.assertEqual([resumed.next_indices() for _ in range(5)], expected)
        self.assertFalse(set(first) & set(expected[0]))

    def test_training_resume_matches_uninterrupted_and_keeps_dataset(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.dataset.rglob("*") if p.is_file()}
        full, resumed = self.output / "full", self.output / "resume"
        with patch("models.robomimic_bc_rnn.policy.KeypointEncoder", TinyEncoder), redirect_stdout(io.StringIO()):
            for command in (self.command(full, "--max-updates", "4"),
                            self.command(resumed, "--max-updates", "2"),
                            self.command(resumed, "--resume", str(resumed / "latest.pt"), "--max-updates", "4")):
                with patch("sys.argv", command):
                    training.main()
        a = torch.load(full / "latest.pt", map_location="cpu", weights_only=False)
        b = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
        self.assertEqual(a["updates"], 4)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["optimizer_state"]["param_groups"][0]["weight_decay"], 0.)
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        command = self.command(resumed, "--resume", str(resumed / "latest.pt"), "--max-updates", "6", "--batch-size", "4")
        with patch("sys.argv", command), self.assertRaisesRegex(ValueError, "batch-size"):
            training.main()

    def test_periodic_rollout_and_native_evaluator_diagnostics(self):
        command = self.command(None, "--max-updates", "2", "--rollout-eval-every", "2")
        with patch("models.robomimic_bc_rnn.policy.KeypointEncoder", TinyEncoder), patch("sys.argv", command), \
                patch.object(training, "run_rollout", return_value=dict(updates=2, success_rate=.2)) as rollout, redirect_stdout(io.StringIO()):
            training.main()
        rollout.assert_called_once()
        checkpoint = self.output / "best_rollout.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.assertEqual(saved["best_rollout_success_rate"], .2)
        self.assertEqual(saved["rollout_history"], [dict(updates=2, success_rate=.2)])
        created = []

        def build(seed, *args):
            env = FakeEnv("pick", 1, None, seed)
            env.object_ids = ["object_0"]
            env.action_spec = (-np.ones(7), np.ones(7))
            env.get_object_position = lambda *args: np.array([0., 0., .83])
            env._check_success = lambda: False
            created.append(env)
            return env

        output = self.root / "eval"
        command = ["eval", "--checkpoint", str(checkpoint), "--output-dir", str(output), "--device", "cpu",
                   "--episodes", "2", "--num-seeds", "1", "--max-steps", "12", "--log-rollout"]
        with patch("models.robomimic_bc_rnn.policy.KeypointEncoder", TinyEncoder), patch("sys.argv", command), \
                patch("scripts.evaluate_visual_bc.build_environment", side_effect=build), redirect_stdout(io.StringIO()):
            summary = evaluation.main()
        self.assertEqual(summary["overall_success_rate"], 0.)
        self.assertEqual(summary["tasks"][0]["failure_counts"], {"no_lift": 2})
        self.assertEqual(summary["rnn_reset_interval"], 3)
        self.assertEqual(summary["checkpoint_updates"], 2)
        self.assertEqual(summary["image_size"], 84)
        self.assertTrue(all(e.closed for e in created))
        for path in (output / "rollouts").glob("*.jsonl"):
            steps = [r for line in path.read_text().splitlines() if (r := json.loads(line)).get("event") == "step"]
            self.assertEqual([r["step"] for r in steps if r["hidden_state_reset"]], [1, 4, 7, 10])
            self.assertTrue(all(r["gripper_logit"] is None and r["previous_gripper_input"] is None for r in steps))
            self.assertTrue(all(r["normalized_action"] == r["raw_action"] for r in steps))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
