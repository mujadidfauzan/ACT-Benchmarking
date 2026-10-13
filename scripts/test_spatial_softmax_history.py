"""Offline CPU regression tests; synthetic data, no downloads or simulator runs."""

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
import torch
from torch.utils.data import default_collate

from data.manipulation_dataset import (
    CAMERA_OBSERVATION_KEYS, EEF_GRIPPER_PROPRIO_KEYS, ManipulationDataset,
)
from data.normalization import PolicyNormalizer
from data.policy_history import PolicyHistory
from models.history_spatial_softmax_visual_bc import HistorySpatialSoftmaxVisualBCPolicy
from models.spatial_softmax_visual_bc import SpatialSoftmaxVisualBCPolicy
from scripts.train_spatial_softmax_visual_bc import (
    build_loaders, checkpoint_payload, parse_args, prepare_batch, resume_training,
    run_epoch, validate_spatial_args,
)


class RGBHistoryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.device = torch.device("cpu")
        self.episodes = []
        entries = []
        for episode_index in range(2):
            steps = 10
            rng = np.random.default_rng(episode_index)
            episode = {"actions": np.arange(steps * 7, dtype=np.float32).reshape(steps, 7) / 70}
            for key in CAMERA_OBSERVATION_KEYS.values():
                episode[key] = rng.integers(0, 256, (steps + 1, 32, 32, 3), dtype=np.uint8)
            for key, dimension in zip(EEF_GRIPPER_PROPRIO_KEYS, (3, 4, 2)):
                episode[key] = np.full((steps + 1, dimension), episode_index, dtype=np.float32)
                episode[key] += np.arange(steps + 1, dtype=np.float32)[:, None] / 10
            episode["phases"] = np.asarray(["approach", "pre_grasp", "descend", "close", "lift"] * 2)
            filename = f"episode_{episode_index}.npz"
            np.savez(self.root / filename, **episode)
            self.episodes.append(episode)
            entries.append({"path": filename, "task": "pick", "steps": steps,
                            "instruction": "Pick up the red cube."})
        self.split = self.root / "train.jsonl"
        self.split.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
        self.normalizer = PolicyNormalizer(
            np.full(9, 0.25), np.full(9, 2), np.full(7, 0.3), np.full(7, 2),
            proprio_keys=EEF_GRIPPER_PROPRIO_KEYS,
        )
        self.normalization_path = self.root / "normalization.json"
        self.normalization_path.write_text(json.dumps({
            "split_sha256": hashlib.sha256(self.split.read_bytes()).hexdigest(),
            "task_balanced_proprio": {"keys": EEF_GRIPPER_PROPRIO_KEYS,
                                      "mean": [0.25] * 9, "std": [2] * 9},
            "task_balanced_action": {"mean": [0.3] * 7, "std": [2] * 7},
        }), encoding="utf-8")

    def dataset(self, **kwargs):
        options = dict(dataset_root=self.root, split_manifest=self.split, mode="bc",
                       camera_keys=tuple(CAMERA_OBSERVATION_KEYS.values()),
                       proprio_keys=EEF_GRIPPER_PROPRIO_KEYS, history_size=8,
                       rgb_history=True)
        options.update(kwargs)
        return ManipulationDataset(**options)

    def model(self, **kwargs):
        options = dict(camera_names=tuple(CAMERA_OBSERVATION_KEYS), history_size=8,
                       pretrained_visual=False, dropout=0)
        options.update(kwargs)
        return HistorySpatialSoftmaxVisualBCPolicy(**options)

    def args(self, extra=()):
        command = ["trainer", "--dataset", str(self.root), "--train-split", str(self.split),
                   "--val-split", str(self.split), "--normalization", str(self.normalization_path),
                   "--output-dir", str(self.root / "outputs"), "--history-size", "8",
                   "--rgb-history", "--no-phase-balanced", "--batches-per-task", "2",
                   "--batch-size", "2", "--image-size", "32", "--dropout", "0",
                   "--no-vertical-flip", *extra]
        with patch("sys.argv", command):
            args = parse_args()
        validate_spatial_args(args)
        return args

    def test_causality_padding_and_episode_boundaries(self):
        dataset = self.dataset()
        for index in (0, 1, 7, 9, 10, 19):
            sample = dataset[index]
            timestep = index % 10
            count = min(8, timestep + 1)
            self.assertEqual(sample["proprio_history_mask"].sum().item(), count)
            self.assertEqual(sample["action_history_mask"].sum().item(), min(7, timestep))
            for camera, key in (("agentview", "image"), ("robot0_eye_in_hand", "eye_in_hand_image")):
                frames = sample[f"{key}_history"]
                expected = torch.stack([
                    dataset._image_tensor(self.episodes[index // 10][CAMERA_OBSERVATION_KEYS[camera]][t])
                    for t in range(max(0, timestep - 7), timestep + 1)
                ])
                torch.testing.assert_close(frames[-count:], expected)
                torch.testing.assert_close(frames[-1], sample[key])
                self.assertEqual(torch.count_nonzero(frames[:8 - count]).item(), 0)
            past = self.episodes[index // 10]["actions"][max(0, timestep - 7):timestep]
            if len(past):
                torch.testing.assert_close(sample["action_history"][-len(past):], torch.from_numpy(past))
            torch.testing.assert_close(sample["action"], torch.from_numpy(self.episodes[index // 10]["actions"][timestep]))
        flipped = self.dataset(vertical_flip=True)[7]
        torch.testing.assert_close(flipped["image_history"], dataset[7]["image_history"].flip(-2))
        with self.assertRaises(ValueError):
            self.dataset(history_size=1)
        with self.assertRaises(ValueError):
            self.dataset(mode="act")

    def test_training_and_inference_history_match(self):
        dataset = self.dataset()
        # A fresh buffer per episode also tests reset semantics.
        for episode_index in range(2):
            buffer = PolicyHistory(8, rgb_history=True)
            for timestep in range(10):
                sample = dataset[episode_index * 10 + timestep]
                batch = default_collate([sample])
                expected, _, _, _ = prepare_batch(batch, self.normalizer, self.device, 32)
                cameras = {key: batch[key] for key in ("image", "eye_in_hand_image")}
                actual = buffer.observe(cameras, batch["proprio"], self.normalizer)
                self.assertEqual(set(actual), set(expected))
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key])
                buffer.record_action(sample["action"])
        resized, _, _, _ = prepare_batch(default_collate([dataset[0]]), self.normalizer, self.device, 64)
        self.assertEqual(resized["image_history"].shape, (1, 8, 3, 64, 64))

    def test_forward_backward_masking_and_checkpoint(self):
        dataset = self.dataset()
        inputs, target, _, _ = prepare_batch(default_collate([dataset[0], dataset[7]]), self.normalizer, self.device, 32)
        model = self.model()
        self.assertEqual(model.fusion_input_dim, 2184)
        model.train()
        before = {name: value.clone() for name, value in model.visual_encoder.backbone.named_buffers()}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        output = model(**inputs)
        self.assertEqual(output["action"].shape, (2, 7))
        self.assertEqual(output["agentview_spatial_feature"].shape, (2, 8, 128))
        loss = torch.nn.functional.mse_loss(output["action"], target)
        loss.backward()
        for module in (model.agentview_encoder, model.eye_in_hand_encoder, model.control_head, model.action_head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.backbone.parameters()))
        for name, value in model.visual_encoder.backbone.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        optimizer.step()
        model.eval()
        altered = {key: value.clone() for key, value in inputs.items()}
        for key in ("image_history", "eye_in_hand_image_history", "proprio_history"):
            altered[key][~inputs["proprio_history_mask"]] = 999
        altered["action_history"][~inputs["action_history_mask"]] = 999
        with torch.no_grad():
            expected = model(**inputs)["action"]
            torch.testing.assert_close(model(**altered)["action"], expected, rtol=0, atol=0)
            altered["image_history"][1] = inputs["image_history"][1].flip(0)
            self.assertFalse(torch.allclose(model(**altered)["action"][1], expected[1]))
        args = self.args()
        payload = checkpoint_payload(args, model, optimizer, 3, 10, 0.2, 0.5, [], [])
        checkpoint_path = self.root / "latest.pt"
        torch.save(payload, checkpoint_path)
        clone = self.model()
        clone_optimizer = torch.optim.AdamW([p for p in clone.parameters() if p.requires_grad])
        args.resume = checkpoint_path
        resumed = resume_training(args, clone, clone_optimizer, self.device)
        self.assertEqual(resumed[:4], (4, 10, 0.2, 0.5))
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(clone(**inputs)["action"], expected, rtol=0, atol=0)
        args.history_size = 4
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            resume_training(args, clone, clone_optimizer, self.device)

    def test_loader_epoch_and_legacy_modes(self):
        args = self.args()
        _, _, train, val = build_loaders(args, self.normalizer.proprio_keys)
        model = self.model(hidden_dim=64, spatial_channels=8)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        for loader, optim in ((train, optimizer), (val, None)):
            metrics = run_epoch(model, loader, self.normalizer, self.device, 32,
                                optimizer=optim, max_batches=1, log_every=100)
            self.assertEqual(metrics["batches"], 1)
            self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertEqual(len(train), 2)
        self.assertFalse(args.phase_balanced)
        dataset = self.dataset(rgb_history=False)
        inputs, _, _, _ = prepare_batch(default_collate([dataset[0]]), self.normalizer, self.device, 32)
        with torch.no_grad():
            self.assertEqual(self.model(rgb_history=False)(**inputs)["action"].shape, (1, 7))
        dataset = self.dataset(history_size=1, rgb_history=False)
        inputs, _, _, _ = prepare_batch(default_collate([dataset[0]]), self.normalizer, self.device, 32)
        with torch.no_grad():
            self.assertEqual(SpatialSoftmaxVisualBCPolicy(pretrained_visual=False,
                             camera_names=tuple(CAMERA_OBSERVATION_KEYS))(**inputs)["action"].shape, (1, 7))
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.args(["--phase-balanced", "--batches-per-phase", "1"])

    def test_evaluator_checkpoint_and_episode_reset(self):
        from scripts.evaluate_visual_bc import evaluate_episode, load_policy

        args = self.args()
        model = self.model()
        with torch.no_grad():
            model.action_head.weight.zero_()
            model.action_head.bias.fill_(100)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        path = self.root / "eval.pt"
        torch.save(checkpoint_payload(args, model, optimizer, 3, 10, 0.2, 0.5, [], []), path)
        loaded, normalizer, _, size, color = load_policy(SimpleNamespace(
            checkpoint=path, normalization=self.normalization_path, image_size=None,
        ), self.device)
        self.assertIsInstance(loaded, HistorySpatialSoftmaxVisualBCPolicy)
        self.assertTrue(loaded.rgb_history)
        self.assertEqual((size, color), (32, "red"))
        observed = []
        hook = loaded.register_forward_pre_hook(
            lambda module, positional, keyword: observed.append({
                key: value.detach().clone() for key, value in keyword.items()
            }), with_kwargs=True,
        )
        self.addCleanup(hook.remove)
        data = self.episodes[0]

        class FakeEnv:
            object_ids = ("cube_0",)
            action_spec = (np.full(7, -1), np.full(7, 1))

            def observation(self):
                return {key.removeprefix("obs__"): values[self.time]
                        for key, values in data.items() if key.startswith("obs__")}

            def reset(self):
                self.time = 0
                return self.observation()

            def get_task_metadata(self):
                return {"target_object": "cube_0", "target_color": "red"}

            def get_object_position(self, object_id):
                return np.array([0, 0, 0.83])

            def step(self, action):
                np.testing.assert_array_equal(action, np.ones(7))
                self.time += 1
                return self.observation(), 0, False, {}

            def _check_success(self):
                return False

        env = FakeEnv()
        for index in range(2):
            result = evaluate_episode(
                env, loaded, normalizer, size, self.device, 3, 5, False,
                index, None, 20, color,
            )
            self.assertEqual(result["steps"], 3)
            self.assertEqual(result["failure_reason"], "no_lift")
            first, second = observed[index * 3:index * 3 + 2]
            self.assertEqual(first["proprio_history_mask"].sum().item(), 1)
            self.assertEqual(first["action_history_mask"].sum().item(), 0)
            self.assertEqual(second["action_history_mask"].sum().item(), 1)
            torch.testing.assert_close(second["action_history"][0, -1],
                                       normalizer.normalize_action(torch.ones(7)))
            torch.testing.assert_close(second["image_history"][0, -2], first["image_history"][0, -1])

    def test_trainer_resume_periodic_evaluation_and_artifacts(self):
        from scripts.train_spatial_softmax_visual_bc import main

        args = self.args(["--spatial-channels", "8", "--hidden-dim", "64"])
        model = self.model(spatial_channels=8, hidden_dim=64)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        checkpoint = self.root / "epoch_019.pt"
        torch.save(checkpoint_payload(args, model, optimizer, 19, 19, float("inf"), -1, [], []), checkpoint)
        command = ["trainer", "--dataset", str(self.root), "--train-split", str(self.split),
                   "--val-split", str(self.split), "--normalization", str(self.normalization_path),
                   "--output-dir", str(args.output_dir), "--history-size", "8", "--rgb-history",
                   "--no-phase-balanced", "--batches-per-task", "2", "--batch-size", "2",
                   "--image-size", "32", "--dropout", "0", "--no-vertical-flip",
                   "--spatial-channels", "8", "--hidden-dim", "64", "--epochs", "20",
                   "--resume", str(checkpoint), "--max-train-batches", "1", "--max-val-batches", "1"]
        rollout = {"successful": 10, "attempted": 20, "success_rate": 0.5,
                   "failure_counts": {"no_lift": 10}}
        with patch("sys.argv", command), patch(
            "scripts.evaluate_visual_bc.run_rollout_evaluation", return_value=rollout
        ) as evaluate, redirect_stdout(io.StringIO()):
            main()
        evaluate.assert_called_once()
        self.assertIsInstance(evaluate.call_args.args[0], HistorySpatialSoftmaxVisualBCPolicy)
        self.assertEqual(evaluate.call_args.kwargs["episodes"], 20)
        for name in ("latest.pt", "best_val.pt", "best_rollout.pt", "epoch_020.pt"):
            self.assertTrue((args.output_dir / name).exists(), name)
        saved = torch.load(args.output_dir / "latest.pt", map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 20)
        self.assertEqual(saved["model_config"]["history_size"], 8)
        self.assertTrue(saved["model_config"]["rgb_history"])
        metrics = json.loads((args.output_dir / "rollout_evaluations.json").read_text())
        self.assertEqual(metrics[0]["success_rate"], 0.5)
        self.assertEqual(metrics[0]["epoch"], 20)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
