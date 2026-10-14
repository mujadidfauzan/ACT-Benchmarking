"""Offline tests for configurable spatial-softmax Language-BC, without downloads."""

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
from torch import nn

from data.manipulation_dataset import EEF_GRIPPER_PROPRIO_KEYS
from models.language_bc import LanguageBCPolicy
from scripts import train_language_bc as training


class SpatialLanguageBCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    @staticmethod
    def inputs(mode, dual=True):
        values = {"image": torch.rand(2, 3, 32, 32), "proprio": torch.randn(2, 9)}
        if dual:
            values["eye_in_hand_image"] = torch.rand_like(values["image"])
        if mode == "pooled":
            values["language_embedding"] = torch.randn(2, 384)
        else:
            values["token_embeddings"] = torch.randn(2, 6, 384)
            values["attention_mask"] = torch.tensor([
                [True, True, True, False, False, False],
                [True, True, True, True, True, False],
            ])
        return values

    def test_shapes_gradients_and_padding_both_language_modes(self):
        for mode in ("pooled", "token_attention"):
            for cameras in (("agentview",), ("agentview", "robot0_eye_in_hand")):
                with self.subTest(mode=mode, cameras=cameras):
                    model = LanguageBCPolicy(language_mode=mode, visual_fusion="spatial_softmax",
                                             camera_names=cameras, pretrained_visual=False,
                                             fusion_hidden_dims=(64, 32), dropout=0)
                    self.assertIsNone(model.proprio_encoder)
                    self.assertEqual(model.fusion_input_dim, 128 * len(cameras) + 9 + 128)
                    self.assertIsInstance(model.fusion[2], nn.ReLU)
                    values = self.inputs(mode, dual=len(cameras) == 2)
                    buffers = {name: value.clone() for name, value in model.visual_encoder.backbone.named_buffers()}
                    model.train()
                    # The new mode must not use layer4 or the global projection.
                    with patch.object(model.visual_encoder.backbone.layer4, "forward",
                                      side_effect=AssertionError("layer4 used")), patch.object(
                        model.visual_encoder.projection, "forward", side_effect=AssertionError("global projection used")
                    ):
                        output = model(**values)
                    self.assertEqual(output["action"].shape, (2, 7))
                    self.assertEqual(output["visual_feature"].shape, (2, 128))
                    self.assertEqual(output["language_feature"].shape, (2, 128))
                    torch.testing.assert_close(output["proprio_feature"], values["proprio"])
                    coordinates = output["spatial_attention_coordinates"]
                    self.assertEqual(coordinates.shape, (2, 64, 2))
                    self.assertTrue(torch.all(coordinates.abs() <= 1))
                    weights = output["spatial_attention_weights"]
                    torch.testing.assert_close(weights.sum((-2, -1)), torch.ones(2, 64))
                    output["action"].square().mean().backward()
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        else:
                            self.assertIsNone(parameter.grad, name)
                    for name, buffer in model.visual_encoder.backbone.named_buffers():
                        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
                    if mode == "token_attention":
                        self.assertEqual(output["attention_weights"][~values["attention_mask"]].count_nonzero(), 0)
                        model.eval()
                        with torch.no_grad():
                            expected = model(**values)["action"]
                            altered = {key: value.clone() for key, value in values.items()}
                            altered["token_embeddings"][~values["attention_mask"]] = 999
                            torch.testing.assert_close(model(**altered)["action"], expected, rtol=0, atol=0)

    def test_widths_default_config_and_checkpoint_round_trip(self):
        for mode, channels, widths in (("pooled", 64, None), ("token_attention", 16, (128, 48))):
            model = LanguageBCPolicy(language_mode=mode, visual_fusion="spatial_softmax",
                                     camera_names=("agentview", "robot0_eye_in_hand"),
                                     spatial_channels=channels, fusion_hidden_dims=widths,
                                     pretrained_visual=False, dropout=0)
            model.eval()
            self.assertEqual(model.fusion_hidden_dims, widths or (1024, 1024))
            self.assertEqual(model.fusion_input_dim, channels * 4 + 9 + 128)
            buffer = io.BytesIO()
            torch.save({"model_config": model.get_config(), "model_state": model.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=False)
            restored = LanguageBCPolicy.from_config(checkpoint["model_config"])
            restored.load_state_dict(checkpoint["model_state"])
            restored.eval()
            values = self.inputs(mode)
            with torch.no_grad():
                torch.testing.assert_close(model(**values)["action"], restored(**values)["action"], rtol=0, atol=0)
                changed = {key: value.clone() for key, value in values.items()}
                if mode == "pooled":
                    changed["language_embedding"] = changed["language_embedding"].flip(0)
                else:
                    changed["token_embeddings"] = changed["token_embeddings"].flip(0)
                self.assertFalse(torch.allclose(model(**values)["action"], model(**changed)["action"]))
        for invalid in ((0, 32), (-1, 32), (32,)):
            with self.assertRaises(ValueError):
                LanguageBCPolicy(fusion_hidden_dims=invalid, pretrained_visual=False)

    def test_legacy_checkpoint_configuration(self):
        for mode in ("global", "spatial_attention"):
            legacy = LanguageBCPolicy(visual_fusion=mode, pretrained_visual=False)
            config = {key: value for key, value in legacy.get_config().items()
                      if key in ("language_mode", "visual_fusion", "camera_names", "action_dim",
                                 "proprio_dim", "language_input_dim")}
            if mode == "global":
                del config["visual_fusion"]
            restored = LanguageBCPolicy.from_config(config)
            restored.load_state_dict(legacy.state_dict(), strict=True)
            training.validate_checkpoint_architecture({"model_config": config}, restored)
            self.assertEqual(restored.fusion_hidden_dims, (256, 128))
            self.assertIsInstance(restored.proprio_encoder, nn.Module)
            self.assertIsInstance(restored.fusion[2], nn.GELU)

    def test_cli_and_resume_reject_wrong_architecture(self):
        for flags, expected in ((["--hidden-dims", "512", "256"], (512, 256)),
                                (["--hidden-dim", "256"], (256, 256)), ([], None)):
            with patch("sys.argv", ["train", "--output-dir", "unused", "--visual-fusion", "spatial_softmax", *flags]):
                args = training.parse_args()
            training.validate_args(args)
            self.assertEqual(training.fusion_hidden_dims(args), expected)
        for flags in (["--hidden-dim", "0"], ["--hidden-dims", "32", "-1"],
                      ["--unfreeze-layer4-epoch", "2"], ["--spatial-channels", "7"]):
            with patch("sys.argv", ["train", "--output-dir", "unused", "--visual-fusion", "spatial_softmax", *flags]):
                args = training.parse_args()
            with self.assertRaises(ValueError):
                training.validate_args(args)
        model = LanguageBCPolicy(visual_fusion="spatial_softmax", pretrained_visual=False,
                                 fusion_hidden_dims=(64, 32))
        checkpoint = {"model_config": {**model.get_config(), "fusion_hidden_dims": [128, 32]}}
        with self.assertRaisesRegex(ValueError, "fusion_hidden_dims"):
            training.validate_checkpoint_architecture(checkpoint, model)

    def test_multitask_trainer_resume_and_evaluator(self):
        from scripts.evaluate_language_bc import load_policy

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(7)
            instructions = ["Lift the red cube.", "Place the blue cube in the red tray.",
                            "Stack blue and red from bottom to top."]
            entries = []
            for task, instruction in zip(("pick", "place", "stack"), instructions):
                for episode in range(2):
                    path = f"{task}_{episode}.npz"
                    arrays = {"actions": rng.normal(size=(4, 7)).astype(np.float32),
                              "phases": np.asarray(["approach", "pre_grasp", "close", "lift"])}
                    for camera in ("agentview", "robot0_eye_in_hand"):
                        arrays[f"obs__{camera}_image"] = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
                    for key, dimension in zip(EEF_GRIPPER_PROPRIO_KEYS, (3, 4, 2)):
                        arrays[key] = rng.normal(size=(5, dimension)).astype(np.float32)
                    np.savez(root / path, **arrays)
                    entries.append({"task": task, "episode": episode, "path": path,
                                    "steps": 4, "instruction": instruction})
            split = root / "train.jsonl"
            split.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
            normalizer = root / "normalization.json"
            normalizer.write_text(json.dumps({
                "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                "task_balanced_proprio": {"keys": EEF_GRIPPER_PROPRIO_KEYS, "mean": [0] * 9, "std": [1] * 9},
                "task_balanced_action": {"mean": [0] * 7, "std": [1] * 7},
            }), encoding="utf-8")
            embeddings = rng.normal(size=(3, 384)).astype(np.float32)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
            cache = root / "pooled.npz"
            np.savez(cache, instructions=np.asarray(instructions), embeddings=embeddings,
                     metadata_json=np.asarray("{}"))
            output = root / "outputs"
            command = ["train", "--dataset", str(root), "--train-split", str(split),
                       "--val-split", str(split), "--normalization", str(normalizer),
                       "--pooled-cache", str(cache), "--output-dir", str(output),
                       "--visual-fusion", "spatial_softmax", "--hidden-dims", "64", "32",
                       "--camera-names", "agentview", "robot0_eye_in_hand", "--batch-size", "2",
                       "--image-size", "32", "--batches-per-task", "1", "--dropout", "0",
                       "--no-vertical-flip", "--epochs", "1", "--device", "cpu"]
            with patch("sys.argv", command), redirect_stdout(io.StringIO()):
                training.main()
            command.extend(["--resume", str(output / "latest.pt"), "--epochs", "2"])
            with patch("sys.argv", command), redirect_stdout(io.StringIO()):
                training.main()
            checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(checkpoint["model_config"]["fusion_hidden_dims"], [64, 32])
            self.assertEqual(checkpoint["history"][-1]["train"]["batches"], 3)
            self.assertEqual(set(checkpoint["history"][-1]["train"]["loss_per_task"]), {"pick", "place", "stack"})
            previous_config = (output / "config.json").read_bytes()
            mismatched = command.copy()
            mismatched[mismatched.index("--hidden-dims") + 1] = "128"
            with patch("sys.argv", mismatched), redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                ValueError, "fusion_hidden_dims"
            ):
                training.main()
            self.assertEqual((output / "config.json").read_bytes(), previous_config)
            loaded, _, _, _, _ = load_policy(SimpleNamespace(
                checkpoint=output / "latest.pt", normalization=normalizer, pooled_cache=cache,
                token_cache=cache, strict_language_cache=True, image_size=None,
            ), torch.device("cpu"))
            self.assertEqual(loaded.fusion_input_dim, 393)
            self.assertEqual(loaded.fusion_hidden_dims, (64, 32))
            reference = LanguageBCPolicy.from_config(checkpoint["model_config"])
            reference.load_state_dict(checkpoint["model_state"])
            reference.eval()
            values = self.inputs("pooled")
            with torch.no_grad():
                torch.testing.assert_close(loaded(**values)["action"], reference(**values)["action"], rtol=0, atol=0)

            token_cache = root / "tokens.npz"
            masks = np.ones((3, 6), dtype=np.bool_)
            masks[:, -1] = False
            np.savez(token_cache, instructions=np.asarray(instructions),
                     token_embeddings=rng.normal(size=(3, 6, 384)).astype(np.float32),
                     attention_masks=masks, input_ids=np.ones((3, 6), dtype=np.int64),
                     tokens_json=np.asarray(json.dumps([["token"] * 6] * 3)),
                     metadata_json=np.asarray("{}"))
            token_model = LanguageBCPolicy(visual_fusion="spatial_softmax", language_mode="token_attention",
                                           camera_names=("agentview", "robot0_eye_in_hand"),
                                           fusion_hidden_dims=(96, 48), spatial_channels=16,
                                           pretrained_visual=False, dropout=0)
            token_model.eval()
            token_checkpoint = root / "token.pt"
            torch.save({"model_config": token_model.get_config(), "model_state": token_model.state_dict()},
                       token_checkpoint)
            token_loaded, _, _, _, _ = load_policy(SimpleNamespace(
                checkpoint=token_checkpoint, normalization=normalizer, pooled_cache=cache,
                token_cache=token_cache, strict_language_cache=True, image_size=32,
            ), torch.device("cpu"))
            self.assertEqual(token_loaded.fusion_hidden_dims, (96, 48))
            self.assertEqual(token_loaded.spatial_channels, 16)
            values = self.inputs("token_attention")
            with torch.no_grad():
                torch.testing.assert_close(token_loaded(**values)["action"], token_model(**values)["action"],
                                           rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
