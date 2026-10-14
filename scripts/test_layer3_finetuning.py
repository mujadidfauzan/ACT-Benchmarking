"""Offline tests for staged layer3 fine-tuning, resume, and initialization."""

import copy
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from torch.utils.data import default_collate

from scripts import test_spatial_softmax_history as fixtures
from scripts import train_spatial_softmax_visual_bc as training


class Layer3FineTuneTests(fixtures.RGBHistoryTests):
    def fine_args(self):
        return self.args(["--hidden-dim", "64", "--spatial-channels", "8",
                          "--unfreeze-layer3-epoch", "2", "--backbone-learning-rate", "0.00001"])

    def fine_model(self):
        return self.model(hidden_dim=64, spatial_channels=8)

    def batch(self):
        dataset = self.dataset()
        return training.prepare_batch(default_collate([dataset[0], dataset[7]]),
                                      self.normalizer, self.device, 32)[:2]

    @staticmethod
    def step(model, optimizer, inputs, target):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(**inputs)["action"], target)
        loss.backward()
        optimizer.step()
        return loss

    def test_warmup_then_layer3_updates_without_batchnorm_drift(self):
        args = self.fine_args()
        model = self.fine_model()
        optimizer = training.build_optimizer(args, model)
        inputs, target = self.batch()
        initial = {name: parameter.detach().clone()
                   for name, parameter in model.visual_encoder.backbone.named_parameters()}
        buffers = {name: value.clone() for name, value in model.visual_encoder.backbone.named_buffers()}
        self.step(model, optimizer, inputs, target)
        for name, parameter in model.visual_encoder.backbone.named_parameters():
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad, name)
        self.assertTrue(training.enable_layer3_finetuning(args, model, optimizer))
        self.assertFalse(training.enable_layer3_finetuning(args, model, optimizer))
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-4, 1e-5])
        ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        loss = self.step(model, optimizer, inputs, target)
        self.assertTrue(torch.isfinite(loss))
        changed = False
        for name, parameter in model.visual_encoder.backbone.named_parameters():
            if name.startswith("layer3."):
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                changed |= not torch.equal(parameter, initial[name])
            else:
                self.assertFalse(parameter.requires_grad, name)
                self.assertIsNone(parameter.grad, name)
                torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
        self.assertTrue(changed)
        for name, value in model.visual_encoder.backbone.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        model.eval()
        model.train()
        self.assertTrue(all(not module.training for module in model.visual_encoder.backbone.modules()
                            if isinstance(module, torch.nn.BatchNorm2d)))

    def test_resume_restores_active_layer3_and_two_optimizer_groups(self):
        args = self.fine_args()
        model = self.fine_model()
        optimizer = training.build_optimizer(args, model)
        inputs, target = self.batch()
        training.enable_layer3_finetuning(args, model, optimizer)
        self.step(model, optimizer, inputs, target)
        path = self.root / "fine.pt"
        payload = training.checkpoint_payload(args, model, optimizer, 2, 2, 0.2, 0.5, [], [])
        torch.save(payload, path)
        self.assertTrue(payload["model_config"]["layer3_trainable"])
        self.assertFalse(payload["model_config"]["freeze_visual_backbone"])
        clone = self.fine_model()
        clone_optimizer = training.build_optimizer(args, clone)
        args.resume = path
        args.backbone_learning_rate = 2e-5
        result = training.resume_training(args, clone, clone_optimizer, self.device)
        self.assertEqual(result[0], 3)
        self.assertTrue(clone.visual_encoder._layer3_trainable)
        self.assertEqual(len(clone_optimizer.param_groups), 2)
        self.assertEqual(clone_optimizer.param_groups[1]["lr"], 2e-5)
        self.assertEqual(len(clone_optimizer.state), len(optimizer.state))
        model.eval()
        clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(**inputs)["action"], clone(**inputs)["action"], rtol=0, atol=0)
        self.step(clone, clone_optimizer, inputs, target)
        args.unfreeze_layer3_epoch = 3
        rejected_model = self.fine_model()
        with self.assertRaisesRegex(ValueError, "retain"):
            training.resume_training(args, rejected_model, training.build_optimizer(args, rejected_model), self.device)

    def test_legacy_frozen_resume_can_start_finetuning(self):
        args = self.fine_args()
        args.unfreeze_layer3_epoch = None
        model = self.fine_model()
        optimizer = training.build_optimizer(args, model)
        inputs, target = self.batch()
        self.step(model, optimizer, inputs, target)
        payload = training.checkpoint_payload(args, model, optimizer, 1, 1, 0.2, -1, [], [])
        for field in ("layer3_trainable", "batch_norm_statistics_frozen"):
            payload["model_config"].pop(field)
        payload["training_config"].pop("unfreeze_layer3_epoch")
        payload["optimizer_state"]["param_groups"][0].pop("name")
        path = self.root / "legacy.pt"
        torch.save(payload, path)
        args.resume = path
        args.unfreeze_layer3_epoch = 2
        clone = self.fine_model()
        clone_optimizer = training.build_optimizer(args, clone)
        result = training.resume_training(args, clone, clone_optimizer, self.device)
        self.assertEqual(result[0], 2)
        self.assertFalse(clone.visual_encoder._layer3_trainable)
        self.assertEqual(len(clone_optimizer.param_groups), 1)
        training.enable_layer3_finetuning(args, clone, clone_optimizer)
        self.step(clone, clone_optimizer, inputs, target)
        self.assertTrue(clone.visual_encoder._layer3_trainable)

    def test_trainer_schedule_init_checkpoint_and_inference(self):
        from scripts.evaluate_visual_bc import load_policy
        from models import visual_encoder

        args = self.fine_args()
        args.history_size, args.rgb_history = 1, False
        args.epochs = 2
        args.max_train_batches = args.max_val_batches = 1
        args.rollout_eval_every = 0
        with patch.object(training, "parse_args", return_value=args), redirect_stdout(io.StringIO()):
            training.main()
        metrics = json.loads((args.output_dir / "metrics.json").read_text())
        self.assertEqual([entry["layer3_trainable"] for entry in metrics], [False, True])
        self.assertEqual(metrics[0]["learning_rates"], {"heads": 1e-4})
        self.assertEqual(metrics[1]["learning_rates"], {"heads": 1e-4, "backbone_layer3": 1e-5})
        checkpoint_path = args.output_dir / "latest.pt"
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded, _, _, _, _ = load_policy(SimpleNamespace(
            checkpoint=checkpoint_path, normalization=self.normalization_path, image_size=32,
        ), self.device)
        self.assertTrue(all(not p.requires_grad for p in loaded.visual_encoder.backbone.parameters()))
        torch.testing.assert_close(loaded.visual_encoder.backbone.layer3[0].conv1.weight,
                                   payload["model_state"]["visual_encoder.backbone.layer3.0.conv1.weight"], rtol=0, atol=0)
        fork = copy.copy(args)
        fork.init_checkpoint = checkpoint_path
        fork.output_dir = self.root / "fork"
        fork.epochs, fork.unfreeze_layer3_epoch = 1, 1
        fork.pretrained_visual = True
        original_resnet18 = visual_encoder.resnet18

        def no_download(*args, **kwargs):
            self.assertIsNone(kwargs["weights"])
            return original_resnet18(*args, **kwargs)

        with patch.object(training, "parse_args", return_value=fork), patch.object(
            visual_encoder, "resnet18", side_effect=no_download
        ), redirect_stdout(io.StringIO()):
            training.main()
        fresh = torch.load(fork.output_dir / "latest.pt", map_location="cpu", weights_only=False)
        self.assertEqual((fresh["epoch"], fresh["global_step"], len(fresh["history"])), (1, 1, 1))
        self.assertTrue(fresh["model_config"]["layer3_trainable"])
        for state in fresh["optimizer_state"]["state"].values():
            self.assertEqual(state["step"].item(), 1)

    def test_new_argument_validation(self):
        for flags in (["--unfreeze-layer3-epoch", "0"], ["--backbone-learning-rate", "0"]):
            with self.assertRaises(ValueError):
                self.args(flags)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
