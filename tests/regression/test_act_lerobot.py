"""Numerical parity with pinned upstream ACT, native IO, subsets, and resume."""

import ast
import copy
import hashlib
import io
import json
import math
import unittest
import __future__
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from models.act.dataset import ACTDataset
from models.act_lerobot import LeRobotACTPolicy
from models.act_lerobot import policy
from models.act_lerobot.vendor import core
from models.act_lerobot.audit import audit_inputs
from scripts import train_act, train_act_lerobot, evaluate_act_lerobot
from scripts import inspect_act_lerobot, evaluate_visual_bc_validation as validation
from scripts.evaluation_utils import policy_observation
from data.normalization import PolicyNormalizer
from tests.regression import test_visual_bc as fixtures

ROOT = Path(__file__).resolve().parents[2]


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 1)
        self.bn1 = FrozenBatchNorm2d(4)
        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveAvgPool2d((2, 2))
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Conv2d(4, 512, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)


def tiny_resnet(*args, **kwargs):
    assert kwargs.get("weights") is None, "No automatic downloads"
    return TinyResNet()


def upstream_module():
    """Execute the original core AST, without installing the LeRobot package."""
    text = (ROOT / "tests/fixtures/lerobot_act_modeling_upstream.py").read_text()
    metadata = json.loads((ROOT / "models/act_lerobot/reference.json").read_text())
    assert hashlib.sha256(text.encode()).hexdigest() == metadata["source_sha256"]
    wanted = {"ACT", "ACTEncoder", "ACTEncoderLayer", "ACTDecoder", "ACTDecoderLayer",
              "create_sinusoidal_pos_embedding", "ACTSinusoidalPositionEmbedding2d", "get_activation_fn"}
    nodes = [node for node in ast.parse(text).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    # These are the only three einops expressions used by the pinned core.
    einops = SimpleNamespace(repeat=lambda x, pattern, b: x.unsqueeze(0).expand(b, -1, -1),
                             rearrange=lambda x, pattern: x.flatten(2).permute(2, 0, 1))
    namespace = {**vars(core), "einops": einops}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_upstream.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**namespace)


class LeRobotACTTests(unittest.TestCase):
    def setUp(self):
        fixtures.VisualBCTests.setUp(self)
        for path in (self.dataset / "pick").glob("*.hdf5"):
            with h5py.File(path, "r+") as episode:
                episode["actions"][:] = np.clip(episode["actions"][:], -1, 1)

    def model(self):
        return LeRobotACTPolicy(chunk_size=4, hidden_dim=32, num_heads=4, encoder_layers=1,
                                decoder_layers=1, feedforward_dim=64, latent_dim=4,
                                dropout=0., image_size=32)

    def command(self, output=None, *extra):
        return ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                "--output-dir", str(output or self.output), "--device", "cpu", "--allow-scratch",
                "--train-episodes", "2", "--validation-source", "train", "--batch-size", "2",
                "--chunk-size", "4", "--hidden-dim", "32", "--num-heads", "4", "--encoder-layers", "1",
                "--decoder-layers", "1", "--feedforward-dim", "64", "--latent-dim", "4", "--image-size", "32",
                "--checkpoint-every", "2", "--validation-batches", "1", "--rollout-eval-every", "0", *extra]

    def test_requires_explicit_pretrained_and_new_checkpoint_type(self):
        with patch("sys.argv", ["train", "--dataset", "missing", "--output-dir", str(self.output)]), \
                self.assertRaisesRegex(ValueError, "requires --backbone-weights"):
            train_act_lerobot.main()
        self.assertFalse(self.output.exists())
        checkpoint = self.root / "legacy.pt"
        torch.save(dict(policy_type="act"), checkpoint)
        with patch("sys.argv", self.command(None, "--resume", str(checkpoint))), \
                self.assertRaisesRegex(ValueError, "act_lerobot"):
            train_act_lerobot.main()
        self.assertEqual(train_act_lerobot.DEFAULTS["learning_rate"], 1e-5)
        self.assertEqual(train_act_lerobot.DEFAULTS["decoder_layers"], 1)

    def test_core_forward_backward_matches_pinned_upstream(self):
        upstream = upstream_module()
        with patch.object(core.torchvision.models, "resnet18", side_effect=tiny_resnet):
            model = self.model()
            reference = upstream.ACT(model.core.config)
        reference.load_state_dict(model.core.state_dict(), strict=True)
        image = torch.rand(2, 3, 32, 32)
        eye = torch.rand(2, 3, 32, 32)
        state = torch.rand(2, 16)
        actions = torch.rand(2, 4, 7)
        padding = torch.tensor([[False, False, True, True], [False, True, True, True]])
        batch = {core.OBS_IMAGES: model.preprocess_images(image, eye), core.OBS_STATE: state,
                 core.ACTION: actions.masked_fill(padding[..., None], 0), "action_is_pad": padding}
        model.train(); reference.train()
        torch.manual_seed(93)
        actual = model(image, state, eye_in_hand_image=eye, actions=actions, is_pad=padding)
        torch.manual_seed(93)
        expected, (mu, logvar) = reference(batch)
        torch.testing.assert_close(actual["action"], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual["mu"], mu, rtol=0, atol=0)
        torch.testing.assert_close(actual["logvar"], logvar, rtol=0, atol=0)
        loss = train_act.action_losses(actual["action"], actions, padding, actual["mu"], actual["logvar"])["loss"]
        expected_loss = train_act.action_losses(expected, actions, padding, mu, logvar)["loss"]
        loss.backward(); expected_loss.backward()
        for (name, p), (other, q) in zip(model.core.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        model.eval(); reference.eval()
        with torch.inference_mode():
            actual = model(image, state, eye_in_hand_image=eye, actions=actions, is_pad=padding)
            expected, latent = reference(batch)
            torch.testing.assert_close(actual["action"], expected, rtol=0, atol=0)
            self.assertIsNone(actual["mu"])
            self.assertEqual(latent, (None, None))

    def test_images_scale_once_and_local_weights_are_loaded_strictly(self):
        with patch.object(core.torchvision.models, "resnet18", side_effect=tiny_resnet):
            model = self.model()
        image = torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
        for a,b in zip(model.preprocess_images(image,image), model.preprocess_images(image.float()/255,image.float()/255)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError, "in \\[0,1\\]"):
            model.preprocess_images(image.float(),image.float())
        path = self.root / "resnet.pth"
        weights = TinyResNet().state_dict()
        weights["conv1.weight"].fill_(.123)
        torch.save(weights,path)
        model.load_backbone_weights(path)
        self.assertTrue(torch.allclose(model.backbone.conv1.weight,torch.full_like(model.backbone.conv1.weight,.123)))

    def test_training_and_rollout_preprocessing_match_for_identical_observation(self):
        prepared=fixtures.VisualBCTests.prepare(self)
        norm=PolicyNormalizer.from_statistics(prepared['normalization'])
        data=ACTDataset(self.dataset,self.prepared/'train.jsonl',prepared['camera_names'],norm.proprio_keys,4)
        self.addCleanup(data.close)
        sample=data[0]
        ep,t=data._sample_index[0]
        with data._episode(ep) as archive:
            obs={key.removeprefix('obs__'):archive[key][t] for key in data.policy_observation_keys}
        camera,state=policy_observation(obs,32,self.device,norm.proprio_keys,prepared['camera_names'])
        with patch.object(core.torchvision.models,'resnet18',side_effect=tiny_resnet):
            model=self.model().eval()
        train_images=model.preprocess_images(sample['image'][None],sample['eye_in_hand_image'][None])
        rollout_images=model.preprocess_images(**camera)
        for a,b in zip(train_images,rollout_images):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(norm.normalize_proprio(sample['proprio'][None]),norm.normalize_proprio(state),rtol=0,atol=0)

    def test_temporal_ensemble_matches_oldest_first_reference(self):
        with patch.object(core.torchvision.models,'resnet18',side_effect=tiny_resnet):
            model=self.model().eval()
        model.configure_inference(1,True,.5)
        image,state=torch.zeros(1,3,32,32),torch.zeros(1,16)
        chunks=[torch.arange(4,dtype=torch.float32)[None,:,None].expand(1,4,7)+offset for offset in [0,10,20]]
        hidden=None
        with patch.object(model,'forward',side_effect=[dict(action=c) for c in chunks]):
            for t in range(3):
                output=model.step(image,state,eye_in_hand_image=image,hidden=hidden)
                hidden=output['hidden']
        expected=(2+11*math.exp(-.5)+20*math.exp(-1))/(1+math.exp(-.5)+math.exp(-1))
        self.assertAlmostEqual(output['action'][0,0].item(),expected,places=5)

    def test_native_subset_audit_and_exact_resume(self):
        prepared = fixtures.VisualBCTests.prepare(self)
        original = {p: p.read_bytes() for p in self.prepared.iterdir()}
        full = self.root / "full"
        with patch.object(core.torchvision.models, "resnet18", side_effect=tiny_resnet), redirect_stdout(io.StringIO()):
            with patch("sys.argv", self.command(full, "--max-updates", "4")):
                train_act_lerobot.main()
            with patch("sys.argv", self.command(None, "--max-updates", "2")):
                train_act_lerobot.main()
            with patch("sys.argv", ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                    "--output-dir", str(self.output), "--resume", str(self.output / "latest.pt"),
                    "--max-updates", "4", "--device", "cpu"]):
                train_act_lerobot.main()
            model, norm, checkpoint = evaluate_act_lerobot.load_policy(self.output / "latest.pt",self.device)
            loaded = validation.load_validation_policy(self.output / "latest.pt",self.device,16)
            self.assertEqual(loaded[-1],0)
            self.assertEqual(model.policy_type,"act_lerobot")
        uninterrupted = torch.load(full / "latest.pt",weights_only=False)
        self.assertEqual(checkpoint["history"],uninterrupted["history"])
        for key,value in checkpoint["model_state"].items():
            torch.testing.assert_close(value,uninterrupted["model_state"][key],rtol=0,atol=0)
        self.assertEqual(len(checkpoint["training_subset"]),2)
        self.assertEqual(checkpoint["training_config"]["validation_source"],"train")
        self.assertEqual(original,{p:p.read_bytes() for p in original})
        report=json.loads((self.output/'input_audit.json').read_text())
        self.assertTrue(report['source_alignment_pass'])
        self.assertTrue(report['normalization_roundtrip_pass'])
        arrays=np.load(self.output/'prediction_chunks.npz',allow_pickle=False)
        self.assertEqual(arrays['predicted_chunk_raw'].shape[-2:],(4,7))
        self.assertTrue((self.output/'prediction_diagnostics.json').exists())

    def test_chunk_execution_and_full_chunk_metadata(self):
        with patch.object(core.torchvision.models, "resnet18", side_effect=tiny_resnet):
            model = self.model().eval()
        image, state = torch.rand(2,3,32,32), torch.rand(2,16)
        model.configure_inference(2)
        with torch.inference_mode():
            a=model.step(image,state,eye_in_hand_image=image)
            b=model.step(image,state,eye_in_hand_image=image,hidden=a['hidden'])
        self.assertEqual(a['chunk_offset'],0)
        self.assertEqual(b['chunk_offset'],1)
        self.assertIsNone(b['action_chunk'])
        torch.testing.assert_close(b['action'],a['action_chunk'][:,1],rtol=0,atol=0)

    def test_inspection_exports_images_and_all_offsets(self):
        fixtures.VisualBCTests.prepare(self)
        inspection = self.root / "inspection"
        with patch.object(core.torchvision.models, "resnet18", side_effect=tiny_resnet), redirect_stdout(io.StringIO()):
            with patch("sys.argv", self.command(None,"--max-updates","2")):
                train_act_lerobot.main()
            with patch("sys.argv", ["inspect", "--dataset",str(self.dataset),"--training-dir",str(self.prepared),
                    "--checkpoint",str(self.output/'best.pt'),"--output-dir",str(inspection),
                    "--split","train","--samples","1","--device","cpu"]):
                inspect_act_lerobot.main()
        self.assertTrue((inspection/'sample_000_observation.png').exists())
        self.assertTrue((inspection/'sample_000_chunk.png').exists())
        arrays=np.load(inspection/'chunks.npz',allow_pickle=False)
        self.assertEqual(arrays['predicted_chunk_raw'].shape,(1,4,7))

    def test_validation_preflight_selects_checkpoint_training_scenes(self):
        prepared=dict(tasks=['place'],scene_counts=[[1,1]],fixed_target_color='red')
        paths=[dict(path=f'place/episode_{i:06d}.hdf5',episode=i,steps=2,task='place') for i in range(4)]
        subset=paths[:2]
        checkpoint=self.root/'subset.pt'
        torch.save(dict(policy_type='act_lerobot',data_config=prepared,training_subset=subset),checkpoint)
        (self.prepared).mkdir()
        (self.prepared/'train.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in paths))
        target=self.root/'scenes'
        args=['eval','--dataset',str(self.dataset),'--training-dir',str(self.prepared),
              '--checkpoints',f'NEW={checkpoint}','--output-dir',str(target),
              '--split','train','--checkpoint-training-subset','--device','cpu']
        with patch('sys.argv',args),patch.object(validation,'load_bc_training',return_value=(prepared,self.prepared)), \
                patch.object(validation,'load_validation_policy',side_effect=RuntimeError('passed preflight')), \
                self.assertRaisesRegex(RuntimeError,'passed preflight'):
            validation.main()
        rows=json.loads((target/'episode_summary.json').read_text())
        self.assertEqual({r['source_path'] for r in rows},{p['path'] for p in subset})


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
