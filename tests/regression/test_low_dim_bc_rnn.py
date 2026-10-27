"""Numeric-only IO, causal alignment, resume and shared physical evaluation."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import torch

from data.bc_preparation import prepare_bc_training
from data.episode_io import write_episode
from models.low_dim_bc_rnn import LowDimBCRNNPolicy
from models.low_dim_bc_rnn.dataset import LowDimSequenceDataset
from models.low_dim_bc_rnn.observations import STATE_FIELDS, pack_state, validate_scene
from scripts import train_low_dim_bc_rnn as training, evaluate_low_dim_bc_rnn as evaluation
from scripts import evaluate_visual_bc as shared_evaluation
from tests.regression.test_variable_scene_counts import FakeEnv, metadata


class StateEnv(FakeEnv):
    def __init__(self, seed):
        super().__init__("place", 1, 1, seed)
        self.object_ids, self.receptacle_ids = ["object_0"], ["receptacle_0"]
        self.num_receptacles = 1
        self.action_spec = (-np.ones(7), np.ones(7))

    def observation(self):
        result = super().observation()
        for key in list(result):
            if key.endswith("_image"):
                del result[key]
        result.update(object_0_pos=self.get_object_position(),
                      object_0_quat=np.array([0., 0., 0., 1.]),
                      receptacle_0_pos=self.get_receptacle_position(),
                      receptacle_0_quat=np.array([0., 0., 0., 1.]))
        return result

    def get_object_position(self, identifier=None):
        return np.array([0., 0., .83])

    def get_receptacle_position(self, identifier=None):
        return np.array([.1, .1, .829])

    def placement_status(self, *args):
        return dict(success=False, wrong_target=False, inside_target=False,
                    released=True, distractors_stable=True)


class LowDimBCRNNTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset, self.prepared, self.output = (self.root / name for name in ("data", "split", "run"))
        (self.dataset / "place").mkdir(parents=True)
        entries = []
        for index, steps in enumerate((3, 7, 2, 5, 4, 6, 3, 4, 2, 3)):
            meta = metadata("place", 1, 1)
            meta["control_orientation"] = True
            time = np.arange(steps + 1, dtype=np.float32)[:, None]
            actions = np.arange(steps * 7, dtype=np.float32).reshape(steps, 7) / 100
            actions[:, 6] = np.where(np.arange(steps) % 2, 1, -1)
            arrays = {"actions": actions, "metadata_json": np.asarray(json.dumps(meta)),
                      "obs__robot0_joint_pos": np.repeat(time / 100, 7, axis=1)}
            for offset, (name, dim) in enumerate(STATE_FIELDS):
                arrays["obs__" + name] = (np.tile([0., 0., 0., 1.], (steps + 1, 1))
                                          if name.endswith("_quat") else
                                          np.repeat(time / 100 + offset / 10 + index / 100, dim, axis=1))
            arrays["obs__agentview_image"] = np.full((steps + 1, 32, 32, 3), 80 + index, dtype=np.uint8)
            name = f"episode_{index:06d}.hdf5"
            write_episode(self.dataset / "place" / name, arrays)
            entries.append(dict(episode=index, path=name, steps=steps, metadata=meta, num_objects=1,
                                num_receptacles=1, instruction=meta["instruction"]))
        (self.dataset / "place/manifest.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
        self.config = prepare_bc_training(self.dataset, self.prepared)

    def command(self, output=None, *extra):
        return ["train", "--dataset", str(self.dataset), "--training-dir", str(self.prepared),
                "--output-dir", str(output or self.output), "--device", "cpu", "--batch-size", "2",
                "--hidden-dim", "16", "--encoder-dim", "12", "--num-layers", "1",
                "--sequence-length", "3", "--checkpoint-every", "2",
                "--validation-batches", "2", "--rollout-eval-every", "0", *extra]

    def test_numeric_only_windows_alignment_padding_and_rollout_order(self):
        original = h5py.Dataset.__getitem__

        def numeric_only(dataset, key):
            self.assertFalse(dataset.name.endswith("_image"), dataset.name)
            return original(dataset, key)

        with patch.object(h5py.Dataset, "__getitem__", numeric_only):
            dataset = LowDimSequenceDataset(self.dataset, self.prepared / "train.jsonl", 10)
            sample = dataset[1]
        self.assertEqual(len(dataset), self.config["normalization"]["training_timesteps"])
        self.assertEqual(set(sample), {"state", "action"})
        self.assertEqual(sample["state"].shape, (10, 23))
        episode, start = dataset._sample_index[1]
        with h5py.File(self.dataset / dataset.episodes[episode]["path"], "r") as handle:
            arrays = {"obs__" + name: handle["obs__" + name][:] for name, _ in STATE_FIELDS}
            expected = torch.from_numpy(pack_state(arrays, "obs__"))[start:-1]
            action = torch.from_numpy(handle["actions"][:])[start:]
            obs = {name: arrays["obs__" + name][start] for name, _ in STATE_FIELDS}
        torch.testing.assert_close(sample["state"][:len(expected)], expected)
        torch.testing.assert_close(sample["state"][len(expected):], expected[-1:].expand(10 - len(expected), -1))
        torch.testing.assert_close(sample["action"][:len(action)], action)
        torch.testing.assert_close(sample["action"][len(action):], action[-1:].expand(10 - len(action), -1))
        model = LowDimBCRNNPolicy(hidden_dim=16)
        torch.testing.assert_close(model.observation_inputs(obs, "cpu")["state"][0], sample["state"][0])

    def test_missing_misaligned_nonfinite_states_and_unbounded_actions_fail(self):
        entry = json.loads((self.prepared / "train.jsonl").read_text().splitlines()[0])
        path = self.dataset / entry["path"]
        for key, value, message in (
            ("obs__object_0_pos", None, "missing low-dim"),
            ("obs__object_0_pos", np.zeros((1, 3)), "shape"),
            ("obs__receptacle_0_pos", np.full((entry["steps"] + 1, 3), np.nan), "nonfinite"),
            ("obs__object_0_quat", np.zeros((entry["steps"] + 1, 4)), "unit norm"),
            ("actions", np.full((entry["steps"], 7), 2.), "controller actions"),
        ):
            with self.subTest(key=key, message=message):
                with h5py.File(path, "r+") as handle:
                    old = handle[key][:]
                    del handle[key]
                    if value is not None:
                        handle.create_dataset(key, data=value)
                with self.assertRaisesRegex(ValueError, message):
                    LowDimSequenceDataset(self.dataset, self.prepared / "train.jsonl")
                with h5py.File(path, "r+") as handle:
                    if key in handle:
                        del handle[key]
                    handle.create_dataset(key, data=old)

    def test_streaming_chunk_resets_gradients_and_config_contract(self):
        torch.manual_seed(4)
        model = LowDimBCRNNPolicy(hidden_dim=16, encoder_dim=12, sequence_length=3).eval()
        state = torch.randn(2, 8, 23)
        result = model(state)
        self.assertEqual(result["action"].shape, (2, 8, 7))
        self.assertTrue((result["action"].abs() <= 1).all())
        result["action"].square().mean().backward()
        self.assertGreater(model.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.lstm.weight_ih_l0.grad.abs().sum().item(), 0)
        with torch.inference_mode():
            expected = torch.cat([model(state[:, i:i + 3])["action"] for i in range(0, 8, 3)], 1)
            hidden, actions = None, []
            for i in range(8):
                output = model.step(state[:, i], hidden=hidden)
                self.assertEqual(output["hidden_state_reset"], i % 3 == 0)
                hidden = output["hidden"]
                actions.append(output["action"])
            torch.testing.assert_close(torch.stack(actions, 1), expected)
            torch.testing.assert_close(model.step(state[:, 0], previous_gripper=1)["action"],
                                       model.step(state[:, 0], previous_gripper=-1)["action"])
        restored = LowDimBCRNNPolicy.from_config(model.get_config())
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(state)["action"], result["action"])
        invalid = model.get_config()
        invalid["state_fields"].reverse()
        with self.assertRaisesRegex(ValueError, "observation order"):
            LowDimBCRNNPolicy.from_config(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["scene_counts"] = [[3, 1]]
        with self.assertRaisesRegex(ValueError, "1 cube and 1 tray"):
            validate_scene(invalid)

    def test_training_resume_exact_and_preserves_data_and_split(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                  for root in (self.dataset, self.prepared) for p in root.rglob("*") if p.is_file()}
        full, resumed = self.output / "full", self.output / "resume"
        with redirect_stdout(io.StringIO()):
            for command in (self.command(full, "--max-updates", "4"),
                            self.command(resumed, "--max-updates", "2"),
                            self.command(resumed, "--resume", str(resumed / "latest.pt"), "--max-updates", "4")):
                with patch("sys.argv", command):
                    training.main()
        a = torch.load(full / "latest.pt", weights_only=False)
        b = torch.load(resumed / "latest.pt", weights_only=False)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["updates"], 4)
        self.assertEqual(a["preprocessing"]["state_dim"], 23)
        for key in a["model_state"]:
            torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        with patch("sys.argv", self.command(resumed, "--resume", str(resumed / "latest.pt"),
                                            "--max-updates", "6", "--batch-size", "4")), \
                self.assertRaisesRegex(ValueError, "batch-size"):
            training.main()
        torch.save({"policy_type": "robomimic_bc_rnn"}, self.root / "vision.pt")
        with self.assertRaisesRegex(ValueError, "low_dim_bc_rnn"):
            evaluation.load_policy(self.root / "vision.pt", "cpu")

    def test_periodic_rollout_and_state_only_evaluator(self):
        with patch("sys.argv", self.command(None, "--max-updates", "2", "--rollout-eval-every", "2")), \
                patch.object(training, "run_rollout", return_value=dict(updates=2, success_rate=.2)) as rollout, \
                redirect_stdout(io.StringIO()):
            training.main()
        rollout.assert_called_once()
        checkpoint = self.output / "best_rollout.pt"
        created = []

        def build(seed, *args):
            env = StateEnv(seed)
            created.append(env)
            return env

        output = self.root / "eval"
        command = ["eval", "--checkpoint", str(checkpoint), "--output-dir", str(output), "--device", "cpu",
                   "--episodes", "2", "--num-seeds", "1", "--max-steps", "8", "--log-rollout"]
        with patch("sys.argv", command), patch.object(shared_evaluation, "build_environment", side_effect=build), \
                patch.object(shared_evaluation, "policy_observation", side_effect=AssertionError("image preprocessing")), \
                redirect_stdout(io.StringIO()):
            summary = evaluation.main()
        self.assertEqual(summary["tasks"][0]["failure_counts"], {"no_lift": 2})
        self.assertEqual(summary["rnn_reset_interval"], 3)
        self.assertEqual(summary["camera_names"], [])
        self.assertIsNone(summary["image_size"])
        self.assertTrue(summary["privileged_state"])
        self.assertTrue(all(env.closed for env in created))
        for path in (output / "rollouts").glob("*.jsonl"):
            steps = [r for line in path.read_text().splitlines() if (r := json.loads(line)).get("event") == "step"]
            self.assertEqual([r["step"] for r in steps if r["hidden_state_reset"]], [1, 4, 7])
            self.assertTrue(all(r["gripper_logit"] is None and r["previous_gripper_input"] is None for r in steps))

    def test_environment_cameras_enabled_only_for_optional_video(self):
        from scripts import record_demonstrations as recording

        model = LowDimBCRNNPolicy(hidden_dim=16)
        for video in (False, True):
            model.render_observation_cameras = video
            with patch.object(recording, "build_task", return_value=(object(), None)) as factory:
                shared_evaluation.build_environment(2000, 1, False, model, self.config, 300)
            args = factory.call_args.args[0]
            self.assertEqual(args.camera_obs, video)
            self.assertEqual(args.camera_names, ["agentview"])
            self.assertEqual(args.max_steps, 300)

    def test_rollout_command_and_infrastructure_errors(self):
        with patch("sys.argv", self.command(None, "--max-updates", "2")):
            args = training.parse_args()
        args.rollout_eval_episodes, args.rollout_eval_num_seeds = 10, 5
        args.rollout_eval_seed, args.rollout_eval_max_steps = 1000, 300
        output = self.output / "rollouts/update_0000002"
        output.mkdir(parents=True)
        for failures in ({}, {"exception": 1}):
            (output / "summary.json").write_text(json.dumps({"overall_success_rate": .4,
                                                            "tasks": [{"failure_counts": failures}]}))
            with patch.object(training.subprocess, "run") as run:
                if failures:
                    with self.assertRaisesRegex(RuntimeError, "infrastructure"):
                        training.run_rollout(args, 2, torch.device("cpu"))
                else:
                    self.assertEqual(training.run_rollout(args, 2, torch.device("cpu"))["success_rate"], .4)
            self.assertIn("scripts.evaluate_low_dim_bc_rnn", run.call_args.args[0])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
