"""Paired validation scenes, action/contact events, and trajectory exports."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts import evaluate_visual_bc_validation as validation


def sample_state(step, contact=False, lifted=False, inside=False):
    return dict(step=step, time_s=step / 20, joint_pos=np.arange(7, dtype=float) + step / 100,
        joint_vel=np.full(7, .1), gripper_qpos=np.array([.02, -.02]), gripper_qvel=np.zeros(2),
        eef_pos=np.array([.01 * step, .02, .9]), eef_quat=np.array([0., 0., 0., 1.]),
        cube_pos=np.array([0., .1, .88 if lifted else .83]), cube_quat=np.array([0., 0., 0., 1.]),
        tray_pos=np.array([0., -.1, .829]), tray_quat=np.array([0., 0., 0., 1.]),
        two_finger_contact=contact,
        placement=dict(inside_target=inside, released=False, gripper_open=False, success=False),
        sim_qpos=np.zeros(23), sim_qvel=np.zeros(21), sim_ctrl=np.zeros(8), sim_act=np.empty(0))


class ValidationRolloutTests(unittest.TestCase):
    def test_cli_and_checkpoint_labels(self):
        args = ["eval", "--dataset", "chosen/data", "--training-dir", "chosen/split",
                "--checkpoints", "A=a.pt", "C=c.pt", "--output-dir", "chosen/output"]
        with patch("sys.argv", args):
            parsed = validation.parse_args()
        self.assertEqual(parsed.split, "val")
        self.assertEqual(parsed.max_steps, 300)
        self.assertEqual(validation.checkpoint_specs(parsed.checkpoints), {"A": Path("a.pt"), "C": Path("c.pt")})
        for values in (["A=a.pt", "A=b.pt"], ["../A=a.pt"], ["missing_separator"]):
            with self.assertRaises(ValueError):
                validation.checkpoint_specs(values)

    def test_quaternion_sign_and_inconsistent_initial_state(self):
        self.assertEqual(validation.quaternion_error([0, 0, 0, 1], [0, 0, 0, -1]), 0.)
        env = SimpleNamespace(reset=lambda: None)
        episode = validation.ValidationEpisode(env, {}, SimpleNamespace(), expected_hash="expected")
        with patch.object(validation, "restore_scene", return_value=({}, {"state_sha256": "different"})), \
                self.assertRaisesRegex(ValueError, "identical reconstructed"):
            episode.reset()

    @staticmethod
    def traced_episode():
        env = SimpleNamespace(step=lambda action: ({}, 0, False, {}), control_freq=20)
        expert = {"robot0_joint_pos": np.zeros((4, 7)), "robot0_eef_pos": np.tile([0., 0., .9], (4, 1)),
                  "robot0_gripper_qpos": np.zeros((4, 2)), "object_0_pos": np.tile([0., .1, .83], (4, 1))}
        scene = dict(entry={"episode": 5, "path": "place/episode_000005.hdf5"}, expert=expert,
                     metadata={"target_object": "object_0"})
        episode = validation.ValidationEpisode(env, scene, SimpleNamespace())
        episode.reconstruction = {"state_sha256": "same", "exact_original_physics_replay": False}
        episode.states = [sample_state(0)]
        states = [sample_state(1), sample_state(2, contact=True, lifted=True),
                  sample_state(3, lifted=True, inside=True)]
        with patch.object(episode, "snapshot", side_effect=states):
            for command in (1, 1, -1):
                episode.step(np.array([0., 0., 0., 0., 0., 0., command]))
        return episode

    def test_command_contact_and_lift_are_distinct_events(self):
        episode = self.traced_episode()
        events = episode.events
        self.assertEqual([e["step"] for e in events if e["event"] == "close_command"], [1])
        self.assertEqual([e["step"] for e in events if e["event"] == "two_finger_contact_on"], [2])
        self.assertEqual([e["step"] for e in events if e["event"] == "lift_above_initial_2cm"], [2])
        self.assertEqual([e["step"] for e in events if e["event"] == "open_command"], [3])
        close = events[0]
        self.assertEqual(close["command_time_s"], 0.)
        self.assertEqual(close["observation_time_s"], .05)
        self.assertEqual(len(close["joint_pos_before"]), 7)

    def test_complete_numeric_and_plot_exports(self):
        import matplotlib
        matplotlib.use("Agg")
        episode = self.traced_episode()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            policy = [{"event": "episode_start"}]
            policy += [dict(event="step", step=i + 1, raw_action=a, normalized_action=a,
                            gripper_logit=float(a[-1])) for i, a in enumerate(episode.actions)]
            validation.write_jsonl(folder / "trajectory.jsonl", policy)
            data = validation.save_numeric(folder, episode)
            saved = np.load(folder / "trajectory.npz", allow_pickle=False)
            self.assertEqual(saved["joint_pos"].shape, (4, 7))
            self.assertEqual(saved["actions"].shape, (3, 7))
            self.assertEqual(saved["sim_act"].shape, (4, 0))
            with (folder / "trajectory.csv").open() as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 4)
            self.assertEqual(csv_rows[0]["action_6"], "")
            self.assertIn("close_command", csv_rows[1]["events"])
            parsed = [json.loads(s) for s in (folder / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(parsed[1]["robot_state_before"]["step"], 0)
            self.assertEqual(parsed[1]["robot_state_after"]["step"], 1)
            validation.plot_episode(folder, episode, data, dict(success=False, failure_reason="no_lift"), "A")
            for name in ("trajectory.png", "joints.png"):
                self.assertIsNotNone(cv2.imread(str(folder / name)))

    def test_d_dispatch_uses_raw_inputs_source_size_and_checkpoint_reset(self):
        model = SimpleNamespace(sequence_length=10, observation_image_size=128)
        checkpoint = {"data_config": {"fixed_target_color": "red"}, "updates": 500}
        with patch.object(validation.torch, "load", return_value={"policy_type": "robomimic_bc_rnn"}), \
                patch.object(validation, "load_bcrnn_policy", return_value=(model, checkpoint)), \
                patch.object(validation, "load_visual_policy") as visual:
            loaded = validation.load_validation_policy(Path("D.pt"), torch.device("cpu"), 16)
        self.assertIs(loaded[0], model)
        self.assertEqual(loaded[3], 128)
        self.assertEqual(loaded[5], 10)
        state = torch.arange(9, dtype=torch.float32)
        self.assertTrue(torch.equal(loaded[1].normalize_proprio(state), state))
        self.assertTrue(torch.equal(loaded[1].denormalize_action(state), state))
        visual.assert_not_called()

    def test_d_exports_missing_logits_without_pickled_numpy_arrays(self):
        episode = self.traced_episode()
        episode.policy_type = "robomimic_bc_rnn"
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            policy = [{"event": "episode_start"}]
            policy += [dict(event="step", step=i + 1, raw_action=a, normalized_action=a,
                            gripper_logit=None) for i, a in enumerate(episode.actions)]
            validation.write_jsonl(folder / "trajectory.jsonl", policy)
            validation.save_numeric(folder, episode)
            with np.load(folder / "trajectory.npz", allow_pickle=False) as saved:
                self.assertTrue(np.isnan(saved["gripper_logits"]).all())
                self.assertEqual(saved["predicted_actions"].shape, (3, 7))
                self.assertNotIn("normalized_actions", saved)
            header = json.loads((folder / "trajectory.jsonl").read_text().splitlines()[0])
            self.assertEqual(header["prediction_space"], "raw_controller")

    def test_real_d_checkpoint_numpy_rng_passes_dispatch_and_preflight(self):
        prepared = dict(tasks=["place"], scene_counts=[[1, 1]], fixed_target_color="red")
        checkpoint = dict(policy_type="robomimic_bc_rnn", data_config=prepared,
                          numpy_rng_state=np.random.get_state(), epoch=1, updates=500)
        model = SimpleNamespace(sequence_length=10, observation_image_size=128)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "D.pt"
            torch.save(checkpoint, path)
            # Unlike the lightweight mock fixture, this exercises NumPy's
            # serialized reconstruction function in a real training payload.
            with patch.object(validation, "load_bcrnn_policy", return_value=(model, checkpoint)):
                loaded = validation.load_validation_policy(path, torch.device("cpu"), 0)
            self.assertEqual(loaded[2]["updates"], 500)
            self.assertEqual(loaded[5], 10)
            (root / "val.jsonl").write_text(json.dumps(dict(task="place", episode=0,
                path="place/episode_000000.hdf5", steps=2)) + "\n")
            flags = ["eval", "--dataset", str(root), "--training-dir", str(root),
                     "--checkpoints", f"D={path}", "--output-dir", str(root / "output"), "--device", "cpu"]
            with patch("sys.argv", flags), patch.object(validation, "load_bc_training", return_value=(prepared, root)), \
                    patch.object(validation, "load_validation_policy", side_effect=RuntimeError("passed preflight")), \
                    self.assertRaisesRegex(RuntimeError, "passed preflight"):
                validation.main()

    def test_summary_distinguishes_failure_success_and_pending(self):
        results = {"A": [dict(source_path="place/1.hdf5", source_episode=1,
                              success=False, failure_reason="no_lift", steps=300)]}
        entries = [{"episode": 1, "path": "place/1.hdf5"}, {"episode": 2, "path": "place/2.hdf5"}]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            validation.write_episode_summary(folder, results, entries, ["A", "D"])
            rows = json.loads((folder / "episode_summary.json").read_text())
            self.assertEqual(rows[0]["models"]["A"]["status"], "FAILURE")
            self.assertEqual(rows[0]["models"]["A"]["failure_reason"], "no_lift")
            self.assertEqual(rows[0]["models"]["D"]["status"], "PENDING")
            self.assertIsNone(rows[0]["models"]["D"]["success"])
            self.assertIn("FAIL: no_lift (300)", (folder / "episode_summary.md").read_text())
            self.assertTrue((folder / "D/episode_summary.csv").exists())

    def test_reuse_checks_same_scenes_and_protocol(self):
        args = SimpleNamespace(split="val", max_steps=300, seed=123)
        collection = {"method": "visual_bc"}
        prepared = {"environment_configs": {"place": collection}}
        entries = [{"episode": 5, "path": "place/5.hdf5"}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            validation.save_json(root / "config.json", vars(args))
            for name in ("A", "B"):
                folder = root / name
                folder.mkdir()
                validation.save_json(folder / "summary.json", dict(attempted=1, scene_origin="reconstructed_dataset_split", success_hold_steps=20))
                validation.write_jsonl(folder / "episodes.jsonl", [dict(source_path="place/5.hdf5", source_episode=5,
                    split="val", success=True, failure_reason=None, steps=200,
                    metadata={"collection_config": collection}, reconstruction={"state_sha256": "same"})])
            results, hashes, _ = validation.load_reused_results(root, entries, args, prepared)
            self.assertEqual(set(results), {"A", "B"})
            self.assertEqual(hashes, {"place/5.hdf5": "same"})
            b = results["B"][0]
            b["reconstruction"]["state_sha256"] = "different"
            validation.write_jsonl(root / "B/episodes.jsonl", [b])
            with self.assertRaisesRegex(ValueError, "inconsistent initial"):
                validation.load_reused_results(root, entries, args, prepared)
            args.max_steps = 400
            with self.assertRaisesRegex(ValueError, "same split"):
                validation.load_reused_results(root, entries, args, prepared)

    def test_paired_comparison_and_no_silent_overwrite(self):
        import matplotlib
        matplotlib.use("Agg")
        base = [dict(source_path=f"place/{i}.hdf5", source_episode=i, steps=10,
                     failure_reason=None, success=value) for i, value in enumerate((True, False, True))]
        other = copy.deepcopy(base)
        other[1]["success"], other[2]["success"] = True, False
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            validation.compare_results(folder, {"A": base, "B": other})
            report = json.loads((folder / "comparison.json").read_text())
            self.assertEqual(report["paired_counts"]["A_vs_B"],
                             dict(both_success=1, a_only=1, b_only=1, both_fail=0))
            flags = ["eval", "--dataset", "unused", "--training-dir", "unused",
                     "--checkpoints", "A=unused.pt", "--output-dir", str(folder)]
            with patch("sys.argv", flags), self.assertRaises(FileExistsError):
                validation.main()

    def test_incomplete_checkpoint_fails_before_dataset_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "broken.pt"
            checkpoint.write_bytes(b"PK truncated checkpoint")
            flags = ["eval", "--dataset", "unused", "--training-dir", "unused", "--checkpoints",
                     f"C={checkpoint}", "--output-dir", str(root / "out")]
            with patch("sys.argv", flags), patch.object(validation, "load_bc_training") as load, \
                    self.assertRaisesRegex(ValueError, "missing/incomplete checkpoint"):
                validation.main()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
