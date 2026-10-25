"""Audit native action lengths and split budgets without changing recordings."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data.trajectory_recorder import TrajectoryRecorder
from scripts import audit_dataset as audit
from tests.regression.test_variable_scene_counts import FakeEnv


class DatasetAuditStatisticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "demos"
        self.split_dir = self.root / "splits"
        self.output = self.root / "audit"
        (self.dataset / "place").mkdir(parents=True)
        self.split_dir.mkdir()
        self.entries = []
        for episode, length in enumerate((3, 9, 17)):
            env = FakeEnv("place", 1, 1, episode)
            recorder = TrajectoryRecorder(env, phase_provider=lambda: "approach")
            recorder.start(env.reset())
            for _ in range(length):
                env.step(np.zeros(7, dtype=np.float32))
            recorder.stop()
            name = f"episode_{episode:06d}.hdf5"
            summary = recorder.save(self.dataset / "place" / name, env.metadata,
                                    dict(success=True, stage="complete"))
            self.entries.append(dict(task="place", episode=episode, path=name, steps=length,
                                     instruction=env.metadata["instruction"], metadata=env.metadata,
                                     phase_counts=summary["phase_counts"],
                                     camera_names=["agentview", "robot0_eye_in_hand"]))
        self.write_jsonl(self.dataset / "place/manifest.jsonl", self.entries)
        self.train = [{**e, "path": "place/" + e["path"]} for e in self.entries[:2]]
        self.val = [{**e, "path": "place/" + e["path"]} for e in self.entries[2:]]
        self.write_jsonl(self.split_dir / "train.jsonl", self.train)
        self.write_jsonl(self.split_dir / "val.jsonl", self.val)

    @staticmethod
    def write_jsonl(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_audit(self, *extra, explicit_split=True):
        args = ["audit", "--dataset", str(self.dataset), "--tasks", "place",
                "--camera-names", "agentview", "robot0_eye_in_hand", "--batch-size", "2",
                "--output", str(self.output)]
        if explicit_split:
            args += ["--split-dir", str(self.split_dir)]
        stream = io.StringIO()
        with patch("sys.argv", args + list(extra)), redirect_stdout(stream), self.assertRaises(SystemExit) as exit:
            audit.main()
        report = json.loads((self.output / "audit_report.json").read_text())
        return exit.exception.code, report, stream.getvalue()

    def test_counts_real_action_steps_and_training_budgets(self):
        original = {path: path.read_bytes() for path in self.dataset.rglob("*") if path.is_file()}
        code, report, stdout = self.run_audit()
        self.assertEqual(code, 0)
        stats = report["trajectory_steps"]
        self.assertEqual((stats["count"], stats["total"], stats["median"], stats["min"], stats["max"]), (3, 29, 9., 3, 17))
        self.assertAlmostEqual(stats["mean"], 29 / 3)
        self.assertEqual(report["tasks"]["place"]["trajectory_steps"], stats)
        splits = report["split_summary"]["splits"]
        self.assertEqual(splits["train"]["trajectory_steps"]["total"], 12)
        self.assertEqual(splits["val"]["trajectory_steps"]["total"], 17)
        budget = splits["train"]["training_budget"]
        self.assertEqual(budget["visual_bc_windows"], 3)  # ceil(3/8) + ceil(9/8)
        self.assertEqual(budget["visual_bc_updates_per_epoch"], 2)  # partial final batch counts
        self.assertEqual(budget["bcrnn_updates_per_full_pass"], 6)  # floor(12/2)
        self.assertEqual(budget["visual_bc_updates_in_100_epochs"], 200)
        self.assertEqual(budget["bcrnn_updates_in_100_full_passes"], 600)
        with (self.output / "episode_lengths.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["steps"]) for row in rows], [3, 9, 17])
        self.assertTrue(all(row["valid"] == "True" for row in rows))
        self.assertIn("total=29 action steps", stdout)
        self.assertIn("Total T", (self.output / "audit_report.md").read_text())
        self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_estimate_options_and_automatic_split_discovery(self):
        with patch.object(audit, "default_training_dir", return_value=self.split_dir):
            code, report, _ = self.run_audit("--batch-size", "5", "--sequence-stride", "2",
                                             "--gradient-accumulation-steps", "2", explicit_split=False)
        self.assertEqual(code, 0)
        budget = report["split_summary"]["splits"]["train"]["training_budget"]
        self.assertEqual(budget["visual_bc_windows"], 7)  # ceil(3/2) + ceil(9/2)
        self.assertEqual(budget["visual_bc_batches_per_epoch"], 2)
        self.assertEqual(budget["visual_bc_updates_per_epoch"], 1)
        self.assertEqual(budget["bcrnn_updates_per_full_pass"], 2)  # drops two samples

    def test_stale_steps_and_split_overlap_fail_the_audit(self):
        self.train[0]["steps"] = 999
        self.write_jsonl(self.split_dir / "train.jsonl", self.train)
        self.write_jsonl(self.split_dir / "val.jsonl", self.val + self.train[:1])
        code, report, _ = self.run_audit()
        self.assertEqual(code, 1)
        errors = report["split_summary"]["errors"]
        self.assertTrue(any("steps differ" in value for value in errors))
        self.assertTrue(any("overlapping" in value for value in errors))
        self.assertEqual(report["split_summary"]["splits"]["train"]["trajectory_steps"]["total"], 12)

    def test_missing_auto_splits_do_not_block_dataset_statistics(self):
        with patch.object(audit, "default_training_dir", return_value=self.root / "absent"):
            code, report, stdout = self.run_audit(explicit_split=False)
        self.assertEqual(code, 0)
        self.assertEqual(report["split_summary"]["status"], "unavailable")
        self.assertEqual(report["trajectory_steps"]["total"], 29)
        self.assertIn("Split manifests unavailable", stdout)

    def test_limited_audit_keeps_full_split_step_counts(self):
        code, report, _ = self.run_audit("--limit-per-task", "1")
        self.assertEqual(code, 0)
        self.assertEqual(report["trajectory_steps"]["total"], 3)
        self.assertEqual(report["split_summary"]["splits"]["train"]["trajectory_steps"]["total"], 12)
        self.assertEqual(report["split_summary"]["splits"]["val"]["trajectory_steps"]["total"], 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
