"""Guard the maintained entry points, documentation commands, and repository layout."""

import importlib
import io
import re
import shlex
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class WorkspaceLayoutTests(unittest.TestCase):
    def test_two_platform_guides_and_local_links(self):
        docs = ROOT / "docs"
        self.assertEqual({path.name for path in docs.iterdir()}, {"WINDOWS.md", "LINUX.md"})
        for path in (ROOT / "README.md", *docs.glob("*.md")):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" not in target:
                    self.assertTrue((path.parent / target.split("#")[0]).exists(), (path, target))

    def test_documented_python_commands_parse_without_running_jobs(self):
        parsed = 0
        for path in (ROOT / "docs").glob("*.md"):
            blocks = re.findall(r"```(?:bash|bat)\n(.*?)```", path.read_text(encoding="utf-8"), re.S)
            for block in blocks:
                block = block.replace("\\\n", " ").replace("^\n", " ")
                for line in block.splitlines():
                    if not line.startswith("python -m "):
                        continue
                    tokens = shlex.split(line)
                    name = tokens[2]
                    if name in ("pip", "tests"):
                        continue
                    module = importlib.import_module(name)
                    self.assertTrue(callable(getattr(module, "main", None)), name)
                    parse = getattr(module, "parse_args", None)
                    if parse is None:
                        continue
                    args = tokens[3:]
                    with self.subTest(guide=path.name, command=name), patch("sys.argv", [name, *args]):
                        try:
                            with redirect_stdout(io.StringIO()):
                                parse()
                        except SystemExit as result:
                            self.assertEqual(result.code, 0, line)
                    parsed += 1
        self.assertGreaterEqual(parsed, 12)

    def test_only_maintained_models_are_exported(self):
        import models

        for name in models.__all__:
            self.assertIsNotNone(getattr(models, name))
        self.assertEqual(set(models.__all__), {"VisualBCPolicy", "VisualEncoder"})
        self.assertEqual({path.name for path in (ROOT / "models").glob("*.py")},
                         {"__init__.py", "visual_bc.py", "visual_encoder.py"})
        self.assertFalse((ROOT / "language").exists())
        self.assertFalse(list((ROOT / "scripts/legacy").glob("*.py")))
        self.assertEqual({path.name for path in (ROOT / "scripts").glob("*.py")},
                         {"__init__.py", "record_demonstrations.py", "train_visual_bc.py",
                          "train_robomimic_bc_rnn.py", "evaluate_robomimic_bc_rnn.py",
                          "evaluate_visual_bc.py", "evaluation_utils.py", "audit_dataset.py",
                          "audit_pose_distribution.py"})

    def test_artifact_paths_follow_explicit_inputs(self):
        dataset = ["--dataset", "chosen/demos"]
        cases = [
            ("scripts.audit_dataset", dataset, {"output": "results/dataset_audit/demos"}),
            ("scripts.train_visual_bc", [*dataset, "--output-dir", "chosen/run"],
             {"dataset": "chosen/demos", "output_dir": "chosen/run"}),
            ("scripts.evaluate_visual_bc", ["--checkpoint", "chosen/best.pt", "--output-dir", "chosen/eval"],
             {"checkpoint": "chosen/best.pt", "output_dir": "chosen/eval"}),
            ("tests.smoke.test_dataset_loader", dataset, {"dataset": "chosen/demos"}),
        ]
        for name, flags, expected in cases:
            module = importlib.import_module(name)
            with self.subTest(module=name), patch("sys.argv", [name, *flags]):
                args = module.parse_args()
                for key, value in expected.items():
                    self.assertEqual(getattr(args, key), Path(value))
            with self.subTest(missing_inputs=name), patch("sys.argv", [name]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    module.parse_args()
                self.assertEqual(result.exception.code, 2)

    def test_no_historical_dataset_defaults_and_generated_splits_are_ignored(self):
        pattern = re.compile(r"data/(?:demos|splits)/(?:pilot_v\d+|pick_fixed_red_v\d+)")
        for directory in ("scripts", "diagnostics", "tests/smoke"):
            for path in (ROOT / directory).glob("*.py"):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), str(path))
        self.assertIn("data/splits/", (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    def test_visual_workflow_has_no_unused_ml_dependencies(self):
        code = (
            "import sys; "
            "from scripts import train_visual_bc, evaluate_visual_bc; "
            "assert not any(name in sys.modules for name in "
            "('transformers', 'sentence_transformers', 'huggingface_hub', 'torchvision'))"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_exploratory_probes_or_root_generated_images(self):
        for filename in ("inspect_action.py", "inspect_env.py", "test_move_to.py",
                         "test_move_above_cube.py", "test_gripper.py", "test_env.py",
                         "test_pick_expert.py", "test_place_expert.py", "test_stack_expert.py"):
            self.assertFalse((ROOT / "scripts" / filename).exists(), filename)
        self.assertFalse(list(ROOT.glob("*.png")))
        self.assertFalse(list((ROOT / "scripts").glob("test_*.py")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
