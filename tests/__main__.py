"""Run the offline regression suite, without dataset downloads or real rollouts."""

from pathlib import Path
import unittest

import torch


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent.parent
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests" / "regression"), top_level_dir=str(root),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
