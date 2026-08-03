from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.runs import latest_run, run_directory


class RunDirectoryTests(unittest.TestCase):
    def test_completed_runs_are_unique_and_latest_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with run_directory(root, "mlp") as first:
                (first / "best.pt").touch()
            with run_directory(root, "mlp") as second:
                (second / "best.pt").touch()

            self.assertNotEqual(first, second)
            self.assertEqual(latest_run(root, "mlp", required=("best.pt",)), second)
            self.assertTrue((second / "COMPLETE").is_file())

    def test_incomplete_and_failed_runs_are_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with run_directory(root, "cnn_gray") as complete:
                (complete / "best.pt").touch()
            incomplete = root / "cnn_gray" / "20990101T000000.000000Z"
            incomplete.mkdir()
            (incomplete / "best.pt").touch()
            try:
                with run_directory(root, "cnn_gray") as failed:
                    (failed / "best.pt").touch()
                    raise RuntimeError("training failed")
            except RuntimeError:
                pass

            self.assertEqual(latest_run(root, "cnn_gray", required=("best.pt",)), complete)

    def test_legacy_layout_is_a_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "ridge"
            legacy.mkdir()
            (legacy / "best_pipeline.joblib").touch()

            self.assertEqual(
                latest_run(root, "ridge", required=("best_pipeline.joblib",)),
                legacy,
            )


if __name__ == "__main__":
    unittest.main()
