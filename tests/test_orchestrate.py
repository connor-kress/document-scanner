import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from preprocess.download import processed_data_ready, raw_data_ready, safe_extract
from scripts.orchestrate import pipeline_steps, training_steps


class OrchestrateTests(unittest.TestCase):
    def test_raw_data_requires_metadata_and_all_backgrounds(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory)
            (frames / "metadata.csv.gz").touch()
            for index in range(1, 6):
                (frames / f"background{index:02d}").mkdir()
            self.assertTrue(raw_data_ready(frames))
            (frames / "background05").rmdir()
            self.assertFalse(raw_data_ready(frames))

    def test_safe_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("../outside.txt")
                payload = b"unsafe"
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))

            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                safe_extract(archive, root / "output")

    def test_safe_extract_extracts_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "good.tar.gz"
            output = root / "output"
            output.mkdir()
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("frames/metadata.csv.gz")
                payload = b"metadata"
                member.size = len(payload)
                bundle.addfile(member, io.BytesIO(payload))

            safe_extract(archive, output)
            self.assertEqual((output / "frames" / "metadata.csv.gz").read_bytes(), payload)

    def test_training_searches_retrain_each_neural_model(self):
        steps = training_steps()
        searches = [command for command, _ in steps if "scripts/grid_search.py" in command]
        self.assertEqual(len(searches), 3)
        self.assertTrue(all("--train-best" in command for command in searches))
        self.assertEqual(steps[-1][0][1], "scripts/final_eval.py")

    def test_pipeline_prepares_data_before_training(self):
        steps = pipeline_steps(2)
        self.assertEqual(steps[0][0][1:], ["scripts/build_data.py", "all", "--workers", "2"])

    def test_processed_data_requires_valid_manifests_and_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory)
            (processed / "arrays").mkdir()
            (processed / "frames_384").mkdir()
            (processed / "qc.csv").write_text("ok\n1\n")
            (processed / "manifest_subset.csv").write_text("ok\n1\n")
            (processed / "frames_384" / "labels.csv").write_text("ok\n1\n")
            (processed / "manifest.csv").write_text(
                "image_path,video_id,usable,split_video,split_doc\nframe.jpg,1,True,train,train\n"
            )
            for name in ("tab_64x36_clahe", "img_128x72_rgb"):
                with zipfile.ZipFile(processed / "arrays" / f"{name}.npz", "w") as bundle:
                    for member in ("X.npy", "y.npy", "split_video.npy", "video_id.npy"):
                        bundle.writestr(member, b"data")

            self.assertTrue(processed_data_ready(processed))
            (processed / "manifest_subset.csv").unlink()
            self.assertFalse(processed_data_ready(processed))


if __name__ == "__main__":
    unittest.main()
