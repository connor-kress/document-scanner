import argparse
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts import infer


def arguments(**overrides):
    values = {
        "image": Path("document.jpg"),
        "model_type": None,
        "weights": None,
        "pca": None,
        "config": None,
        "latest_trained": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class InferenceSelectionTests(unittest.TestCase):
    def test_defaults_to_released_rgb_cnn(self):
        args = arguments()

        self.assertEqual(infer.validate_selection(infer.build_parser(), args), "cnn-rgb")

    def test_custom_artifacts_require_explicit_model_type(self):
        args = arguments(weights=Path("model.pt"))

        with self.assertRaises(SystemExit):
            infer.validate_selection(infer.build_parser(), args)

    def test_latest_trained_rejects_custom_artifacts(self):
        args = arguments(model_type="mlp", weights=Path("model.pt"), latest_trained=True)

        with self.assertRaises(SystemExit):
            infer.validate_selection(infer.build_parser(), args)

    def test_custom_mlp_uses_adjacent_config_and_pca(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "model.pt"
            config = root / "config.yaml"
            pca = root / "pca.joblib"
            weights.touch()
            pca.touch()
            config.write_text("use_pca: true\n")

            artifacts = infer.custom_artifacts("mlp", weights, None, pca)

            self.assertEqual(artifacts, infer.Artifacts(weights, config, pca))

    def test_custom_mlp_requires_pca_when_configured(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "model.pt"
            weights.touch()
            (root / "config.yaml").write_text("use_pca: true\n")

            with self.assertRaisesRegex(FileNotFoundError, "requires a PCA"):
                infer.custom_artifacts("mlp", weights, None, None)

    @patch("scripts.infer.released_artifacts")
    @patch("scripts.infer.latest_run", side_effect=FileNotFoundError("no completed run"))
    def test_latest_trained_does_not_fall_back_to_release(self, latest, released):
        with self.assertRaisesRegex(FileNotFoundError, "no completed run"):
            infer.trained_artifacts("cnn-rgb")

        latest.assert_called_once()
        released.assert_not_called()

    def test_latest_trained_resolves_completed_mlp_without_pca(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            run = results / "mlp" / "20260804T120000.000000Z"
            run.mkdir(parents=True)
            (run / "best.pt").touch()
            (run / "config.yaml").write_text("use_pca: false\n")
            (run / "COMPLETE").touch()

            with patch("scripts.infer.RESULTS", results):
                artifacts = infer.trained_artifacts("mlp")

            self.assertEqual(artifacts, infer.Artifacts(run / "best.pt", run / "config.yaml"))

    def test_latest_trained_rejects_legacy_result_without_complete_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            directory = results / "ridge"
            directory.mkdir(parents=True)
            (directory / "best_pipeline.joblib").touch()

            with (
                patch("scripts.infer.RESULTS", results),
                self.assertRaisesRegex(FileNotFoundError, "not a completed training run"),
            ):
                infer.trained_artifacts("ridge")

    def test_missing_release_artifacts_trigger_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary) / "models"

            def install(destination, **kwargs):
                directory = destination / "cnn_rgb"
                directory.mkdir(parents=True)
                (directory / "model.pt").touch()
                (directory / "config.yaml").touch()

            with (
                patch("scripts.infer.MODELS", models),
                patch("scripts.infer.models_ready", return_value=False),
                patch("scripts.infer.download_models", side_effect=install) as download,
            ):
                artifacts = infer.released_artifacts("cnn-rgb")

            self.assertEqual(artifacts.weights, models / "cnn_rgb" / "model.pt")
            download.assert_called_once_with(models, force=False)


class InferencePreprocessingTests(unittest.TestCase):
    def test_mlp_clahe_preprocessing_has_training_shape_and_scale(self):
        image = np.full((80, 120, 3), 127, dtype=np.uint8)

        values = infer.preprocess_mlp(
            image,
            {"input_source": "tab_64x36_clahe", "preprocessing": "none"},
            None,
        )

        self.assertEqual(values.shape, (1, 64 * 36))
        self.assertEqual(values.dtype, np.float32)
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)

        resized = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        np.testing.assert_array_equal(values, expected.reshape(1, -1).astype(np.float32) / 255.0)

    def test_mlp_rgb_to_gray_and_sharpening_match_training_order(self):
        image = np.arange(72 * 128 * 3, dtype=np.uint8).reshape(72, 128, 3)

        values = infer.preprocess_mlp(
            image,
            {
                "input_source": "img_128x72_rgb",
                "preprocessing": "rgb_to_gray",
                "sharpened": True,
            },
            None,
        )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)
        expected = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        np.testing.assert_array_equal(values, expected.reshape(1, -1).astype(np.float32) / 255.0)

    def test_rgb_cnn_converts_bgr_and_normalizes_channels(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[:, :, 2] = 255

        values = infer.preprocess_cnn(
            image,
            "cnn-rgb",
            {"color_mode": "rgb", "image_size": [2, 2]},
        )

        self.assertEqual(values.shape, (1, 3, 2, 2))
        expected_red = (1.0 - 0.6346) / 0.1559
        expected_green = (0.0 - 0.5556) / 0.1483
        self.assertAlmostEqual(float(values[0, 0, 0, 0]), expected_red, places=4)
        self.assertAlmostEqual(float(values[0, 1, 0, 0]), expected_green, places=4)

    def test_gray_cnn_uses_per_image_normalization(self):
        image = np.array(
            [[[0, 0, 0], [255, 255, 255]], [[64, 64, 64], [128, 128, 128]]],
            dtype=np.uint8,
        )

        values = infer.preprocess_cnn(
            image,
            "cnn-gray",
            {"color_mode": "gray", "image_size": [2, 2]},
        )

        self.assertEqual(values.shape, (1, 1, 2, 2))
        self.assertAlmostEqual(float(values.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(values.std()), 1.0, places=4)

    def test_pixel_corners_use_original_image_dimensions(self):
        prediction = np.array([0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])

        corners = infer.pixel_corners(prediction, width=1000, height=500)

        np.testing.assert_allclose(
            corners,
            [[100, 100], [900, 100], [900, 400], [100, 400]],
        )

    def test_unreadable_image_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.jpg"

            with (
                patch("scripts.infer.cv2.imread", return_value=None),
                self.assertRaisesRegex(FileNotFoundError, "could not read image"),
            ):
                infer.load_image(path)


class InferenceCliTests(unittest.TestCase):
    @patch("scripts.infer.predict", return_value=np.full(8, 0.5, dtype=np.float32))
    @patch("scripts.infer.resolve_artifacts", return_value=infer.Artifacts(Path("model.pt")))
    @patch("scripts.infer.load_image", return_value=np.zeros((100, 200, 3), dtype=np.uint8))
    def test_main_prints_corners_and_uses_default_model(self, load_image, resolve, predict):
        output = StringIO()

        with redirect_stdout(output):
            result = infer.main(["document.jpg"])

        self.assertEqual(result, 0)
        resolve.assert_called_once()
        self.assertEqual(resolve.call_args.args[0], "cnn-rgb")
        predict.assert_called_once()
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "corner_0: (100.00, 50.00)",
                "corner_1: (100.00, 50.00)",
                "corner_2: (100.00, 50.00)",
                "corner_3: (100.00, 50.00)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
