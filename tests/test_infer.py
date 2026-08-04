import argparse
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(
            infer.validate_selection(infer.build_parser(), arguments()),
            "cnn-rgb",
        )

    def test_custom_artifacts_require_explicit_model_type(self):
        with self.assertRaises(SystemExit):
            infer.validate_selection(
                infer.build_parser(),
                arguments(weights=Path("model.pt")),
            )

    def test_latest_trained_rejects_custom_artifacts(self):
        with self.assertRaises(SystemExit):
            infer.validate_selection(
                infer.build_parser(),
                arguments(model_type="mlp", weights=Path("model.pt"), latest_trained=True),
            )

    def test_custom_mlp_requires_configured_pca(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "model.pt"
            weights.touch()
            (root / "config.yaml").write_text("use_pca: true\n")

            with self.assertRaisesRegex(FileNotFoundError, "requires a PCA"):
                infer.custom_artifacts("mlp", weights, None, None)

    def test_latest_trained_requires_complete_run(self):
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
                infer.released_artifacts("cnn-rgb")

            download.assert_called_once_with(models, force=False)


class InferenceTests(unittest.TestCase):
    def test_preprocessing_shapes(self):
        image = np.full((80, 120, 3), 127, dtype=np.uint8)

        mlp = infer.preprocess_mlp(
            image,
            {"input_source": "tab_64x36_clahe", "preprocessing": "none"},
            None,
        )
        cnn = infer.preprocess_cnn(
            image,
            "cnn-rgb",
            {"color_mode": "rgb", "image_size": [384, 216]},
        )

        self.assertEqual(mlp.shape, (1, 64 * 36))
        self.assertEqual(cnn.shape, (1, 3, 216, 384))

    def test_pixel_corners_use_original_image_dimensions(self):
        prediction = np.array([0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8])

        corners = infer.pixel_corners(prediction, width=1000, height=500)

        np.testing.assert_allclose(
            corners,
            [[100, 100], [900, 100], [900, 400], [100, 400]],
        )


class InferenceCliTests(unittest.TestCase):
    def run_main(self, options):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        prediction = np.full(8, 0.5, dtype=np.float32)
        with (
            patch("scripts.infer.load_image", return_value=image),
            patch("scripts.infer.resolve_artifacts", return_value=infer.Artifacts(Path("model.pt"))),
            patch("scripts.infer.predict", return_value=prediction),
            patch("scripts.infer.display_corners") as display,
            redirect_stdout(StringIO()) as output,
        ):
            result = infer.main(["document.jpg", *options])
        return result, output.getvalue(), display

    def test_main_prints_and_displays_corners(self):
        result, output, display = self.run_main([])

        self.assertEqual(result, 0)
        self.assertIn("corner_0: (100.00, 50.00)", output)
        self.assertEqual(display.call_args.kwargs, {"show": True, "output": None})

    def test_no_show_skips_figure(self):
        _, _, display = self.run_main(["--no-show"])

        display.assert_not_called()

    def test_output_can_save_without_showing(self):
        _, _, display = self.run_main(
            ["--no-show", "--output", "predictions/result.png"]
        )

        self.assertEqual(
            display.call_args.kwargs,
            {"show": False, "output": Path("predictions/result.png")},
        )


if __name__ == "__main__":
    unittest.main()
