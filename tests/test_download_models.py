import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_models import (
    DEFAULT_MODEL_VERSION,
    REQUIRED_FILES,
    VERSION_FILE,
    archive_root,
    download_models,
    latest_model_version,
    main,
    models_ready,
    normalize_version,
    safe_extract,
    sha256,
)


class DownloadModelsTests(unittest.TestCase):
    def make_archive(self, root: Path, version: str = DEFAULT_MODEL_VERSION) -> Path:
        package = root / archive_root(version)
        for relative in REQUIRED_FILES:
            path = package / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative)
        archive = root / f"{archive_root(version)}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(package, arcname=archive_root(version))
        return archive

    def test_downloads_verifies_and_installs_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.make_archive(root)
            destination = root / "models"

            result = download_models(
                destination,
                url=archive.as_uri(),
                expected_sha256=sha256(archive),
            )

            self.assertEqual(result, destination)
            self.assertTrue(models_ready(destination))
            self.assertEqual((destination / VERSION_FILE).read_text().strip(), DEFAULT_MODEL_VERSION)
            self.assertFalse((destination / archive_root(DEFAULT_MODEL_VERSION)).exists())

    def test_version_accepts_optional_v_prefix(self):
        self.assertEqual(normalize_version("1.0.0"), "1.0.0")
        self.assertEqual(normalize_version("v1.0.0"), "1.0.0")
        self.assertEqual(normalize_version("v-1.0.0"), "1.0.0")

    def test_rejects_invalid_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.make_archive(root)
            destination = root / "models"

            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                download_models(destination, url=archive.as_uri(), expected_sha256="0" * 64)

            self.assertFalse(destination.exists())

    def test_safe_extract_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                entry = tarfile.TarInfo("../outside")
                entry.size = 1
                bundle.addfile(entry, io.BytesIO(b"x"))

            with self.assertRaisesRegex(RuntimeError, "unsafe path"):
                safe_extract(archive, root / "output")

    @patch(
        "scripts.download_models.read_url",
        return_value=b'[{"tag_name":"v2.0.0"},{"tag_name":"models-v1.2.0"}]',
    )
    def test_resolves_latest_model_release(self, read_url):
        self.assertEqual(latest_model_version(), "1.2.0")
        read_url.assert_called_once()

    @patch("scripts.download_models.download_models")
    @patch("scripts.download_models.latest_model_version", return_value="1.2.0")
    def test_latest_cli_uses_resolved_version(self, latest, download):
        self.assertEqual(main(["--latest"]), 0)
        latest.assert_called_once_with()
        download.assert_called_once_with(
            Path(__file__).resolve().parents[1] / "models",
            version="1.2.0",
            force=False,
        )


if __name__ == "__main__":
    unittest.main()
