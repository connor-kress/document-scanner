import unittest
from unittest.mock import patch

from preprocess.cli import main


class BuildDataTests(unittest.TestCase):
    @patch("preprocess.cli.ensure_raw_data")
    def test_download_only_ensures_raw_data(self, ensure_raw_data):
        self.assertEqual(main(["download"]), 0)
        ensure_raw_data.assert_called_once_with()

    @patch("preprocess.cli.run_qc")
    @patch("preprocess.cli.processed_data_ready", return_value=True)
    @patch("preprocess.cli.ensure_raw_data")
    def test_all_reuses_valid_processed_data(
        self, ensure_raw_data, processed_data_ready, run_qc
    ):
        self.assertEqual(main(["all"]), 0)
        ensure_raw_data.assert_called_once_with()
        processed_data_ready.assert_called_once_with()
        run_qc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
