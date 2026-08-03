from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.subset import grouped_subset_indices


class GroupedSubsetTests(unittest.TestCase):
    def test_samples_every_group_deterministically(self) -> None:
        groups = np.repeat([10, 20, 30], [10, 5, 2])
        first = grouped_subset_indices(groups, fraction=0.2, seed=42)
        second = grouped_subset_indices(groups, fraction=0.2, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(groups[first]), {10, 20, 30})
        self.assertEqual(len(first), 4)

    def test_full_fraction_keeps_every_row(self) -> None:
        groups = np.array([2, 1, 2, 1])
        np.testing.assert_array_equal(
            grouped_subset_indices(groups, fraction=1.0, seed=42),
            np.arange(4),
        )

    def test_rejects_invalid_fractions(self) -> None:
        for fraction in (0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    grouped_subset_indices(np.array([1, 1]), fraction, seed=42)


if __name__ == "__main__":
    unittest.main()
