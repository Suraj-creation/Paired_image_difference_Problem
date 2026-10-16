import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import binary_pair_metrics, mask_metrics, selective_pair_metrics
from src.localization.pixel_baseline import connected_regions, difference_map, filter_small_components


class CoreMetricTests(unittest.TestCase):
    def test_pair_confusion_counts(self):
        result = binary_pair_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["f1"], 0.5)

    def test_mask_exact_and_empty_conventions(self):
        truth = np.array([[0, 1], [0, 1]], dtype=bool)
        self.assertEqual(mask_metrics(truth, truth)["iou"], 1.0)
        empty = np.zeros((2, 2), dtype=bool)
        self.assertEqual(mask_metrics(empty, empty)["dice"], 1.0)

    def test_pixel_difference_and_components(self):
        before = np.zeros((5, 6, 3), dtype=np.uint8)
        after = before.copy()
        after[1:3, 2:5] = 255
        delta = difference_map(before, after)
        self.assertEqual(float(delta[1, 2]), 1.0)
        regions = connected_regions(delta >= 0.5)
        self.assertEqual(regions, [{"x0": 2, "y0": 1, "x1": 5, "y1": 3, "area": 6}])

    def test_reject_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            difference_map(np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((3, 2, 3), dtype=np.uint8))

    def test_min_area_keeps_components_not_bounding_box_contents(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        mask[3, 3] = True  # isolated interior pixel inside the large component's box
        result = filter_small_components(mask, min_area=2)
        self.assertFalse(result[3, 3])
        self.assertEqual(int(result.sum()), 24)

    def test_selective_metrics_exclude_abstentions_from_covered_accuracy(self):
        result = selective_pair_metrics([True, False, True], [True, False, None])
        self.assertEqual(result["covered"], 2)
        self.assertEqual(result["abstained"], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["selective_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
