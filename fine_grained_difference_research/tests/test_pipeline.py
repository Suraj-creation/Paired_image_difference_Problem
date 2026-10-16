from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from src.localization.photometric import compensate_global_photometric_change
from src.pipeline.analyzer import analyze_pair
from src.models.qwen_captioner import _parse_json_object
from src.localization.registration import estimate_global_translation


class PipelineTests(unittest.TestCase):
    def test_global_brightness_is_suppressed_only_in_secondary_view(self):
        before = np.full((64, 64, 3), 80, dtype=np.uint8)
        after = np.clip(before.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        before[20:32, 20:32] = (20, 40, 60)
        after[20:32, 20:32] = (50, 70, 90)
        residual, transform = compensate_global_photometric_change(before, after)
        self.assertLess(float(residual.mean()), 0.03)
        self.assertEqual(len(transform["gain_rgb"]), 3)

    def test_photometric_fit_stays_stable_on_flat_channels(self):
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        before[:, :32] = (235, 233, 224)
        before[:, 32:] = (192, 180, 158)
        after = np.clip(before.astype(np.float32) * 1.22, 0, 255).astype(np.uint8)
        residual, info = compensate_global_photometric_change(before, after)
        self.assertLess(float(np.quantile(residual, 0.95)), 0.035)
        self.assertTrue(np.isfinite(info["transform_strength"]))

    def test_parser_normalizes_change_type_misplaced_in_status(self):
        value = _parse_json_object('{"status":"removal","changes":[{"type":"removal"}]}')
        self.assertEqual(value["status"], "changed")
        self.assertEqual(value["model_status_normalized_from"], "removal")

    def test_translation_estimate_flags_camera_shift_without_warping(self):
        before = np.zeros((64, 64, 3), dtype=np.uint8)
        before[12:30, 14:35] = (150, 80, 30)
        after = np.zeros_like(before)
        after[16:34, 22:43] = (150, 80, 30)
        result = estimate_global_translation(before, after)
        self.assertTrue(result["available"])
        self.assertTrue(result["candidate"])
        self.assertGreater(abs(result["estimated_shift_xy_px"][0]), 5)

    def test_pipeline_saves_masks_regions_and_requires_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = Image.new("RGB", (64, 64), "white")
            after = before.copy()
            ImageDraw.Draw(after).rectangle((10, 10, 17, 17), fill="red")
            before.save(root / "before.png")
            after.save(root / "after.png")
            result = analyze_pair(root / "before.png", root / "after.png", root / "out", min_area=4)
            self.assertTrue((root / "out/mask_raw.png").is_file())
            self.assertTrue((root / "out/mask_photometric_residual.png").is_file())
            self.assertTrue(result["grounding"]["review_required"])
            self.assertTrue(result["analysis"]["raw_pixel"]["candidate_change_present"])
            self.assertTrue((root / "out/analysis.json").is_file())

    def test_pipeline_refuses_implicit_resizing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (10, 10), "white").save(root / "before.png")
            Image.new("RGB", (11, 10), "white").save(root / "after.png")
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                analyze_pair(root / "before.png", root / "after.png", root / "out")


if __name__ == "__main__":
    unittest.main()
