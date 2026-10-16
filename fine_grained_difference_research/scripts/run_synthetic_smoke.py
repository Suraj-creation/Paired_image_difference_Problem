#!/usr/bin/env python3
"""Deterministic mechanics demonstration on a generated square change."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import binary_pair_metrics, mask_metrics
from src.localization.pixel_baseline import connected_regions, difference_map, overlay_image

before = np.full((64, 64, 3), 128, dtype=np.uint8)
after = before.copy()
after[19:25, 31:38] = (240, 30, 30)
truth = np.zeros((64, 64), dtype=bool)
truth[19:25, 31:38] = True
pred = difference_map(before, after) >= 0.08
result = {
    "scope": "constructed noiseless square only; not a real benchmark",
    "pair_presence": binary_pair_metrics([True], [bool(pred.any())]),
    "localization": mask_metrics(truth, pred),
    "regions_xyxy_half_open": connected_regions(pred),
    "method": "raw RGB mean absolute difference, threshold=0.08",
}
out = ROOT / "results/metrics/synthetic_smoke.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
vis = ROOT / "results/visualizations/synthetic_baseline_overlay.png"
vis.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(before).save(ROOT / "results/visualizations/synthetic_before.png")
Image.fromarray(after).save(ROOT / "results/visualizations/synthetic_after.png")
overlay_image(after, pred).save(vis)
Image.fromarray(truth.astype(np.uint8) * 255).save(ROOT / "results/visualizations/synthetic_ground_truth_mask.png")
print(json.dumps(result, indent=2))
