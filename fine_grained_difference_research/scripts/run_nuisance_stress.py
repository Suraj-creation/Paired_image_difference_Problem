#!/usr/bin/env python3
"""Demonstrate how raw RGB subtraction responds to global illumination."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import mask_metrics
from src.localization.pixel_baseline import difference_map, overlay_image

before = np.full((64, 64, 3), 100, dtype=np.uint8)
after_light_only = np.full_like(before, 130)
after_change_and_light = after_light_only.copy()
after_change_and_light[20:24, 30:35] = (250, 60, 60)
truth_none = np.zeros((64, 64), dtype=bool)
truth_change = truth_none.copy()
truth_change[20:24, 30:35] = True
threshold = 0.08

def evaluate(after: np.ndarray, truth: np.ndarray, tag: str) -> dict:
    prediction = difference_map(before, after) >= threshold
    vis = ROOT / f"results/visualizations/{tag}_overlay.png"
    overlay_image(after, prediction).save(vis)
    return {"case": tag, "candidate_change_present": bool(prediction.any()),
            "predicted_pixel_fraction": float(prediction.mean()),
            "localization": mask_metrics(truth, prediction)}

result = {
    "scope": "constructed 64x64 images; diagnostic only, not a benchmark",
    "baseline": "raw RGB mean absolute difference, threshold=0.08",
    "illumination_only": evaluate(after_light_only, truth_none, "stress_illumination_only"),
    "local_change_plus_illumination": evaluate(after_change_and_light, truth_change, "stress_change_plus_illumination"),
}
out = ROOT / "results/failure_analysis/pixel_baseline_illumination_stress.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
