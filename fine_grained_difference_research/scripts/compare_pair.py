#!/usr/bin/env python3
"""Run a transparent pixel-difference baseline on one ordered image pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.localization.pixel_baseline import connected_regions, difference_map, filter_small_components, load_rgb, overlay_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--threshold", type=float, default=0.08, help="mean absolute RGB difference threshold [0,1]")
    parser.add_argument("--min-area", type=int, default=1, help="discard smaller 8-connected components")
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    parser.add_argument("--mask", type=Path, help="optional 8-bit predicted mask image")
    parser.add_argument("--overlay", type=Path, help="optional red overlay on the after image")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.min_area < 1:
        parser.error("threshold must be in [0,1] and min-area >= 1")

    before, after = load_rgb(args.before), load_rgb(args.after)
    diff = difference_map(before, after)
    mask = filter_small_components(diff >= args.threshold, args.min_area)
    regions = connected_regions(mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": "raw_rgb_mean_absolute_difference", "before": str(args.before),
               "after": str(args.after), "width": int(before.shape[1]), "height": int(before.shape[0]),
               "threshold": args.threshold, "min_area": args.min_area,
               "candidate_change_present": bool(mask.any()),
               "changed_pixel_fraction": float(mask.mean()), "mean_difference": float(diff.mean()),
               "regions_xyxy_half_open": regions,
               "interpretation_limit": "Pixel variation only; no semantic change or nuisance inference."}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.mask:
        args.mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255).save(args.mask)
    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay_image(after, mask).save(args.overlay)
    print(json.dumps({k: payload[k] for k in ("candidate_change_present", "changed_pixel_fraction", "regions_xyxy_half_open")}))


if __name__ == "__main__":
    main()
