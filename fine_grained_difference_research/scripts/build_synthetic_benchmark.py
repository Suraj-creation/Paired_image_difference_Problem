#!/usr/bin/env python3
"""Package the constructed pairs with explicit synthetic change/no-change masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "results/predictions/qwen3vl_zero_shot_synthetic.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests/qwen3vl_synthetic_diagnostic.json")
    args = parser.parse_args()
    source = json.loads(args.predictions.read_text(encoding="utf-8"))
    rows = []
    masks_dir = ROOT / "data/synthetic_ground_truth_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    semantic_changes = {"small_addition", "small_removal_reversed", "color_change", "movement",
                        "movement_reversed", "text_edit"}
    for case in source["cases"]:
        with Image.open(case["before"]) as image:
            before = np.asarray(image.convert("RGB"))
        with Image.open(case["after"]) as image:
            after = np.asarray(image.convert("RGB"))
        if before.shape != after.shape:
            raise ValueError(f"synthetic case has mismatched dimensions: {case['case_id']}")
        # For controlled renderings, exact pixel inequality is the localization
        # annotation. The illumination-only pair is intentionally semantic no-change.
        mask = np.any(before != after, axis=2) if case["case_id"] in semantic_changes else np.zeros(before.shape[:2], bool)
        mask_path = masks_dir / f"{case['case_id']}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        rows.append({
            "pair_id": case["case_id"],
            "before": os.path.relpath(Path(case["before"]).resolve(), args.output.parent.resolve()),
            "after": os.path.relpath(Path(case["after"]).resolve(), args.output.parent.resolve()),
            "split": "synthetic_diagnostic",
            "group_id": case["case_id"],
            "change_present": case["case_id"] in semantic_changes,
            "truth_mask": os.path.relpath(mask_path.resolve(), args.output.parent.resolve()),
            "change_records": [{"description": case["expected"]}],
            "annotation_source": "constructed pair; exact rendered pixel difference; illumination-only labeled semantic no-change",
        })
    # Add hard cases that distinguish a real local edit under exposure change,
    # multiple simultaneous edits, and a global camera translation nuisance.
    hard_dir = ROOT / "results/visualizations/synthetic_hard_cases"
    hard_dir.mkdir(parents=True, exist_ok=True)
    size = 448

    def base_scene() -> Image.Image:
        image = Image.new("RGB", (size, size), (235, 233, 224))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 350, size, size), fill=(192, 180, 158))
        draw.line((0, 350, size, 350), fill=(95, 88, 78), width=3)
        return image

    # Two changes in one pair: a rectangle changes color and a ball is added.
    before = base_scene()
    ImageDraw.Draw(before).rectangle((95, 192, 245, 286), fill=(42, 102, 174))
    after = base_scene()
    draw = ImageDraw.Draw(after)
    draw.rectangle((95, 192, 245, 286), fill=(215, 102, 45))
    draw.ellipse((303, 223, 351, 271), fill=(222, 178, 40))
    hard_cases = {"multiple_changes": (before, after, True, None,
                    "The rectangle changes from blue to orange and a small yellow circle is added.")}

    # Genuine small addition on top of a global brightness nuisance.
    before = base_scene()
    ImageDraw.Draw(before).rectangle((95, 192, 245, 286), fill=(42, 102, 174))
    after = ImageEnhance.Brightness(before).enhance(1.22)
    ImageDraw.Draw(after).ellipse((303, 223, 351, 271), fill=(222, 178, 40))
    textmask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(textmask).ellipse((303, 223, 351, 271), fill=255)
    hard_cases["addition_plus_illumination"] = (before, after, True, textmask,
        "A small yellow circle is added on the right while the whole image becomes brighter.")

    # Global camera translation: all depicted content is stationary in the scene.
    before = base_scene()
    ImageDraw.Draw(before).rectangle((95, 192, 245, 286), fill=(42, 102, 174))
    after = Image.new("RGB", (size, size), (235, 233, 224))
    after.paste(before, (8, 4))
    hard_cases["camera_translation_nuisance"] = (before, after, False,
        Image.new("L", (size, size), 0), "No depicted semantic change; the camera view shifts slightly.")

    for case_id, (before, after, changed, explicit_mask, expected) in hard_cases.items():
        before_path, after_path = hard_dir / f"{case_id}_before.png", hard_dir / f"{case_id}_after.png"
        before.save(before_path)
        after.save(after_path)
        if explicit_mask is None:
            before_array = np.asarray(before.convert("RGB"))
            after_array = np.asarray(after.convert("RGB"))
            mask = np.any(before_array != after_array, axis=2)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        else:
            mask_image = explicit_mask
        mask_path = masks_dir / f"{case_id}.png"
        mask_image.save(mask_path)
        rows.append({
            "pair_id": case_id,
            "before": os.path.relpath(before_path.resolve(), args.output.parent.resolve()),
            "after": os.path.relpath(after_path.resolve(), args.output.parent.resolve()),
            "split": "synthetic_diagnostic",
            "group_id": case_id,
            "change_present": changed,
            "truth_mask": os.path.relpath(mask_path.resolve(), args.output.parent.resolve()),
            "change_records": [{"description": expected}],
            "annotation_source": "constructed controlled pair; hand-authored semantic label and pixel mask; nuisance-only regions intentionally excluded",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} constructed pairs to {args.output}")


if __name__ == "__main__":
    main()

