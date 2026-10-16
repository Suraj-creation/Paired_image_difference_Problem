#!/usr/bin/env python3
"""Create a visual audit sheet from executed predictions and synthetic labels."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/runs/qwen3vl_end_to_end"
IDS = ("addition_plus_illumination", "text_edit", "illumination_only",
       "camera_translation_nuisance", "multiple_changes")
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, bold: bool = False):
    path = FONT_BOLD_PATH if bold else FONT_PATH
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def main() -> None:
    manifest = json.loads((RUN / "predictions_manifest.json").read_text(encoding="utf-8"))
    review = json.loads((ROOT / "results/metrics/qwen3vl_end_to_end_manual_review.json").read_text(encoding="utf-8"))
    by_id = {row["pair_id"]: row for row in manifest}
    reviews = {row["pair_id"]: row for row in review["records"]}
    panel, gutter, header_h, row_h = 176, 18, 92, 216
    labels = ["BEFORE", "AFTER", "TRUTH MASK", "PREDICTED MASK", "PREDICTED OVERLAY"]
    width = 5 * panel + 6 * gutter + 310
    height = header_h + len(IDS) * row_h + gutter
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gutter, 12), "End-to-end diagnostic cases", font=font(27, True), fill="#111111")
    draw.text((gutter, 50), "Constructed examples only. Masks are candidate evidence; claim-to-region grounding is unverified.",
              font=font(14), fill="#444444")
    for i, label in enumerate(labels):
        x = gutter + i * (panel + gutter)
        draw.text((x, header_h - 25), label, font=font(12, True), fill="#333333")
    for index, case_id in enumerate(IDS):
        row = by_id[case_id]
        result = json.loads((RUN / row["result_json"]).read_text(encoding="utf-8"))
        review_row = reviews[case_id]
        y = header_h + index * row_h
        draw.line((gutter, y, width - gutter, y), fill="#d7dde2", width=2)
        before = Image.open(result["inputs"]["before"]).convert("RGB")
        after = Image.open(result["inputs"]["after"]).convert("RGB")
        truth = Image.open(RUN / row["truth_mask"]).convert("L").convert("RGB")
        pred = Image.open(RUN / row["predicted_mask"]).convert("L").convert("RGB")
        overlay = Image.open((RUN / row["result_json"]).parent /
                             result["artifacts"]["photometric_residual_overlay"]).convert("RGB")
        images = [before, after, truth, pred, overlay]
        for j, image in enumerate(images):
            x = gutter + j * (panel + gutter)
            image.thumbnail((panel - 4, panel - 4))
            tile = Image.new("RGB", (panel, panel), "#f4f5f6")
            tile.paste(image, ((panel - image.width) // 2, (panel - image.height) // 2))
            canvas.paste(tile, (x, y + 10))
        text_x = gutter + 5 * (panel + gutter)
        draw.text((text_x, y + 14), case_id.replace("_", " "), font=font(16, True), fill="#111111")
        decision = result["pair_decision"]
        draw.text((text_x, y + 42),
                  f"Decision: {decision['status']}  |  model: {decision['model_status']}",
                  font=font(12, True), fill="#125b70")
        parsed = (result.get("semantic_interpretation") or {}).get("parsed") or {}
        claims = parsed.get("changes", [])
        claim_lines = [f"{c.get('type')}: {c.get('entity')}" for c in claims[:3]] or ["No structured change claim"]
        note_lines = textwrap.wrap(review_row["review_note"], width=38)
        y_text = y + 66
        for line in claim_lines:
            draw.text((text_x, y_text), line[:45], font=font(11), fill="#222222")
            y_text += 18
        y_text += 5
        for line in note_lines[:6]:
            draw.text((text_x, y_text), line, font=font(10), fill="#555555")
            y_text += 16
    out = ROOT / "results/visualizations/end_to_end_failure_gallery.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()

