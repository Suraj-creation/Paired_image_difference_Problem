#!/usr/bin/env python3
"""Analyze one ordered pair and save JSON, masks, overlays, and difference maps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.qwen_captioner import QwenPairCaptioner
from src.models.easyocr_adapter import EasyOCRPairReader
from src.pipeline.analyzer import analyze_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id")
    parser.add_argument("--raw-threshold", type=float, default=0.08)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--captioner", choices=("none", "qwen3vl"), default="none")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="model ID or local path; only cached files are used")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--ocr", choices=("none", "easyocr"), default="none")
    args = parser.parse_args()
    if args.raw_threshold < 0 or args.raw_threshold > 1 or args.min_area < 1:
        parser.error("raw-threshold must be in [0,1] and min-area >= 1")
    captioner = None
    if args.captioner == "qwen3vl":
        captioner = QwenPairCaptioner(args.model, args.device, args.max_new_tokens)
    ocr_reader = EasyOCRPairReader() if args.ocr == "easyocr" else None
    result = analyze_pair(args.before, args.after, args.output_dir,
                          raw_threshold=args.raw_threshold, min_area=args.min_area,
                          captioner=captioner, ocr_reader=ocr_reader, pair_id=args.pair_id)
    print(json.dumps({
        "pair_id": result["pair_id"],
        "raw_regions": len(result["analysis"]["raw_pixel"]["regions_xyxy_half_open"]),
        "photometric_residual_regions": len(result["analysis"]["photometric_residual"]["regions_xyxy_half_open"]),
        "captioner_enabled": captioner is not None,
        "pair_decision": result["pair_decision"]["status"],
        "output_json": result["output_json"],
        "review_required": result["grounding"]["review_required"],
    }, indent=2))


if __name__ == "__main__":
    main()

