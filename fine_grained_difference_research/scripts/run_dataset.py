#!/usr/bin/env python3
"""Run the pair pipeline for every row in a research manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.qwen_captioner import QwenPairCaptioner
from src.models.easyocr_adapter import EasyOCRPairReader
from src.pipeline.analyzer import analyze_pair
from src.utilities.manifest import audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON list; paths are relative to this file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--captioner", choices=("none", "qwen3vl"), default="none")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--ocr", choices=("none", "easyocr"), default="none",
                        help="optional OCR cross-check; EasyOCR weights must already be cached")
    parser.add_argument("--raw-threshold", type=float, default=0.08)
    parser.add_argument("--min-area", type=int, default=4)
    parser.add_argument("--mask-source", choices=("raw", "photometric_residual"), default="photometric_residual")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        parser.error("manifest must be a non-empty JSON list")
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    audit = audit_manifest(manifest)
    if audit["missing_files"]:
        parser.error("manifest contains missing image or mask files; run scripts/verify_data.py")
    captioner = QwenPairCaptioner(args.model, args.device, args.max_new_tokens) if args.captioner == "qwen3vl" else None
    ocr_reader = EasyOCRPairReader() if args.ocr == "easyocr" else None
    result_manifest = []
    for index, row in enumerate(rows, 1):
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row["pair_id"]))[:120]
        pair_out = args.output_dir / safe_id
        result = analyze_pair(manifest.parent / row["before"], manifest.parent / row["after"], pair_out,
                              raw_threshold=args.raw_threshold, min_area=args.min_area,
                              captioner=captioner, ocr_reader=ocr_reader, pair_id=str(row["pair_id"]))
        parsed = result["semantic_interpretation"].get("parsed") if result["semantic_interpretation"] else None
        semantic_status = parsed.get("status") if parsed else None
        decision_status = result["pair_decision"]["status"]
        pred_present = True if decision_status == "changed" else False if decision_status == "no_meaningful_change" else None
        result_manifest.append({
            "pair_id": row["pair_id"],
            "result_json": os.path.relpath(pair_out / "analysis.json", args.output_dir),
            "predicted_change_present": pred_present,
            "prediction_status": decision_status,
            "model_status": semantic_status or "unavailable",
            "review_required": result["pair_decision"]["review_required"],
            "raw_mask": os.path.relpath(pair_out / "mask_raw.png", args.output_dir),
            "photometric_residual_mask": os.path.relpath(pair_out / "mask_photometric_residual.png", args.output_dir),
            "predicted_mask": os.path.relpath(
                pair_out / ("mask_raw.png" if args.mask_source == "raw" else "mask_photometric_residual.png"),
                args.output_dir,
            ),
            **({key: row[key] for key in ("change_present", "split", "group_id") if key in row}),
            **({"truth_mask": os.path.relpath((manifest.parent / row["truth_mask"]).resolve(), args.output_dir.resolve())}
               if row.get("truth_mask") else {}),
        })
        print(f"[{index}/{len(rows)}] {row['pair_id']}: {semantic_status or 'pixel evidence only'}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "predictions_manifest.json"
    out.write_text(json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latencies = []
    for row in result_manifest:
        result_json = json.loads((args.output_dir / row["result_json"]).read_text(encoding="utf-8"))
        semantic = result_json.get("semantic_interpretation") or {}
        if semantic.get("latency_seconds") is not None:
            latencies.append(semantic["latency_seconds"])
    metadata = {
        "started_utc": started_utc,
        "pair_count": len(rows),
        "captioner": args.captioner,
        "model_id": args.model if args.captioner == "qwen3vl" else None,
        "ocr": args.ocr,
        "mask_source": args.mask_source,
        "raw_threshold": args.raw_threshold,
        "minimum_component_area": args.min_area,
        "mean_caption_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "total_elapsed_seconds": time.perf_counter() - started,
        "resources": captioner.resource_summary() if captioner is not None else {"device": "CPU"},
        "network_or_package_install_attempted": False,
        "data_scope": "constructed diagnostic pairs only; not a public benchmark",
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

