#!/usr/bin/env python3
"""Validate output completeness and artifact integrity before evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_ARTIFACTS = (
    "raw_mask", "photometric_residual_mask", "raw_overlay", "photometric_residual_overlay",
    "raw_difference", "photometric_residual", "pair_preview",
)


def validate(result_manifest: Path) -> dict:
    rows = json.loads(result_manifest.read_text(encoding="utf-8"))
    errors, warnings = [], []
    if not isinstance(rows, list) or not rows:
        return {"valid": False, "errors": ["prediction manifest must be a non-empty list"], "warnings": []}
    seen = set()
    for row in rows:
        pid = row.get("pair_id", "<missing>")
        if pid in seen:
            errors.append(f"duplicate pair_id {pid}")
        seen.add(pid)
        if not row.get("result_json"):
            errors.append(f"{pid}: missing result_json")
            continue
        path = (result_manifest.parent / row["result_json"]).resolve()
        if not path.is_file():
            errors.append(f"{pid}: result JSON not found: {path}")
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{pid}: result JSON unreadable: {exc}")
            continue
        if result.get("schema_version") != "1.0":
            errors.append(f"{pid}: unsupported or missing schema_version")
        if result.get("grounding", {}).get("review_required") is not True:
            errors.append(f"{pid}: review_required must remain true until calibration")
        if result.get("semantic_interpretation") is None:
            warnings.append(f"{pid}: no semantic model output; only pixel evidence is available")
        artifacts = result.get("artifacts", {})
        for key in REQUIRED_ARTIFACTS:
            artifact = artifacts.get(key)
            if not artifact or not (path.parent / artifact).is_file():
                errors.append(f"{pid}: missing artifact {key}")
        if row.get("predicted_change_present") is None:
            warnings.append(f"{pid}: pair-level decision is abstained or unavailable")
    return {"valid": not errors, "pairs": len(rows), "errors": errors, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.prediction_manifest)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

