#!/usr/bin/env python3
"""Check pair-manifest paths, image readability, dimensions, masks, and leakage clues."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utilities.manifest import audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_manifest(args.manifest)
    result["valid"] = not (result["missing_files"] or result["invalid_rows"]
                           or result["truth_mask_dimension_mismatches"])
    result["warning_count"] = len(result["exact_duplicate_groups"]) + len(result["group_split_conflicts"])
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

