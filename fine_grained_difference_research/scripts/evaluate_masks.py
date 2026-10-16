#!/usr/bin/env python3
"""Evaluate pair-presence and pixel-mask predictions listed in a JSON manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import binary_pair_metrics, mask_metrics, selective_pair_metrics


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON list with truth/prediction fields")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not rows:
        parser.error("manifest must contain at least one row")
    truth, prediction, per_pair = [], [], []
    for row in rows:
        if "change_present" not in row or not isinstance(row["change_present"], bool):
            raise ValueError(f"row {row.get('pair_id')} is missing boolean change_present ground truth")
        truth.append(row["change_present"])
        predicted = row.get("predicted_change_present")
        if predicted is not None and not isinstance(predicted, bool):
            raise ValueError(f"row {row.get('pair_id')} predicted_change_present must be boolean or null")
        prediction.append(predicted)
        if row.get("truth_mask") and row.get("predicted_mask"):
            tm = read_mask(args.manifest.parent / row["truth_mask"])
            pm = read_mask(args.manifest.parent / row["predicted_mask"])
            per_pair.append({"pair_id": row["pair_id"], **mask_metrics(tm, pm)})
    covered = [(t, p) for t, p in zip(truth, prediction) if p is not None]
    result = {"pair_presence": (binary_pair_metrics([t for t, _ in covered], [p for _, p in covered])
                                  if len(covered) == len(truth) else selective_pair_metrics(truth, prediction)),
              "localization_per_pair": per_pair,
              "localization_macro": ({key: float(np.mean([r[key] for r in per_pair]))
                                      for key in ("precision", "recall", "iou", "dice")} if per_pair else None),
              "localization_micro": ({
                  "precision": sum(r["tp"] for r in per_pair) / max(sum(r["tp"] + r["fp"] for r in per_pair), 1),
                  "recall": sum(r["tp"] for r in per_pair) / max(sum(r["tp"] + r["fn"] for r in per_pair), 1),
                  "iou": sum(r["tp"] for r in per_pair) / max(sum(r["tp"] + r["fp"] + r["fn"] for r in per_pair), 1),
              } if per_pair else None),
              "n_pairs": len(rows), "n_pairs_with_masks": len(per_pair)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
