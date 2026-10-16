"""Optional local-only OCR evidence using the already-cached EasyOCR weights."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class EasyOCRPairReader:
    def __init__(self, languages: tuple[str, ...] = ("en",), gpu: bool = False,
                 model_storage_directory: str | Path | None = None):
        import easyocr

        model_dir = Path(model_storage_directory or (Path.home() / ".EasyOCR/model"))
        if not (model_dir / "craft_mlt_25k.pth").is_file():
            raise FileNotFoundError(f"cached EasyOCR detector weights not found in {model_dir}")
        if not (model_dir / "english_g2.pth").is_file():
            raise FileNotFoundError(f"cached EasyOCR recognition weights not found in {model_dir}")
        self.reader = easyocr.Reader(list(languages), gpu=gpu,
                                     model_storage_directory=str(model_dir),
                                     download_enabled=False, verbose=False)

    def compare(self, before_path: str | Path, after_path: str | Path) -> dict[str, Any]:
        before = self._read(before_path)
        after = self._read(after_path)
        before_text = " ".join(item["text"] for item in before).strip()
        after_text = " ".join(item["text"] for item in after).strip()
        normalized_before = "".join(ch for ch in before_text.upper() if ch.isalnum())
        normalized_after = "".join(ch for ch in after_text.upper() if ch.isalnum())
        both_confident = (before and after and
                          min(max(item["confidence"] for item in before),
                              max(item["confidence"] for item in after)) >= 0.3)
        discrepancy = bool(both_confident and normalized_before != normalized_after)
        return {
            "engine": "EasyOCR local cached weights",
            "before_text": before_text or None,
            "after_text": after_text or None,
            "normalized_before": normalized_before or None,
            "normalized_after": normalized_after or None,
            "text_change_candidate": discrepancy,
            "similarity": SequenceMatcher(None, normalized_before, normalized_after).ratio()
                           if normalized_before or normalized_after else None,
            "before_detections": before,
            "after_detections": after,
            "interpretation": "OCR mismatch is a review cue, not proof of a semantic text change; OCR confidence is uncalibrated.",
        }

    def _read(self, path: str | Path) -> list[dict[str, Any]]:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        results = self.reader.readtext(rgb, detail=1, paragraph=False)
        records = []
        for polygon, text, confidence in results:
            points = np.asarray(polygon, dtype=float)
            records.append({
                "text": str(text),
                "confidence": float(confidence),
                "bbox_xyxy_half_open": [
                    int(np.floor(points[:, 0].min())), int(np.floor(points[:, 1].min())),
                    int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max())),
                ],
            })
        return records

