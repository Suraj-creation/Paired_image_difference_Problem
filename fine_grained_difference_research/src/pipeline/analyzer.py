"""End-to-end evidence-preserving paired-image analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.localization.photometric import adaptive_threshold_map, compensate_global_photometric_change
from src.localization.pixel_baseline import (
    connected_regions,
    difference_map,
    filter_small_components,
    load_rgb,
    overlay_image,
)
from src.localization.registration import estimate_global_translation


def analyze_pair(before_path: str | Path, after_path: str | Path, output_dir: str | Path,
                 *, raw_threshold: float = 0.08, min_area: int = 4,
                 captioner: Any | None = None, ocr_reader: Any | None = None,
                 pair_id: str | None = None) -> dict[str, Any]:
    before_path, after_path, output_dir = map(Path, (before_path, after_path, output_dir))
    before, after = load_rgb(before_path), load_rgb(after_path)
    if before.shape != after.shape:
        raise ValueError(
            f"Image dimensions differ (before={before.shape[1]}x{before.shape[0]}, "
            f"after={after.shape[1]}x{after.shape[0]}). Register or resize explicitly and document that preprocessing."
        )
    if not 0 <= raw_threshold <= 1 or min_area < 1:
        raise ValueError("raw_threshold must be in [0,1] and min_area >= 1")

    raw_score = difference_map(before, after)
    raw_mask = filter_small_components(raw_score >= raw_threshold, min_area)
    compensated_score, transform = compensate_global_photometric_change(before, after)
    compensated_binary, compensated_threshold = adaptive_threshold_map(compensated_score)
    compensated_mask = filter_small_components(compensated_binary, min_area)
    translation = estimate_global_translation(before, after)
    raw_regions = connected_regions(raw_mask)
    compensated_regions = connected_regions(compensated_mask)
    raw_fraction = float(raw_mask.mean())
    compensated_fraction = float(compensated_mask.mean())
    nuisance_suspected = raw_fraction >= 0.25 and compensated_fraction <= max(0.03, raw_fraction * 0.15)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "raw_mask": "mask_raw.png",
        "photometric_residual_mask": "mask_photometric_residual.png",
        "raw_overlay": "overlay_raw.png",
        "photometric_residual_overlay": "overlay_photometric_residual.png",
        "raw_difference": "difference_raw.png",
        "photometric_residual": "difference_photometric_residual.png",
        "pair_preview": "pair_preview.png",
    }
    _save_mask(output_dir / artifacts["raw_mask"], raw_mask)
    _save_mask(output_dir / artifacts["photometric_residual_mask"], compensated_mask)
    overlay_image(after, raw_mask).save(output_dir / artifacts["raw_overlay"])
    overlay_image(after, compensated_mask).save(output_dir / artifacts["photometric_residual_overlay"])
    _save_heatmap(output_dir / artifacts["raw_difference"], raw_score)
    _save_heatmap(output_dir / artifacts["photometric_residual"], compensated_score)
    _save_pair_preview(output_dir / artifacts["pair_preview"], before, after)

    semantic = None
    if captioner is not None:
        semantic = captioner.describe(before_path, after_path)
        _attach_regions(semantic, raw_regions, compensated_regions)
    ocr = ocr_reader.compare(before_path, after_path) if ocr_reader is not None else None

    parsed = semantic.get("parsed") if semantic else None
    model_decision = parsed.get("status") if parsed else None
    review_reasons = ["language claims and candidate regions have not been validated by a target-domain human study"]
    if translation.get("candidate") and model_decision == "changed":
        system_decision = "uncertain"
        review_reasons.append("global translation may explain some differences; no automatic registration was applied")
    elif nuisance_suspected and model_decision == "changed" and not compensated_mask.any():
        system_decision = "uncertain"
        review_reasons.append("language model reports change but no localized residual remains after broad photometric compensation")
    elif nuisance_suspected and model_decision == "changed":
        system_decision = "changed"
        review_reasons.append("global photometric variation is present alongside localized residual change candidates")
    elif model_decision in {"changed", "no_meaningful_change"}:
        system_decision = model_decision
    else:
        system_decision = "uncertain"
        review_reasons.append("language-model output is unavailable, incomplete, or uncertain")
    if ocr and ocr.get("text_change_candidate"):
        review_reasons.append("OCR strings differ; inspect the text crop because exact OCR transcription may be wrong")
    if not ocr and semantic is not None:
        review_reasons.append("OCR cross-check is disabled")

    height, width = before.shape[:2]
    result = {
        "schema_version": "1.0",
        "pair_id": pair_id or before_path.stem,
        "inputs": {"before": str(before_path.resolve()), "after": str(after_path.resolve())},
        "dimensions": {"width": int(width), "height": int(height)},
        "analysis": {
            "global_translation_estimate": translation,
            "raw_pixel": {
                "method": "mean_absolute_rgb_difference",
                "threshold": raw_threshold,
                "min_component_area": min_area,
                "changed_pixel_fraction": raw_fraction,
                "candidate_change_present": bool(raw_mask.any()),
                "regions_xyxy_half_open": raw_regions,
            },
            "photometric_residual": {
                "method": "robust_global_per_channel_affine_compensation",
                "threshold": compensated_threshold,
                "min_component_area": min_area,
                "changed_pixel_fraction": compensated_fraction,
                "candidate_change_present": bool(compensated_mask.any()),
                "regions_xyxy_half_open": compensated_regions,
                "estimated_transform": transform,
                "global_nuisance_suspected": nuisance_suspected,
            },
        },
        "semantic_interpretation": semantic,
        "ocr_evidence": ocr,
        "pair_decision": {
            "status": system_decision,
            "model_status": model_decision,
            "global_nuisance_suspected": nuisance_suspected,
            "review_required": True,
            "review_reasons": review_reasons,
            "policy": "escalate global-translation ambiguity, and escalate broad photometric change without localized residual support; preserve localized residual candidates",
        },
        "grounding": {
            "association_status": "unverified: pixel regions are candidate evidence and are not proven to support each generated claim",
            "review_required": True,
        },
        "artifacts": artifacts,
        "limitations": [
            "Pixel differences are not semantic changes.",
            "Photometric compensation can suppress genuine global color or illumination changes; raw evidence is retained.",
            "No automatic image registration was performed; geometric alignment must be explicit.",
            "Language-model confidence, if present, is not calibrated.",
            "This output is a research aid and requires human review until validated on labeled target-domain data.",
        ],
    }
    out_json = output_dir / "analysis.json"
    result["output_json"] = str(out_json.resolve())
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def _attach_regions(semantic: dict[str, Any], raw_regions: list[dict], residual_regions: list[dict]) -> None:
    parsed = semantic.get("parsed")
    if parsed is not None:
        parsed["candidate_regions_xyxy_half_open"] = {
            "raw_pixel": raw_regions,
            "photometric_residual": residual_regions,
        }
        parsed["region_claim_association_verified"] = False


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _save_heatmap(path: Path, score: np.ndarray) -> None:
    scaled = np.clip(score / max(float(np.quantile(score, 0.995)), 1e-6), 0, 1)
    Image.fromarray(np.uint8(np.round(scaled * 255)), mode="L").save(path)


def _save_pair_preview(path: Path, before: np.ndarray, after: np.ndarray) -> None:
    h, w = before.shape[:2]
    canvas = Image.new("RGB", (2 * w + 16, h + 32), "white")
    canvas.paste(Image.fromarray(before), (0, 32))
    canvas.paste(Image.fromarray(after), (w + 16, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 8), "BEFORE", fill="black")
    draw.text((w + 20, 8), "AFTER", fill="black")
    canvas.save(path)

