"""Reproducible data access and evaluation for the DiffSpot web-interface benchmark.

DiffSpot supplies paired rendered screenshots, a changed/no-change label, and a target
element box. It does not supply pixel segmentation masks. This module therefore reports
pair classification and element-box grounding metrics, never pixel-mask F1/IoU.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


DATASET_ID = "tencent/DiffSpot"
THRESHOLDS = (1, 2, 4, 8, 12, 16, 24, 32)


def fetch_diffspot(cache_dir: str | Path, *, progress: bool = True) -> pd.DataFrame:
    """Retrieve the complete public test split and cache only its image pairs locally.

    Original image bytes are streamed from the public Hugging Face dataset and are not
    committed. A stable row manifest and retrieval provenance are written next to them.
    """
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.csv"
    cached = None
    if manifest_path.exists():
        candidate = pd.read_csv(manifest_path)
        if len(candidate) and all((root / p).is_file() for p in candidate.before_path) and all(
                (root / p).is_file() for p in candidate.after_path):
            cached = candidate
    if cached is not None:
        return cached

    try:
        from datasets import Image, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the repository's datasets dependency to fetch DiffSpot") from exc
    stream = load_dataset(DATASET_ID, split="test", streaming=True)
    stream = stream.cast_column("image_before", Image(decode=False))
    stream = stream.cast_column("image_after", Image(decode=False))

    records = []
    for index, row in enumerate(stream):
        before_rel, after_rel = f"pair_{index:05d}_before.png", f"pair_{index:05d}_after.png"
        for image_key, relative in (("image_before", before_rel), ("image_after", after_rel)):
            payload = row[image_key].get("bytes")
            if not payload:
                raise RuntimeError(f"DiffSpot row {index} has no original image bytes for {image_key}")
            path = root / relative
            if not path.exists() or path.stat().st_size == 0:
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape[:2] != (800, 1280):
                    raise RuntimeError(f"invalid image at DiffSpot row {index} {image_key}")
                path.write_bytes(payload)
        mutation = row.get("mutation_types") or []
        records.append({
            "row_idx": index, "id": row["id"], "task_type": row["task_type"],
            "changed": bool(row["task_type"] == "visual_diff"),
            "difficulty": row.get("difficulty", "unknown"), "domain": row.get("domain", "unknown"),
            "mutation_types": ";".join(mutation), "pixel_diff": float(row.get("pixel_diff", 0) or 0),
            "target_bbox_x": int(row.get("target_bbox_x", -1)),
            "target_bbox_y": int(row.get("target_bbox_y", -1)),
            "target_bbox_w": int(row.get("target_bbox_w", -1)),
            "target_bbox_h": int(row.get("target_bbox_h", -1)),
            "ground_truth_diff": row.get("ground_truth_diff", ""),
            "before_path": before_rel, "after_path": after_rel,
        })
        if progress and (index + 1) % 100 == 0:
            print(f"Downloaded and verified original image pairs: {index + 1}", flush=True)
        if (index + 1) % 100 == 0:
            pd.DataFrame(records).to_csv(manifest_path, index=False)

    frame = pd.DataFrame(records).sort_values("row_idx").reset_index(drop=True)
    frame.to_csv(manifest_path, index=False)
    provenance = {
        "dataset": DATASET_ID, "config": "default", "split": "test", "rows": len(frame),
        "license_reported_by_dataset_card": "MIT",
        "source": "https://huggingface.co/datasets/tencent/DiffSpot",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "access": "Hugging Face datasets streaming interface, original image bytes (decode=False)",
        "note": "Downloaded images are cached locally and excluded from Git. The source dataset card and paper govern use and attribution.",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return frame


def deterministic_split(frame: pd.DataFrame, calibration_fraction: float = 0.2) -> pd.DataFrame:
    """Create a deterministic, label-stratified calibration/holdout split by pair ID."""
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between 0 and 0.5")
    out = frame.copy()
    keys = ["task_type", "difficulty"]
    out["_hash"] = out.id.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    out["split"] = "holdout"
    for _, indexes in out.groupby(keys, dropna=False).groups.items():
        indexes = sorted(indexes, key=lambda i: out.at[i, "_hash"])
        n = max(1, int(round(len(indexes) * calibration_fraction))) if len(indexes) > 1 else 0
        out.loc[indexes[:n], "split"] = "calibration"
    return out.drop(columns="_hash")


def _rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def residual_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel maximum absolute RGB difference; no warp or photometric normalization."""
    if a.shape != b.shape:
        raise ValueError(f"viewport mismatch: {a.shape} vs {b.shape}; refusing to warp away evidence")
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2).astype(np.uint8)


def threshold_mask(delta: np.ndarray, threshold: int) -> np.ndarray:
    return delta >= int(threshold)


def otsu_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("viewport mismatch")
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.int16)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.int16)
    delta = np.abs(gray_a - gray_b).astype(np.uint8)
    if delta.max() == 0:
        return np.zeros(delta.shape, bool)
    _, mask = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask > 0


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    yy, xx = np.where(mask)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def residual_grid_coverage(mask: np.ndarray, grid: int = 8) -> float:
    """Fraction of an 8x8 image grid touched by residual pixels."""
    h, w = mask.shape
    active = 0
    for gy in range(grid):
        y1, y2 = round(gy * h / grid), round((gy + 1) * h / grid)
        for gx in range(grid):
            x1, x2 = round(gx * w / grid), round((gx + 1) * w / grid)
            active += bool(mask[y1:y2, x1:x2].any())
    return active / (grid * grid)


def _global_photometric_explained(a: np.ndarray, b: np.ndarray) -> float:
    """Residual ratio after a simple per-channel affine exposure fit (diagnostic only)."""
    aa, bb = a[::4, ::4].astype(np.float32), b[::4, ::4].astype(np.float32)
    fitted = b.astype(np.float32).copy()
    for channel in range(3):
        x, y = bb[..., channel], aa[..., channel]
        slope = float(np.std(y) / max(float(np.std(x)), 1e-5))
        offset = float(np.median(y) - slope * np.median(x))
        fitted[..., channel] = fitted[..., channel] * slope + offset
    raw = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    corrected = float(np.abs(a.astype(np.float32) - fitted).mean())
    return corrected / max(raw, 1e-5)


def review_policy(mask: np.ndarray, a: np.ndarray | None = None, b: np.ndarray | None = None,
                  *, max_grid_coverage: float = 0.05) -> tuple[str, str | None]:
    """Abstain from automatic claims when residuals span much of the viewport.

    A diffuse map can indicate a global appearance edit, exposure change, compression,
    or image displacement. Pixel evidence alone cannot distinguish them reliably, so
    return needs_review and retain the residual map for inspection.
    """
    if not mask.any():
        return "no_change", None
    reasons = []
    if residual_grid_coverage(mask) > max_grid_coverage:
        reasons.append("residual_spans_many_viewport_regions")
    if a is not None and b is not None and float(mask.mean()) > 0.005:
        if _global_photometric_explained(a, b) < 0.10:
            reasons.append("global_photometric_transform_explains_residual")
        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b)
        if response > 0.1 and float(np.hypot(dx, dy)) > 1.0:
            reasons.append("global_translation_suspected")
    if reasons:
        return "needs_review", ";".join(reasons)
    return "localized_change", None


def _nuisance_variants(a: np.ndarray) -> dict[str, np.ndarray]:
    h, w = a.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    shadow_gain = 1 - 0.30 * np.exp(-(((xx - w * 0.5) / (w * 0.24)) ** 2 +
                                       ((yy - h * 0.5) / (h * 0.35)) ** 2))
    jpeg_bytes = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 70])[1]
    jpeg = cv2.cvtColor(cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return {
        "global_brightness": np.clip(a.astype(np.float32) * 1.15 + 18, 0, 255).astype(np.uint8),
        "local_shadow": np.clip(a.astype(np.float32) * shadow_gain[..., None], 0, 255).astype(np.uint8),
        "two_pixel_translation": cv2.warpAffine(a, np.float32([[1, 0, 2], [0, 1, 1]]), (w, h),
                                                 borderMode=cv2.BORDER_REPLICATE),
        "jpeg_recompression_q70": jpeg,
    }


def calibrate_review_gate(frame: pd.DataFrame, threshold: int,
                          candidates=(0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25),
                          seed: int = 2026, progress: bool = True) -> tuple[float, pd.DataFrame]:
    """Choose the broad-residual review threshold using calibration pairs only.

    The selection maximizes automatic positive coverage subject to zero automatic false
    changes on synthetic nuisance transforms of calibration no-change screenshots.
    """
    calibration = frame[frame.split == "calibration"]
    positives = calibration[calibration.changed]
    controls = calibration[~calibration.changed].sample(
        frac=1, random_state=seed) if not calibration[~calibration.changed].empty else calibration[~calibration.changed]
    cue_rows = []
    for n, row in enumerate(positives.itertuples(index=False), 1):
        a, b = _rgb(frame.attrs["cache_root"] / row.before_path), _rgb(frame.attrs["cache_root"] / row.after_path)
        mask = threshold_mask(residual_map(a, b), threshold)
        _, reason = review_policy(mask, a, b, max_grid_coverage=1.01)
        cue_rows.append((residual_grid_coverage(mask), bool(reason)))
        if progress and n % 200 == 0:
            print(f"Review-gate calibration on labeled changes: {n}/{len(positives)}", flush=True)
    nuisance_rows = []
    for row in controls.itertuples(index=False):
        a = _rgb(frame.attrs["cache_root"] / row.before_path)
        for nuisance, b in _nuisance_variants(a).items():
            mask = threshold_mask(residual_map(a, b), threshold)
            _, reason = review_policy(mask, a, b, max_grid_coverage=1.01)
            nuisance_rows.append((nuisance, residual_grid_coverage(mask), bool(reason)))
    results = []
    for candidate in candidates:
        auto_recall = sum((grid <= candidate) and not cue for grid, cue in cue_rows) / max(len(cue_rows), 1)
        auto_fp = sum((grid <= candidate) and not cue for _, grid, cue in nuisance_rows) / max(len(nuisance_rows), 1)
        results.append({"max_grid_coverage": float(candidate),
                        "calibration_automatic_change_recall": auto_recall,
                        "calibration_nuisance_automatic_false_change_rate": auto_fp,
                        "calibration_nuisance_review_rate": 1 - auto_fp})
    table = pd.DataFrame(results)
    safe = table[table.calibration_nuisance_automatic_false_change_rate == 0]
    chosen = float(safe.max_grid_coverage.max()) if not safe.empty else float(table.max_grid_coverage.min())
    table["selected"] = table.max_grid_coverage == chosen
    return chosen, table


def bbox_iou(a: list[int] | None, b: list[int] | None) -> float | None:
    if a is None or b is None:
        return None
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0


def target_box(row: dict | pd.Series) -> list[int] | None:
    x, y, w, h = (int(row[k]) for k in ("target_bbox_x", "target_bbox_y", "target_bbox_w", "target_bbox_h"))
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _pair_stats(records: list[dict]) -> dict:
    ok = [r for r in records if r["error"] is None]
    tp = sum(r["gt_changed"] and r["pred_changed"] for r in ok)
    fp = sum((not r["gt_changed"]) and r["pred_changed"] for r in ok)
    fn = sum(r["gt_changed"] and (not r["pred_changed"]) for r in ok)
    tn = sum((not r["gt_changed"]) and (not r["pred_changed"]) for r in ok)
    div = lambda n, d: n / d if d else None
    loc = [r["target_bbox_iou"] for r in ok if r["gt_changed"] and r["target_bbox_iou"] is not None]
    inside = [r["residual_inside_target_fraction"] for r in ok if r["gt_changed"] and r["residual_inside_target_fraction"] is not None]
    pair_den = tp + fp
    prec, rec = div(tp, pair_den), div(tp, tp + fn)
    accepted = [r for r in ok if r["decision"] == "localized_change"]
    atp = sum(r["gt_changed"] for r in accepted)
    afp = sum(not r["gt_changed"] for r in accepted)
    afn = sum(r["gt_changed"] for r in ok if r["decision"] != "localized_change")
    arec = div(atp, atp + afn)
    aprec = div(atp, atp + afp)
    return {
        "pairs": len(ok), "errors": len(records) - len(ok), "pair_tp": tp, "pair_fp": fp,
        "pair_fn": fn, "pair_tn": tn, "pair_precision": prec, "pair_recall": rec,
        "pair_f1": div(2 * tp, 2 * tp + fp + fn), "pair_accuracy": div(tp + tn, len(ok)),
        "no_change_false_positive_rate": div(fp, fp + tn),
        "automatic_change_precision": aprec, "automatic_change_recall": arec,
        "automatic_change_f1": div(2 * atp, 2 * atp + afp + afn),
        "automatic_false_positive_rate_no_change": div(afp, sum(not r["gt_changed"] for r in ok)),
        "review_rate": div(sum(r["decision"] == "needs_review" for r in ok), len(ok)),
        "automatic_decision_coverage": div(sum(r["decision"] != "needs_review" for r in ok), len(ok)),
        "mean_target_box_iou": float(np.mean(loc)) if loc else None,
        "median_target_box_iou": float(np.median(loc)) if loc else None,
        "target_box_center_hit_rate": div(sum(bool(r["target_box_center_hit"]) for r in ok
                                                if r["target_box_center_hit"] is not None),
                                           sum(r["target_box_center_hit"] is not None for r in ok)),
        "residual_inside_target_fraction": float(np.mean(inside)) if inside else None,
    }


def _record(row: pd.Series, method: str, mask: np.ndarray, gt_changed: bool, pred_changed: bool,
            threshold: int | None, min_pixels: int, a: np.ndarray | None = None,
            b: np.ndarray | None = None, max_grid_coverage: float = 0.05) -> dict:
    predicted = bbox_from_mask(mask) if pred_changed else None
    decision, review_reason = (review_policy(mask, a, b, max_grid_coverage=max_grid_coverage)
                               if pred_changed else ("no_change", None))
    truth = target_box(row) if gt_changed else None
    iou = bbox_iou(predicted, truth)
    count = int(mask.sum())
    center_hit = None
    inside = None
    if truth is not None and predicted is not None:
        cx, cy = (predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2
        center_hit = bool(truth[0] <= cx <= truth[2] and truth[1] <= cy <= truth[3])
        y1, y2 = max(0, truth[1]), min(mask.shape[0], truth[3])
        x1, x2 = max(0, truth[0]), min(mask.shape[1], truth[2])
        inside = float(mask[y1:y2, x1:x2].sum() / max(count, 1))
    return {
        "id": row.id, "row_idx": int(row.row_idx), "split": row.split,
        "task_type": row.task_type, "difficulty": row.difficulty, "domain": row.domain,
        "mutation_types": row.mutation_types, "method": method,
        "gt_changed": bool(gt_changed), "pred_changed": bool(pred_changed),
        "decision": decision, "review_reason": review_reason,
        "residual_grid_coverage": residual_grid_coverage(mask),
        "pred_changed_pixels": count, "threshold": threshold, "min_changed_pixels": int(min_pixels),
        "pred_bbox_x1": predicted[0] if predicted else None,
        "pred_bbox_y1": predicted[1] if predicted else None,
        "pred_bbox_x2": predicted[2] if predicted else None,
        "pred_bbox_y2": predicted[3] if predicted else None,
        "target_bbox_iou": iou, "target_box_center_hit": center_hit,
        "residual_inside_target_fraction": inside, "error": None,
    }


def select_threshold(frame: pd.DataFrame, thresholds: Iterable[int] = THRESHOLDS,
                     min_pixels: int = 1, progress: bool = True) -> tuple[int, pd.DataFrame]:
    """Choose a pixel threshold on calibration rows only; ties prefer the stricter threshold."""
    rows = frame[frame.split == "calibration"]
    if rows.empty:
        raise ValueError("no calibration rows")
    counts = {int(t): [] for t in thresholds}
    labels = []
    for n, row in enumerate(rows.itertuples(index=False), 1):
        a, b = _rgb(frame.attrs["cache_root"] / row.before_path), _rgb(frame.attrs["cache_root"] / row.after_path)
        delta = residual_map(a, b)
        labels.append(bool(row.changed))
        for threshold in counts:
            counts[threshold].append(int((delta >= threshold).sum()))
        if progress and n % 100 == 0:
            print(f"Threshold calibration: {n}/{len(rows)}", flush=True)
    results = []
    y = np.asarray(labels, bool)
    for threshold, pix_counts in counts.items():
        p = np.asarray(pix_counts) >= min_pixels
        tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        results.append({"threshold": threshold, "min_changed_pixels": min_pixels,
                        "pair_f1": f1, "pair_tp": tp, "pair_fp": fp, "pair_fn": fn,
                        "pair_tn": int((~p & ~y).sum())})
    scores = pd.DataFrame(results).sort_values(["pair_f1", "threshold"], ascending=[False, False])
    return int(scores.iloc[0].threshold), scores


def evaluate_methods(frame: pd.DataFrame, calibrated_threshold: int, *,
                     fixed_threshold: int = 8, min_changed_pixels: int = 1,
                     review_grid_threshold: float = 0.05,
                     progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Otsu, fixed RGB residual, and calibration-selected RGB residual."""
    rows = frame[frame.split == "holdout"]
    records = []
    methods = [("unregistered_grayscale_otsu", None),
               ("fixed_rgb_residual", int(fixed_threshold)),
               ("calibrated_rgb_residual", int(calibrated_threshold))]
    for n, row in enumerate(rows.itertuples(index=False), 1):
        a, b = _rgb(frame.attrs["cache_root"] / row.before_path), _rgb(frame.attrs["cache_root"] / row.after_path)
        delta = residual_map(a, b)
        for name, threshold in methods:
            mask = otsu_mask(a, b) if threshold is None else threshold_mask(delta, threshold)
            count = int(mask.sum())
            records.append(_record(pd.Series(row._asdict()), name, mask, bool(row.changed),
                                   count >= min_changed_pixels, threshold, min_changed_pixels, a, b,
                                   review_grid_threshold))
        if progress and n % 100 == 0:
            print(f"Holdout evaluation: {n}/{len(rows)} pairs", flush=True)
    per_pair = pd.DataFrame(records)
    summaries = []
    for method, group in per_pair.groupby("method", sort=False):
        summaries.append({"method": method, **_pair_stats(group.to_dict("records"))})
    return per_pair, pd.DataFrame(summaries)


def stratified_summary(per_pair: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (method, difficulty), group in per_pair.groupby(["method", "difficulty"], sort=False):
        out.append({"method": method, "difficulty": difficulty, **_pair_stats(group.to_dict("records"))})
    return pd.DataFrame(out)


def nuisance_stress(frame: pd.DataFrame, threshold: int, *, n_controls: int = 100,
                    split: str = "holdout", review_grid_threshold: float = 0.05,
                    seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stress the review gate on no-change screenshots with controlled nuisance transforms.

    These perturbations are derived from real benchmark no-change screenshots, but are
    synthetic stress cases and are reported separately from the official test metrics.
    """
    controls = frame[(frame.task_type == "no_diff") & (frame.split == split)]
    if controls.empty:
        raise ValueError("no held-out no-change controls")
    controls = controls.sample(min(n_controls, len(controls)), random_state=seed)
    records = []
    for row in controls.itertuples(index=False):
        a = _rgb(frame.attrs["cache_root"] / row.before_path)
        for nuisance, b in _nuisance_variants(a).items():
            delta = residual_map(a, b)
            mask = threshold_mask(delta, threshold)
            decision, reason = review_policy(mask, a, b, max_grid_coverage=review_grid_threshold)
            records.append({"source_id": row.id, "nuisance": nuisance,
                            "candidate_pixels": int(mask.sum()),
                            "candidate_detected": bool(mask.any()),
                            "grid_coverage": residual_grid_coverage(mask),
                            "decision": decision,
                            "review_reason": reason,
                            "automatic_false_change": decision == "localized_change"})
    per_case = pd.DataFrame(records)
    summary = (per_case.groupby("nuisance", sort=False)
               .agg(pairs=("source_id", "count"), raw_candidate_rate=("candidate_detected", "mean"),
                    automatic_false_change_rate=("automatic_false_change", "mean"),
                    review_rate=("decision", lambda x: (x == "needs_review").mean()),
                    mean_grid_coverage=("grid_coverage", "mean"),
                    median_candidate_pixels=("candidate_pixels", "median"))
               .reset_index())
    return per_case, summary
