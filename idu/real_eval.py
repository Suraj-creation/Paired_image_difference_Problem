"""Small, transparent real-dataset evaluation helpers.

The pipeline deliberately does not bundle third-party datasets.  This module downloads a
*fixed, test-split slice* through the Hugging Face dataset-server API only when a notebook
explicitly asks for it, writes source provenance next to every image, and never uses labels
to tune a prediction threshold.  Dataset mirrors are called mirrors -- they are not claimed
to be official distributions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests


HF_ROWS = "https://datasets-server.huggingface.co/rows"


def _read_url(url: str) -> bytes:
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r.content


def fetch_hf_test_slice(dataset_id: str, out_dir: str | Path, offsets: list[int], *, config="default") -> list[dict]:
    """Fetch a deterministic test slice with imageA/imageB/label image fields.

    The row API is used instead of downloading a whole mirror.  The API response supplies
    temporary asset URLs; stable dataset id, split, row index, and retrieval time are saved
    in ``provenance.json``.  Existing verified files are reused.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for idx in offsets:
        cached = {key: out / f"row_{idx:05d}_{key}.png" for key in ("a", "b", "mask")}
        if all(path.exists() and path.stat().st_size > 0 for path in cached.values()):
            paths = {key: str(path.resolve()) for key, path in cached.items()}
            for path in cached.values():
                if cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED) is None:
                    raise RuntimeError(f"cached file is not readable: {path}")
            rows_out.append({"id": f"{dataset_id.replace('/', '__')}_test_{idx:05d}", "image_a": paths["a"],
                             "image_b": paths["b"], "mask": paths["mask"], "row_idx": int(idx),
                             "dataset_id": dataset_id, "split": "test"})
            continue
        response = requests.get(HF_ROWS, params={"dataset": dataset_id, "config": config,
                                                  "split": "test", "offset": int(idx), "length": 1}, timeout=90)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("rows"):
            raise RuntimeError(f"no test row {idx} returned for {dataset_id}")
        row = payload["rows"][0]["row"]
        paths = {}
        for source_key, local_key in (("imageA", "a"), ("imageB", "b"), ("label", "mask")):
            if source_key not in row or "src" not in row[source_key]:
                raise RuntimeError(f"{dataset_id} row {idx} has no {source_key}.src")
            path = out / f"row_{idx:05d}_{local_key}.png"
            if not path.exists() or path.stat().st_size == 0:
                path.write_bytes(_read_url(row[source_key]["src"]))
            # OpenCV is a cheap integrity check and makes a corrupt cached response fail early.
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"downloaded {source_key} for {dataset_id}/{idx} is not readable")
            paths[local_key] = str(path.resolve())
        rows_out.append({"id": f"{dataset_id.replace('/', '__')}_test_{idx:05d}", "image_a": paths["a"],
                         "image_b": paths["b"], "mask": paths["mask"], "row_idx": int(idx),
                         "dataset_id": dataset_id, "split": "test"})
    provenance = {"dataset_mirror": dataset_id, "mirror_api": HF_ROWS, "config": config, "split": "test",
                  "row_indices": [int(x) for x in offsets], "retrieved_unix": time.time(),
                  "warning": "This is a public mirror slice; consult the original dataset provider for licence and full download."}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return rows_out


def _rgb(path: str) -> np.ndarray:
    a = cv2.imread(path, cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError(f"unreadable image: {path}")
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def read_binary_mask(path: str, shape_hw=None) -> np.ndarray:
    """Read an annotated mask without guessing semantic classes: non-zero means changed."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"unreadable mask: {path}")
    if shape_hw is not None and tuple(m.shape) != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 0


def naive_residual_baseline(path_a: str, path_b: str) -> np.ndarray:
    """Unregistered per-image Otsu RGB-residual baseline (no labels, no correspondence)."""
    a, b = _rgb(path_a), _rgb(path_b)
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(2).astype(np.uint8)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m > 0


def evidence_mask(art: dict, decisions=("accepted", "needs_review")) -> np.ndarray:
    """Rasterize decision-linked masks in original A coordinates.

    Only masks attached to claims with the requested decisions count.  This intentionally
    excludes abstentions and semantic-only hypotheses, making the metric evidence-grounded.
    """
    h, w = art["img_a"].shape[:2]
    out = np.zeros((h, w), bool)
    selected = {c.claim_id for c in art.get("claims", []) if c.decision in decisions}
    for cid in selected:
        item = art.get("masks", {}).get(cid)
        if item is None:
            continue
        mask, origin, scale = item
        if scale != 1:
            mask = cv2.resize(mask.astype(np.uint8), None, fx=1 / scale, fy=1 / scale,
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        x, y = map(int, np.round(origin))
        x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x + mask.shape[1]), min(h, y + mask.shape[0])
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] |= mask[y1 - y:y2 - y, x1 - x:x2 - x]
    return out


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Pixel and pair metrics with explicit, non-inflated zero-denominator handling."""
    if pred.shape != truth.shape:
        pred = cv2.resize(pred.astype(np.uint8), (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    div = lambda n, d: (n / d) if d else None
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
            "f1": div(2 * tp, 2 * tp + fp + fn), "iou": div(tp, tp + fp + fn),
            "dice": div(2 * tp, 2 * tp + fp + fn), "accuracy": div(tp + tn, tp + fp + fn + tn),
            "pair_gt_changed": bool(truth.any()), "pair_pred_changed": bool(pred.any())}


def summarise(rows: list[dict]) -> dict:
    """Micro pixel metrics plus pair-level detection metrics for a fixed evaluated slice."""
    sums = {k: sum(int(r[k]) for r in rows) for k in ("tp", "fp", "fn", "tn")}
    div = lambda n, d: (n / d) if d else None
    tp, fp, fn, tn = (sums[k] for k in ("tp", "fp", "fn", "tn"))
    pair_tp = sum(r["pair_gt_changed"] and r["pair_pred_changed"] for r in rows)
    pair_fp = sum((not r["pair_gt_changed"]) and r["pair_pred_changed"] for r in rows)
    pair_fn = sum(r["pair_gt_changed"] and (not r["pair_pred_changed"]) for r in rows)
    pair_tn = sum((not r["pair_gt_changed"]) and (not r["pair_pred_changed"]) for r in rows)
    return {"pairs": len(rows), **sums, "precision": div(tp, tp + fp), "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "iou": div(tp, tp + fp + fn),
            "dice": div(2 * tp, 2 * tp + fp + fn), "accuracy": div(tp + tn, tp + fp + fn + tn),
            "pair_tp": pair_tp, "pair_fp": pair_fp, "pair_fn": pair_fn, "pair_tn": pair_tn,
            "pair_precision": div(pair_tp, pair_tp + pair_fp), "pair_recall": div(pair_tp, pair_tp + pair_fn),
            "pair_f1": div(2 * pair_tp, 2 * pair_tp + pair_fp + pair_fn),
            "pair_accuracy": div(pair_tp + pair_tn, len(rows))}


def evaluate_slice(pipe, pairs: list[dict], *, out_dir: str | Path, decisions=("accepted", "needs_review")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the existing pipeline, then compute metrics at the annotation's native resolution."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for n, p in enumerate(pairs, 1):
        started = time.perf_counter()
        res, art = pipe.run(p["image_a"], p["image_b"], out_dir=str(out / p["id"]), do_export=True)
        truth = read_binary_mask(p["mask"], art["img_a"].shape[:2])
        pred = evidence_mask(art, decisions=decisions)
        m = binary_metrics(pred, truth)
        raw_rows.append({"dataset": p["dataset_id"], "id": p["id"], "row_idx": p["row_idx"], **m,
                         "latency_s": time.perf_counter() - started, "peak_gpu_gb": res["efficiency"]["gpu"].get("peak_gb"),
                         "comparison_status": res["comparison_status"], "accepted": res["summary_counts"]["accepted"],
                         "needs_review": res["summary_counts"]["needs_review"], "abstained": res["summary_counts"]["abstained"]})
        print(f"[{n}/{len(pairs)}] {p['id']}: F1={m['f1']}, IoU={m['iou']}, {res['comparison_status']}")
    per_pair = pd.DataFrame(raw_rows)
    summary_rows = []
    for ds, frame in per_pair.groupby("dataset", sort=False):
        s = summarise(frame.to_dict("records"))
        s.update({"dataset": ds, "mean_latency_s": frame["latency_s"].mean(), "max_peak_gpu_gb": frame["peak_gpu_gb"].fillna(0).max()})
        summary_rows.append(s)
    return per_pair, pd.DataFrame(summary_rows)
