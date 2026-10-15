"""Build the executable DiffSpot-focused upgrade notebook."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Fine_Grained_Image_Difference_Upgrade_Evaluation.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(r"""
# Fine-Grained Image Difference Upgrade: DiffSpot End-to-End Evaluation

This notebook evaluates a lean, evidence-first screenshot comparison path on the **complete 4,400-pair DiffSpot test split**. The data contains 3,900 controlled, fine-grained changed pairs and 500 no-change controls. Each pair has a target element box, but **no pixel-level change mask**.

The earlier LEVIR-CD/SYSU-CD report showed that the full multimodal pipeline scored below the simple residual baseline on both five-pair slices (F1 0.108 vs 0.374 on LEVIR-CD; 0.234 vs 0.406 on SYSU-CD). Those results were pixel-mask scores from a tiny remote-sensing sample and are not directly comparable to the box-grounded benchmark below. They do show that registration, model fusion, and thresholds need measured evidence before inclusion.

## Design decisions from the problem statement

1. Keep the two source screenshots at native size and compare their pixels directly. The benchmark uses the same 1280×800 viewport and filters examples so visible residuals are inside the changed element. A geometric warp can smear, attenuate, or erase the subtle CSS changes under test.
2. Use the residual only to propose and localize changed pixels. Emit a location, pixel evidence, and an explicit “semantic type not inferred” status; never let a VLM invent a mask.
3. Calibrate a single residual threshold on a deterministic 20% calibration split, then report all metrics on the disjoint 80% holdout. The threshold is selected using pair-level F1 only.
4. Compare against per-image Otsu and a fixed-threshold residual. No pixel F1/IoU is computed because the target boxes are element annotations, not segmentation masks.

## Component selection

| Component | Decision for this benchmark | Reason |
|---|---|---|
| Native-resolution RGB residual + threshold | Core | Directly measures the visible appearance change while preserving tiny text/style edits. |
| Connected residual extent / target box grounding | Core | Produces coordinates and permits evaluation against the supplied element box. |
| Per-pair Otsu | Baseline only | Useful reference, but its threshold changes with each page's content and can erase subtle edits. |
| LightGlue / LoFTR / ORB geometric registration | Excluded | Paired screenshots have identical dimensions and fixed viewport; warping risks removing the target signal. |
| DINOv2 semantic similarity / texture | Excluded | Semantic invariance is undesirable when a few pixels or an attribute change matter. |
| AnyChange / SAM remote-sensing change model | Excluded | Model domain and objective do not match web UI style/text changes. |
| Qwen3-VL free-form localization | Excluded from detector | Localization must come from measured pixels; the DiffSpot paper reports low recall for current VLMs, especially hard cases. A VLM may be added later as a bounded explanation-only stage and independently audited. |
| OCR | Optional follow-up, not detector | Useful for reading text edits, but irrelevant to color/spacing/border edits and cannot establish visual change by itself. |

This is a focused web-interface evaluation, not a claim that a single residual method solves natural-scene, product, document, and remote-sensing comparison. Those domains need their own registration and annotation protocols.

## Research and data sources

- [DiffSpot paper (arXiv)](https://arxiv.org/abs/2605.29615): controlled CSS mutations, grounding gate, three difficulty tiers, and no-difference controls.
- [DiffSpot dataset card](https://huggingface.co/datasets/tencent/DiffSpot): schema, published split, and MIT dataset-card license.
- [Spot-the-Diff (EMNLP 2018)](https://aclanthology.org/D18-1436/): real surveillance pairs with human difference descriptions; useful for semantic description, but no pixel masks for this localization metric.
- [Learning to Measure Change (CosimNet)](https://arxiv.org/abs/1810.09111): explains why illumination, shadows, and viewpoint variation make general scene change detection a different problem from exact same-viewport screenshot diffing.

The benchmark is downloaded through the public Hugging Face streaming interface using the original PNG bytes. Retrieved image files are cached under `data/diffspot_cache/` and excluded from Git; the manifest, provenance, executed notebook, and compact results are committed.
"""),
    code(r"""
from pathlib import Path
import json, os, platform, sys, time
import cv2, matplotlib.pyplot as plt, numpy as np, pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks": ROOT = ROOT.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT / "data" / "diffspot_cache"
REPORT_ROOT = ROOT / "reports" / "fine_grained_upgrade"
REPORT_ROOT.mkdir(parents=True, exist_ok=True)

try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
except Exception:
    cuda_available, gpu_name = False, None

environment = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "opencv": cv2.__version__, "cuda_available": cuda_available,
    "gpu": gpu_name, "benchmark_pairs": 4400,
    "note": "This selected detector uses OpenCV CPU operations; it has no heavy model loads.",
}
(REPORT_ROOT / "environment.json").write_text(json.dumps(environment, indent=2))
pd.DataFrame(environment.items(), columns=["field", "value"])
"""),
    md("## 1. Download and verify the complete public test split\n\nThe retrieval helper caches all 4,400 pairs, records the dataset revision and license metadata, verifies each image decodes, and reuses verified files on later runs."),
    code(r"""
from idu.diffspot import fetch_diffspot, deterministic_split

manifest = fetch_diffspot(DATA_ROOT, progress=True)
manifest = deterministic_split(manifest, calibration_fraction=0.20)
manifest.attrs["cache_root"] = DATA_ROOT
manifest.to_csv(REPORT_ROOT / "diffspot_manifest.csv", index=False)

dataset_summary = (manifest.groupby(["split", "task_type", "difficulty"], dropna=False)
                   .size().rename("pairs").reset_index())
dataset_summary.to_csv(REPORT_ROOT / "split_counts.csv", index=False)
(DATA_ROOT / "split_provenance.json").write_text(json.dumps({
    "calibration_fraction": 0.20,
    "split": "deterministic SHA-256 by (task_type, difficulty), sorted by pair ID hash",
    "calibration_rows": int((manifest.split == "calibration").sum()),
    "holdout_rows": int((manifest.split == "holdout").sum()),
    "threshold_selection_uses": "calibration labels only; pair F1 only",
}, indent=2))
display(dataset_summary)
"""),
    md("## 2. Check actual examples and annotation semantics\n\nThe target box names the changed page element. The benchmark’s grounding gate states that visible differences are restricted to that target, so residual containment is meaningful. A target-element box is not a pixel mask; do not interpret its area as changed pixels."),
    code(r"""
from idu.diffspot import _rgb, residual_map, target_box

example_rows = pd.concat([
    manifest[(manifest.task_type == "visual_diff") & (manifest.difficulty == d)].head(1)
    for d in ("easy", "medium", "hard")
] + [manifest[manifest.task_type == "no_diff"].head(1)], ignore_index=True)
fig, axes = plt.subplots(len(example_rows), 3, figsize=(15, 3.4 * len(example_rows)))
for axrow, (_, row) in zip(axes, example_rows.iterrows()):
    a = _rgb(DATA_ROOT / row.before_path); b = _rgb(DATA_ROOT / row.after_path)
    d = residual_map(a, b)
    gt = target_box(row)
    overlay = a.copy()
    if gt:
        x1,y1,x2,y2 = gt
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,30,30),2)
    axrow[0].imshow(a); axrow[0].set_title(f"Before | {row.difficulty} | {row.mutation_types or row.task_type}")
    axrow[1].imshow(b); axrow[1].set_title("After")
    axrow[2].imshow(d, cmap="inferno", vmin=0, vmax=64); axrow[2].set_title("Absolute RGB residual; target box shown on before")
    for ax in axrow: ax.axis("off")
fig.tight_layout()
fig.savefig(REPORT_ROOT / "benchmark_examples.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    md("## 3. Calibrate on calibration rows only\n\nThe threshold search uses fixed candidates from 1 through 32 RGB levels. The pair decision is a non-empty residual (minimum one pixel); the highest threshold attaining the best calibration pair F1 is selected to suppress low-amplitude noise. No holdout labels are used."),
    code(r"""
from idu.diffspot import THRESHOLDS, select_threshold

calibrated_threshold, threshold_table = select_threshold(manifest, THRESHOLDS, min_pixels=1, progress=True)
threshold_table.to_csv(REPORT_ROOT / "threshold_calibration.csv", index=False)
print(f"Calibration-selected RGB threshold: {calibrated_threshold}; minimum residual pixels: 1")
display(threshold_table)
"""),
    md("## 4. Calibrate the conservative review gate\n\nChoose the most permissive 8×8 grid-coverage threshold that yields zero automatic false-change decisions on synthetic nuisance transforms of calibration no-change pairs. The same threshold must maximize automatic recall on calibration positives under that constraint. The four transforms are generated only from calibration rows; holdout images and labels are not used for gate selection."),
    code(r"""
from idu.diffspot import calibrate_review_gate

review_grid_threshold, review_gate_table = calibrate_review_gate(
    manifest, calibrated_threshold, progress=True)
review_gate_table.to_csv(REPORT_ROOT / "review_gate_calibration.csv", index=False)
print(f"Calibration-selected max residual grid coverage: {review_grid_threshold:.3f}")
display(review_gate_table)
"""),
    md("## 5. End-to-end holdout evaluation\n\nThree methods share the same pair decision rule and target-box evaluator: legacy-style grayscale/Otsu, fixed max-channel RGB residual at 8, and the calibration-selected max-channel RGB residual. The holdout includes changed and no-change controls. A calibrated residual with diffuse/globally explained evidence is `needs_review`; raw candidate detection and automatic change claims are reported separately. Progress is printed every 100 pairs."),
    code(r"""
from idu.diffspot import evaluate_methods, stratified_summary

started = time.perf_counter()
per_pair, metrics = evaluate_methods(manifest, calibrated_threshold,
                                     fixed_threshold=8, min_changed_pixels=1,
                                     review_grid_threshold=review_grid_threshold, progress=True)
elapsed_s = time.perf_counter() - started
by_difficulty = stratified_summary(per_pair)
per_pair.to_csv(REPORT_ROOT / "holdout_per_pair.csv", index=False)
metrics.to_csv(REPORT_ROOT / "holdout_metrics.csv", index=False)
by_difficulty.to_csv(REPORT_ROOT / "holdout_by_difficulty.csv", index=False)
print(f"Holdout run elapsed: {elapsed_s:.1f}s; evaluated rows: {per_pair.row_idx.nunique()}")
display(metrics)
"""),
    md("## 6. Difficulty and edit-type breakdown\n\nAverages are reported per difficulty tier and mutation operator because a single aggregate can hide missed, low-pixel edits. `mean_target_box_iou` compares the residual extent box with the benchmark element box; `residual_inside_target_fraction` checks if predicted residual pixels stay within that element. Neither is a segmentation metric."),
    code(r"""
display(by_difficulty)
mutation_rows = []
for (method, mutation), group in per_pair.groupby(["method", "mutation_types"], dropna=False, sort=True):
    if mutation in ("", "nan"): continue
    r = group.to_dict("records")
    from idu.diffspot import _pair_stats
    mutation_rows.append({"method": method, "mutation_types": mutation, **_pair_stats(r)})
by_mutation = pd.DataFrame(mutation_rows)
by_mutation.to_csv(REPORT_ROOT / "holdout_by_mutation.csv", index=False)
display(by_mutation.sort_values(["method", "pair_f1", "pairs"], ascending=[True, True, False]))
"""),
    md("## 7. Test Qwen3-VL only as a crop-level explanation checker\n\nA fixed, balanced 40-pair holdout sample (10 per change difficulty and 10 no-change controls) tests whether Qwen3-VL crop reinspection adds useful changed/no-change evidence. For changed pairs, the crop is selected by the calibrated pixel residual; the model does not choose coordinates or masks. For no-change controls it sees the full same-size pair. We report only its changed/no-change decisions and abstentions; semantic description quality is retained for review but is not assigned an automatic text-similarity score."),
    code(r"""
import torch
from idu.semantic import HFVLMAdapter
from idu.diffspot import bbox_from_mask, threshold_mask

positive_sample = []
for difficulty in ("easy", "medium", "hard"):
    g = manifest[(manifest.split == "holdout") & (manifest.task_type == "visual_diff") &
                 (manifest.difficulty == difficulty)].sample(n=10, random_state=2026)
    positive_sample.append(g)
negative_sample = [manifest[(manifest.split == "holdout") & (manifest.task_type == "no_diff")]
                   .sample(n=10, random_state=2026)]
vlm_sample = pd.concat(positive_sample + negative_sample, ignore_index=True)

vlm = HFVLMAdapter("Qwen/Qwen3-VL-4B-Instruct", "cuda", use_4bit=False,
                   max_pixels=1024*1024, bbox_format="rel1000", max_new_tokens=256)
vlm_env = vlm.check_environment()
if not vlm_env.get("ok") or not torch.cuda.is_available():
    raise RuntimeError(f"Qwen3-VL diagnostic requires the configured GPU runtime: {vlm_env}")
vlm.load_model()
vlm_rows = []
for n, row in enumerate(vlm_sample.itertuples(index=False), 1):
    a = _rgb(DATA_ROOT / row.before_path); b = _rgb(DATA_ROOT / row.after_path)
    if row.changed:
        delta = residual_map(a, b)
        box = bbox_from_mask(threshold_mask(delta, calibrated_threshold))
    else:
        box = None
    if box:
        x1,y1,x2,y2 = box
        pad = 24
        crop_a = a[max(0,y1-pad):min(a.shape[0],y2+pad), max(0,x1-pad):min(a.shape[1],x2+pad)]
        crop_b = b[max(0,y1-pad):min(b.shape[0],y2+pad), max(0,x1-pad):min(b.shape[1],x2+pad)]
    else:
        crop_a, crop_b = a, b
    verdict = vlm.reinspect(crop_a, crop_b)
    vlm_rows.append({"id": row.id, "difficulty": row.difficulty, "mutation_types": row.mutation_types,
                     "gt_changed": bool(row.changed), "pixel_bbox": box,
                     "vlm_changed": verdict.get("changed", "uncertain"),
                     "vlm_change_type": verdict.get("change_type", ""),
                     "vlm_object": verdict.get("object", ""),
                     "vlm_before": verdict.get("before", ""), "vlm_after": verdict.get("after", ""),
                     "vlm_nuisance": verdict.get("nuisance", ""),
                     "vlm_explanation": verdict.get("explanation", ""),
                     "vlm_raw": verdict.get("raw", "")})
    if n % 5 == 0: print(f"Qwen3-VL crop checks: {n}/{len(vlm_sample)}", flush=True)
del vlm
torch.cuda.empty_cache()
vlm_results = pd.DataFrame(vlm_rows)
vlm_results.to_csv(REPORT_ROOT / "qwen3vl_crop_diagnostic.csv", index=False)
tp = int(((vlm_results.gt_changed) & (vlm_results.vlm_changed == "yes")).sum())
fp = int(((~vlm_results.gt_changed) & (vlm_results.vlm_changed == "yes")).sum())
fn = int(((vlm_results.gt_changed) & (vlm_results.vlm_changed != "yes")).sum())
tn = int(((~vlm_results.gt_changed) & (vlm_results.vlm_changed == "no")).sum())
div = lambda a,b: a/b if b else None
vlm_metrics = {
    "pairs": len(vlm_results), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    "precision": div(tp,tp+fp), "recall": div(tp,tp+fn),
    "f1": div(2*tp,2*tp+fp+fn), "no_change_specificity": div(tn,tn+fp),
    "uncertain_rate": float((vlm_results.vlm_changed == "uncertain").mean()),
}
(REPORT_ROOT / "qwen3vl_crop_metrics.json").write_text(json.dumps(vlm_metrics, indent=2))
print(vlm_metrics)
display(vlm_results[["id", "gt_changed", "vlm_changed", "vlm_change_type", "vlm_nuisance", "vlm_explanation"]])
"""),
    md("## 8. Nuisance stress test and abstention\n\nThe official DiffSpot controls are exact duplicates and do not cover capture nuisances. To check the review policy, 100 held-out no-change screenshots are transformed with global brightness, a broad local shadow, a two-pixel translation, and JPEG quality-70 recompression. These derived cases are **synthetic stress tests**, not additional benchmark examples. The pipeline preserves a residual candidate but emits `needs_review` when the calibrated residual grid coverage exceeds the calibration-selected limit or when global exposure fit / phase correlation flags a likely capture change. This conservative gate trades automatic coverage for safer abstention."),
    code(r"""
from idu.diffspot import nuisance_stress

stress_pairs, stress_summary = nuisance_stress(manifest, calibrated_threshold, n_controls=100,
                                               review_grid_threshold=review_grid_threshold)
stress_pairs.to_csv(REPORT_ROOT / "nuisance_stress_per_pair.csv", index=False)
stress_summary.to_csv(REPORT_ROOT / "nuisance_stress_metrics.csv", index=False)
display(stress_summary)
"""),
    md("## 9. Inspect model choices against measured evidence\n\nThe existing full system’s historical remote-sensing scores are shown as a diagnosis, not as a direct benchmark comparison: those are pixel-mask metrics on LEVIR/SYSU; DiffSpot provides target boxes and no segmentation masks. The decisions here separate localization from semantic explanation and protect against unsupported claims."),
    code(r"""
prior = pd.read_csv(ROOT / "reports" / "real_dataset_evaluation" / "metrics_by_dataset.csv")
display(prior[["dataset", "variant", "pairs", "precision", "recall", "f1", "iou", "mean_latency_s"]])

component_decisions = pd.DataFrame([
    {"component": "same-viewport residual", "role": "keep", "evidence": "native pixel changes retain subtle CSS edits"},
    {"component": "threshold + residual extent", "role": "keep", "evidence": "localization is derived from observed changed pixels"},
    {"component": "geometric matcher / warp", "role": "exclude here", "evidence": "all benchmark pairs share a 1280x800 canvas; warp can suppress tiny edits"},
    {"component": "DINOv2", "role": "exclude here", "evidence": "semantic invariance is counter to pixel-level fine-grained detection"},
    {"component": "AnyChange / SAM", "role": "exclude here", "evidence": "remote-sensing change domain mismatch"},
    {"component": "Qwen3-VL as mask generator", "role": "exclude", "evidence": "free-form localization must not override pixel evidence; published DiffSpot results show low recall"},
    {"component": "Qwen3-VL crop reinspection", "role": "conditional explanation check", "evidence": "pixel-selected crop; measured separately for yes/no and abstention; never writes a mask"},
    {"component": "OCR", "role": "conditional explanation stage", "evidence": "use only for text-region reading after visual localization"},
], columns=["component", "role", "evidence"])
component_decisions.to_csv(REPORT_ROOT / "component_decisions.csv", index=False)
display(component_decisions)
"""),
    md("## 10. Qualitative holdout review\n\nThe saved cases include subtle edits, a hard-tier pair, and a no-change control. The residual extent is outlined in green; the supplied target element is outlined in red. The report does not convert the residual into an unsupported semantic caption."),
    code(r"""
from idu.diffspot import bbox_from_mask, threshold_mask

qual_ids = []
for difficulty in ("easy", "medium", "hard"):
    candidates = manifest[(manifest.split == "holdout") & (manifest.task_type == "visual_diff") &
                          (manifest.difficulty == difficulty)].sort_values("id")
    qual_ids.append(candidates.iloc[0])
qual_ids.append(manifest[(manifest.split == "holdout") & (manifest.task_type == "no_diff")].sort_values("id").iloc[0])
fig, axes = plt.subplots(len(qual_ids), 3, figsize=(15, 3.5 * len(qual_ids)))
for j, (axrow, row) in enumerate(zip(axes, qual_ids)):
    a = _rgb(DATA_ROOT / row.before_path); b = _rgb(DATA_ROOT / row.after_path)
    delta = residual_map(a, b); pred_box = bbox_from_mask(threshold_mask(delta, calibrated_threshold))
    target = target_box(row); overlay = a.copy()
    if target:
        cv2.rectangle(overlay, (target[0],target[1]), (target[2],target[3]), (255,35,35), 2)
    if pred_box:
        cv2.rectangle(overlay, (pred_box[0],pred_box[1]), (pred_box[2],pred_box[3]), (20,255,20), 2)
    axrow[0].imshow(overlay)
    axrow[0].set_title(f"{row.difficulty}; green=residual, red=target element")
    axrow[1].imshow(delta, cmap="inferno", vmin=0, vmax=64)
    axrow[1].set_title(f"Residual ≥ {calibrated_threshold}")
    axrow[2].imshow(b)
    axrow[2].set_title("After")
    for ax in axrow: ax.axis("off")
fig.tight_layout()
fig.savefig(REPORT_ROOT / "qualitative_holdout.png", dpi=130, bbox_inches="tight")
plt.show()
"""),
    md("## 11. Persisted conclusions and limitations\n\nThe notebook writes a concise report and all compact metric tables under `reports/fine_grained_upgrade/`. The actual executed notebook carries its cell outputs and per-stage progress."),
    code(r"""
calib_rows = int((manifest.split == "calibration").sum())
holdout_rows = int((manifest.split == "holdout").sum())
best = metrics.loc[metrics.method == "calibrated_rgb_residual"].iloc[0]
vlm_meta = json.loads((REPORT_ROOT / "qwen3vl_crop_metrics.json").read_text())
stress_lines = "\n".join(
    f"- {r.nuisance}: raw residual candidate rate {r.raw_candidate_rate:.3f}; "
    f"automatic false-change rate {r.automatic_false_change_rate:.3f}; review rate {r.review_rate:.3f}"
    for r in stress_summary.itertuples(index=False))
report = f'''# Fine-grained upgrade evaluation report

**Dataset:** Tencent DiffSpot, complete test split ({len(manifest):,} screenshot pairs; MIT per dataset card).
**Split:** deterministic stratification by task/difficulty; {calib_rows:,} calibration, {holdout_rows:,} holdout.
**Threshold:** {calibrated_threshold} max-channel RGB levels, selected using calibration pair F1 only.
**Review gate:** {review_grid_threshold:.3f} maximum 8×8 residual-grid coverage, selected on calibration positives subject to zero automatic false changes on calibration nuisance transforms.
**Holdout runtime:** {elapsed_s:.1f} seconds.

## Calibrated residual method

- Pair precision: {best.pair_precision:.4f}
- Pair recall: {best.pair_recall:.4f}
- Pair F1: {best.pair_f1:.4f}
- No-change false-positive rate: {best.no_change_false_positive_rate:.4f}
- Automatic change precision after review gate: {best.automatic_change_precision:.4f}
- Automatic change recall after review gate: {best.automatic_change_recall:.4f}
- Automatic change F1 after review gate: {best.automatic_change_f1:.4f}
- Holdout review rate: {best.review_rate:.4f}
- Mean target-box IoU of residual extent: {best.mean_target_box_iou:.4f}
- Target-box center hit rate: {best.target_box_center_hit_rate:.4f}
- Fraction of residual pixels inside the labeled target element: {best.residual_inside_target_fraction:.4f}

Qwen3-VL crop-reinspection diagnostic ({vlm_meta['pairs']} balanced holdout pairs): F1 {vlm_meta['f1']:.3f}, precision {vlm_meta['precision']:.3f}, recall {vlm_meta['recall']:.3f}, no-change specificity {vlm_meta['no_change_specificity']:.3f}, uncertain rate {vlm_meta['uncertain_rate']:.3f}. This small model diagnostic is separate from the detector; it does not establish semantic caption accuracy.

These are pair-detection and element-box grounding metrics. The benchmark does not provide pixel masks; pixel F1, pixel IoU, Dice, pixel accuracy, OCR CER, semantic type accuracy, calibration ECE, and claim hallucination rates are therefore not reported here. Residual containment is a benchmark grounding check, not pixel-mask precision against an annotated change mask.

Pair F1 counts a detected residual candidate, including candidates held for review. Automatic precision/recall/F1 count only `localized_change` decisions. This separates finding a visual discrepancy from asserting that it is a meaningful change.

## Nuisance stress (derived controls)

{stress_lines}

Diffuse residuals are marked `needs_review` rather than automatically asserted as changes. These transforms are synthetic stress checks on real no-change screenshots and do not substitute for naturally captured nuisance pairs.

## Interpretation

This benchmark’s controlled same-viewport setup favors direct pixel evidence. Registration and semantic-embedding models are excluded from the detector because their invariances can discard fine CSS/text edits. Any semantic captioning should be a separate optional stage grounded in the proposed region and verified against visual evidence. This focused result should not be generalized to natural scenes, product photography, physical defects, documents, or remote-sensing images.

## Artifacts

- `holdout_metrics.csv`, `holdout_by_difficulty.csv`, `holdout_by_mutation.csv`
- `review_gate_calibration.csv`
- `qwen3vl_crop_metrics.json`, `qwen3vl_crop_diagnostic.csv`
- `nuisance_stress_metrics.csv`, `nuisance_stress_per_pair.csv`
- `holdout_per_pair.csv`, `threshold_calibration.csv`, `split_counts.csv`
- `benchmark_examples.png`, `qualitative_holdout.png`, `environment.json`
'''
(REPORT_ROOT / "README.md").write_text(report)
print(report)
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3"}}
nbf.write(nb, OUT)
print(OUT)
