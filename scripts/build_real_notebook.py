"""Generate the user-facing real-data execution notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Fine_Grained_Image_Difference_Real_Evaluation.ipynb"

def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")

def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")

cells = [
md("""
# Evidence-grounded fine-grained image difference: real-data execution

This notebook executes the existing alignment-aware pipeline on two real annotated test-split
slices. It saves numeric tables and every visual artifact to `outputs/real_dataset_evaluation/`.
All pixel metrics are computed against the supplied masks at the native A-image resolution.

This is a reproducible CPU execution, not a benchmark-superiority claim. No labels are used to
tune alignment, thresholds, or the evidence policy.
"""),
md("""
## 1. Data provenance, licences, and scope

| Dataset | Annotation / official access | What is used in this execution |
|---|---|---|
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | 637 VHR Google Earth pairs with expert-checked binary building-change masks. The official page links Google Drive/Baidu Drive and restricts use to academic, non-commercial use subject to Google Earth terms. | Fixed test rows from public `ericyu/LEVIRCD_Cropped_256` Parquet **mirror**. It is not represented as an official redistribution; source ID/split/rows/timestamp are written beside data. |
| [SYSU-CD](https://github.com/liumency/SYSU-CD) | Paired 256×256 land-cover change images and binary masks; the official repository is the access authority. | Fixed test rows from public `ericyu/SYSU_CD` Parquet **mirror**. Obtain the official data and comply with current terms before broader use. |

The helper requests only the stated test rows from Hugging Face's dataset-server API, checks each
download is readable, and writes `provenance.json`. It never silently substitutes synthetic data.
The compact fixed slice is an execution check, not the full official benchmark protocol.

**Actually executed stages:** validation → ORB/RANSAC alignment → photometrically robust Lab,
SSIM and edge residuals → native crop re-inspection → flow/illumination/compression/blur nuisance
checks → independent-family evidence reconciliation → accepted/review/abstained claims.
LightGlue/LoFTR, OCR, AnyChange, M³Diff and VLM remain optional and are not claimed as run here.
"""),
code("""
from pathlib import Path
import json, os, platform, sys, time
import cv2, matplotlib.pyplot as plt, numpy as np, pandas as pd, torch

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks": ROOT = ROOT.parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
RUN_ROOT = ROOT / "outputs" / "real_dataset_evaluation"
DATA_ROOT = ROOT / "data" / "real_eval_mirror_slices"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
environment = {"python": sys.version, "platform": platform.platform(), "opencv": cv2.__version__,
               "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "run_root": str(RUN_ROOT)}
(RUN_ROOT / "environment.json").write_text(json.dumps(environment, indent=2))
pd.DataFrame(environment.items(), columns=["field", "value"])
"""),
md("""
## 2. Fetch fixed, held-out slices

Indices were fixed before inspecting labels or predictions. Evaluation and demonstration rows are
disjoint, so cherry-picking cannot change the measured tables. A network error remains a visible
blocker; this notebook will not fall back to made-up data.
"""),
code("""
from idu.real_eval import fetch_hf_test_slice

EVAL_ROWS = {"ericyu/LEVIRCD_Cropped_256": [0, 7, 19, 31, 43],
             "ericyu/SYSU_CD": [0, 11, 23, 37, 53]}
DEMO_ROWS = {"ericyu/LEVIRCD_Cropped_256": [101], "ericyu/SYSU_CD": [127]}
evaluation_pairs, demo_pairs = [], []
for dataset_id, rows in EVAL_ROWS.items():
    evaluation_pairs += fetch_hf_test_slice(dataset_id, DATA_ROOT / dataset_id.replace("/", "__") / "evaluation", rows)
for dataset_id, rows in DEMO_ROWS.items():
    demo_pairs += fetch_hf_test_slice(dataset_id, DATA_ROOT / dataset_id.replace("/", "__") / "demonstrations", rows)
pd.DataFrame(evaluation_pairs + demo_pairs)[["dataset_id", "split", "row_idx", "id", "image_a", "mask"]]
"""),
md("""
## 3. Reproducible configuration

The installed `classical` preset exercises the repository's genuine correspondence, residual,
localization and decision policy with no hidden checkpoint download. Configuration is frozen before
the labelled slice is evaluated and recorded in each result file.
"""),
code("""
from idu import Pipeline, make_config
CFG = make_config("classical", device="cpu", domain="fixed_camera", output_dir=str(RUN_ROOT / "pair_artifacts"),
                  analysis_max_side=512, z_high=6.0, z_low=3.0, min_region_px=12,
                  use_registration=True, use_reinspection=True, use_flow=True,
                  use_texture_features=False, seed=0)
PIPE = Pipeline(CFG)
print(json.dumps(CFG.to_dict(), indent=2))
"""),
md("""
## 4. Baseline and evidence-grounded evaluation

The baseline is an unregistered RGB residual with an image-only Otsu threshold. The improved mask
unions only regions attached to `accepted` or `needs_review` claims; abstentions and semantic-only
hypotheses never count. Metrics are micro pixel precision/recall/F1/Dice/IoU/accuracy plus
pair-level changed/not-changed detection. Pixel accuracy is reported but is not treated as a
standalone quality score because unchanged pixels dominate remote-sensing masks.
"""),
code("""
from idu.real_eval import binary_metrics, evaluate_slice, naive_residual_baseline, read_binary_mask, summarise

baseline_rows = []
for p in evaluation_pairs:
    pred = naive_residual_baseline(p["image_a"], p["image_b"])
    baseline_rows.append({"dataset": p["dataset_id"], "id": p["id"], "row_idx": p["row_idx"],
                          **binary_metrics(pred, read_binary_mask(p["mask"], pred.shape))})
BASELINE_PER_PAIR = pd.DataFrame(baseline_rows)
baseline_summary = pd.DataFrame([{ "dataset": ds, "variant": "naive_unregistered_rgb_otsu", **summarise(x.to_dict("records"))}
                                 for ds, x in BASELINE_PER_PAIR.groupby("dataset", sort=False)])
display(baseline_summary)
"""),
code("""
IMPROVED_PER_PAIR, improved_summary = evaluate_slice(PIPE, evaluation_pairs, out_dir=RUN_ROOT / "improved",
                                                       decisions=("accepted", "needs_review"))
improved_summary.insert(1, "variant", "evidence_grounded_classical")
display(improved_summary)
"""),
md("""
## 5. Ablation: remove registration

Only geometric registration changes in this ablation; the test slice, residual thresholds and
evidence policy remain fixed. This provides a direct, limited measurement of correspondence value
without tuning on the test labels.
"""),
code("""
from dataclasses import replace
NO_REG_PIPE = Pipeline(replace(CFG, use_registration=False))
NO_REG_PER_PAIR, no_reg_summary = evaluate_slice(NO_REG_PIPE, evaluation_pairs,
    out_dir=RUN_ROOT / "ablation_no_registration", decisions=("accepted", "needs_review"))
no_reg_summary.insert(1, "variant", "evidence_no_registration")
COMPARISON = pd.concat([baseline_summary, improved_summary, no_reg_summary], ignore_index=True, sort=False)
COMPARISON.to_csv(RUN_ROOT / "metrics_by_dataset.csv", index=False)
BASELINE_PER_PAIR.to_csv(RUN_ROOT / "baseline_per_pair.csv", index=False)
IMPROVED_PER_PAIR.to_csv(RUN_ROOT / "improved_per_pair.csv", index=False)
NO_REG_PER_PAIR.to_csv(RUN_ROOT / "no_registration_per_pair.csv", index=False)
display(COMPARISON[["dataset", "variant", "pairs", "precision", "recall", "f1", "iou", "dice", "accuracy", "pair_f1", "pair_accuracy", "mean_latency_s"]])
"""),
md("""
## 6. Two additional real-pair demonstrations

These disjoint test rows are excluded from the metric tables. Green overlay is the exact mask that
was measured. The report text is constrained by the repository's claim verifier: without a
successfully executed semantic model it will say localized visual change rather than invent an
object/action label.
"""),
code("""
from idu.real_eval import evidence_mask

def overlay(image, mask):
    out = image.copy(); out[mask] = (0.45 * out[mask] + 0.55 * np.array([0, 255, 0])).astype(np.uint8); return out

DEMO_RESULTS = []
for pair in demo_pairs:
    res, art = PIPE.run(pair["image_a"], pair["image_b"], out_dir=str(RUN_ROOT / "demonstrations" / pair["id"]), do_export=True)
    gt, pred = read_binary_mask(pair["mask"], art["img_a"].shape[:2]), evidence_mask(art)
    metric = binary_metrics(pred, gt); fused = art.get("low", {}).get("maps", {}).get("fused", np.zeros(gt.shape))
    if fused.shape != gt.shape: fused = cv2.resize(fused, (gt.shape[1], gt.shape[0]))
    panels = [(art["img_a"], "A original", None), (art["img_b"], "B original", None),
              (art["aligned_b"], f"B aligned — {res['alignment']['status']}", None),
              (fused, "fused robust residual (z)", "magma"), (gt, "ground truth", "gray"),
              (overlay(art["img_a"], pred), f"evidence prediction; F1={metric['f1']}", None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (image, title, cmap) in zip(axes.flat, panels): ax.imshow(image, cmap=cmap); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{pair['dataset_id']} / test row {pair['row_idx']}"); fig.tight_layout()
    fig.savefig(RUN_ROOT / "demonstrations" / f"{pair['id']}_six_panel.png", dpi=150, bbox_inches="tight"); plt.show()
    print(res["report"]["text"])
    DEMO_RESULTS.append({"dataset": pair["dataset_id"], "row_idx": pair["row_idx"], **metric,
                         "comparison_status": res["comparison_status"], "report": res["report"]["text"]})
pd.DataFrame(DEMO_RESULTS).to_csv(RUN_ROOT / "demonstration_metrics.csv", index=False)
pd.DataFrame(DEMO_RESULTS)[["dataset", "row_idx", "precision", "recall", "f1", "iou", "comparison_status"]]
"""),
md("""
## 7. Measured summary and limitations

The values below are emitted by this run, not copied from a paper. Each pair's `result.json`
records component status, inputs, alignment quality, evidence graph, claims, timings and maps.

Limitations: ten fixed pairs are an execution slice rather than a leaderboard test; binary masks
do not support semantic-caption accuracy; no document/OCR benchmark is included; and no compatible
M³Diff/AnyChange/VLM checkpoint was installed for this CPU run. Optional failures are visible in
the component table rather than silently replaced by predictions.
"""),
code("""
print("Saved tables:")
for path in sorted(RUN_ROOT.glob("*.csv")): print(" -", path)
display(COMPARISON[["dataset", "variant", "pairs", "precision", "recall", "f1", "iou", "dice", "accuracy",
                    "pair_precision", "pair_recall", "pair_f1", "pair_accuracy", "mean_latency_s"]])
print("\\nActual component status from the last executed pair:")
display(pd.DataFrame(res["component_status"]))
PIPE.close(); NO_REG_PIPE.close()
""")
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.13"}})
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(OUT)
