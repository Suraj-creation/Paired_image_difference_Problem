# Real-data evaluation report

**Run date:** 2026-10-15
**Executed notebook:** [`notebooks/Fine_Grained_Image_Difference_Real_Evaluation.ipynb`](../../notebooks/Fine_Grained_Image_Difference_Real_Evaluation.ipynb)
**Run environment:** NVIDIA RTX 4000 Ada, 20.99 GB VRAM; Python 3.12.3; PyTorch 2.6.0+cu124. See [`environment.json`](environment.json).

## Scope and method

The notebook was executed end to end on 10 fixed evaluation pairs: five test rows from the public `ericyu/LEVIRCD_Cropped_256` mirror and five from the public `ericyu/SYSU_CD` mirror. Two additional held-out rows (LEVIR-CD row 101 and SYSU-CD row 127) were used for demonstrations. Dataset row selection and retrieval provenance are included under [`provenance/`](provenance/).

The primary configuration used LightGlue registration, DINOv2 compatibility and texture features, low-level residual analysis, AnyChange with SAM ViT-B, EasyOCR, Qwen3-VL reasoning, evidence reconciliation, and Qwen3-VL verbalization. Component self-tests also exercised LoFTR and ORB. M³Diff was disabled in this notebook configuration. No calibrator was fitted; confidence values remain uncalibrated.

Metrics are pixel-level micro aggregates across each five-pair dataset slice. The evidence-grounded predictions include both `accepted` and `needs_review` claim masks; abstentions are excluded. The comparison variants are a naive unregistered RGB/Otsu baseline, the full multimodal pipeline, the full pipeline without registration, and the classical-only pipeline.

## Results

| Dataset slice | Variant | Precision | Recall | F1 / Dice | IoU | Pair F1 | Mean latency / pair |
|---|---|---:|---:|---:|---:|---:|---:|
| LEVIR-CD (5) | Naive RGB/Otsu | 0.312 | 0.468 | 0.374 | 0.230 | 0.750 | — |
| LEVIR-CD (5) | Full multimodal | 0.137 | 0.089 | 0.108 | 0.057 | 0.750 | 108.7 s |
| LEVIR-CD (5) | No registration | 0.125 | 0.069 | 0.089 | 0.047 | 0.750 | 88.4 s |
| LEVIR-CD (5) | Classical only | 0.104 | 0.020 | 0.033 | 0.017 | 0.857 | 1.6 s |
| SYSU-CD (5) | Naive RGB/Otsu | 0.447 | 0.372 | 0.406 | 0.255 | 1.000 | — |
| SYSU-CD (5) | Full multimodal | 0.331 | 0.181 | 0.234 | 0.132 | 1.000 | 68.8 s |
| SYSU-CD (5) | No registration | 0.372 | 0.199 | 0.259 | 0.149 | 1.000 | 85.3 s |
| SYSU-CD (5) | Classical only | 0.534 | 0.145 | 0.228 | 0.128 | 1.000 | 1.7 s |

The full system localized annotated changed pixels worse than the simple RGB/Otsu baseline on both slices. Registration improved full-pipeline F1 on this LEVIR-CD slice (0.108 vs. 0.089 without registration), but not on SYSU-CD (0.234 vs. 0.259). The result supports continued work on residual calibration, nuisance rejection, and localization; it does not support a benchmark-superiority claim.

Pair detection scores need careful reading: all five SYSU-CD evaluation rows contain a ground-truth change, so its pair F1 does not measure false alarms on unchanged pairs. LEVIR-CD contains two unchanged rows; the full system predicted changes on both, reflected in its pair precision of 0.60.

## Component execution

All seven specialist self-tests passed: DINOv2, LightGlue, LoFTR, ORB, EasyOCR, AnyChange/SAM ViT-B, and Qwen3-VL. The full evaluation and both demonstrations recorded successful runs of AnyChange, the VLM, and the verbalizer. The no-registration ablation also completed with the full enabled model set after releasing cached models between pipeline instances. M³Diff is marked skipped because it is disabled by configuration.

## Demonstrations and artifacts

- [LEVIR-CD row 101 six-panel result](figures/LEVIR-CD-row-101.png): the ground-truth mask is empty, while the pipeline predicts 3,563 changed pixels (F1 0.000), exposing a false-positive case.
- [SYSU-CD row 127 six-panel result](figures/SYSU-CD-row-127.png): precision 0.429, recall 0.247, F1 0.313, IoU 0.186.

The executed notebook embeds its tables, component status, explanations, and visuals. The CSV files here preserve the aggregate and per-pair metrics, demonstration summaries, component self-tests, and final component status in portable form. Detailed per-pair JSON, overlays, maps, and reports are also available under `outputs/real_dataset_evaluation/` in the working tree; that generated directory is ignored by Git.

## Limits

These are small fixed slices from public dataset mirrors, not the official full benchmark protocols. The sample is too small to establish robust generalization, and the SYSU-CD slice has no unchanged pairs. The results show substantial false-positive area and weak pixel localization. The demonstrations are illustrative, not a separate statistical test. Use the original dataset sources and licenses for broader research or redistribution.
