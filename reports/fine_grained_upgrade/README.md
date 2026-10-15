# Fine-grained upgrade evaluation report

**Dataset:** Tencent DiffSpot, complete test split (4,400 screenshot pairs; MIT per dataset card).
**Split:** deterministic stratification by task/difficulty; 880 calibration, 3,520 holdout.
**Threshold:** 16 max-channel RGB levels, selected using calibration pair F1 only.
**Review gate:** 0.075 maximum 8×8 residual-grid coverage, selected on calibration positives subject to zero automatic false changes on calibration nuisance transforms.
**Holdout runtime:** 185.8 seconds.

## Calibrated residual method

- Pair precision: 1.0000
- Pair recall: 1.0000
- Pair F1: 1.0000
- No-change false-positive rate: 0.0000
- Automatic change precision after review gate: 1.0000
- Automatic change recall after review gate: 0.6423
- Automatic change F1 after review gate: 0.7822
- Holdout review rate: 0.3170
- Mean target-box IoU of residual extent: 0.2205
- Target-box center hit rate: 0.9978
- Fraction of residual pixels inside the labeled target element: 0.9929

Qwen3-VL crop-reinspection diagnostic (40 balanced holdout pairs): F1 0.286, precision 1.000, recall 0.167, no-change specificity 1.000, uncertain rate 0.000. This small model diagnostic is separate from the detector; it does not establish semantic caption accuracy.

These are pair-detection and element-box grounding metrics. The benchmark does not provide pixel masks; pixel F1, pixel IoU, Dice, pixel accuracy, OCR CER, semantic type accuracy, calibration ECE, and claim hallucination rates are therefore not reported here. Residual containment is a benchmark grounding check, not pixel-mask precision against an annotated change mask.

Pair F1 counts a detected residual candidate, including candidates held for review. Automatic precision/recall/F1 count only `localized_change` decisions. This separates finding a visual discrepancy from asserting that it is a meaningful change.

## Nuisance stress (derived controls)

- global_brightness: raw residual candidate rate 1.000; automatic false-change rate 0.000; review rate 1.000
- local_shadow: raw residual candidate rate 1.000; automatic false-change rate 0.000; review rate 1.000
- two_pixel_translation: raw residual candidate rate 1.000; automatic false-change rate 0.000; review rate 1.000
- jpeg_recompression_q70: raw residual candidate rate 1.000; automatic false-change rate 0.010; review rate 0.990

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
