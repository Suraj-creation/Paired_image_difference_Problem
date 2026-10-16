# Experimental protocol

## Falsifiable hypotheses

- **H1 — paired signal:** specialist paired-image models outperform generic/independent image baselines on direction, change type, and false-change rate.
- **H2 — scale:** multi-scale differential perception improves small-change localization or semantic completeness over matched single-scale features.
- **H3 — specialization:** change-type experts help only where per-type failures are systematic and labels support reliable routing.
- **H4 — grounding:** caption performance does not reliably predict spatial localization; score both independently.
- **H5 — robustness:** nuisance-only and genuine-change-plus-nuisance slices reveal failure modes hidden by average caption scores.

## Phase 0: environment and benchmark gate

The host has an RTX 4000 Ada (20 GB VRAM) and an existing Python environment with PyTorch, Transformers, and a cached Qwen3-VL 4B checkpoint. A zero-shot synthetic diagnostic was run through that shared environment without installing packages or downloading weights. Before any benchmark run, obtain legally usable annotations and verify code/checkpoint compatibility. The specialist M³Diff checkpoint is about 16.2 GB and its custom loader and inference memory have not yet been validated on this card.

## Phase 1: data and protocol

1. Select the target domain and define meaningful-change policy and error costs.
2. Audit schema, annotation types, licenses, IDs, source grouping, and official splits.
3. Create exact-hash and near-duplicate leakage report; group by scene/source/video before partitioning.
4. Pre-register primary endpoints and category strata; keep final evaluation labels sealed.

## Phase 2: reproduce informative baselines

1. Run official M³Diff inference/checkpoint on a caption benchmark in a compatible environment.
2. Run DUDA on CLEVR-Change for controlled nuisance and localization behavior where checkpoint/configs permit.
3. Add MEDIC only for a per-change-type hypothesis; compare matched DIRL vs DIRL+MEDIC on identical splits.
4. If a candidate cannot be reproduced, log the precise failure/deviation; do not substitute an unofficial score as a reproduction.

## Phase 3: metrics before interpretation

- Pair recognition: precision, recall, F1, specificity, balanced accuracy, false-positive rate, separately on no-change pairs.
- Pixel masks: per-image Dice/IoU and pixel precision/recall, micro and macro aggregation, stratified by changed-area fraction.
- Boxes: box IoU/recall, never label as pixel accuracy.
- Semantics: entity, property, type, direction, addition/removal, multi-change completeness, unsupported-claim rate; human verification on a subset.
- Caption: official benchmark metrics (CIDEr, SPICE, METEOR, ROUGE-L where appropriate), secondary to semantic grounding.
- Robustness: nuisance-only false alarm, genuine change retained under nuisance, resolution/compression sensitivity, small translation/view shift.
- Uncertainty: risk-coverage curve and calibration on validation; self-reported language confidence alone is not calibrated confidence.
- Efficiency: hardware, latency/pair, throughput, peak RAM/VRAM, model/checkpoint size.

Use paired bootstrap confidence intervals at the independent scene/source group level where possible. Fix seeds and disclose annotation ambiguity. Do not tune on final labels.

## Phase 4: controlled stress suite

Evaluate one-character text edits, tiny add/remove, small color/shape changes, subtle movement/orientation, multiple changes, unchanged pairs, shadows/illumination, compression/resizing, global shifts with real changes, and irreducibly ambiguous pairs. Natural annotated cases are primary; synthetic perturbations are separate diagnostics. Synthetic transfer is not real-world evidence.

## Phase 5: improvement decision

Diagnose whether errors arise from recognition, localization, direction, nuisance handling, category routing, calibration, data, or evaluation ambiguity. Change one major factor at a time. A proposed component must improve a prespecified primary outcome on held-out data without materially increasing unsupported claims. Stop if confidence intervals do not support a meaningful gain, results fail to generalize, or supervision is unavailable.

## Current executable experiment

The end-to-end run evaluated 11 constructed pairs covering no-change, addition, removal, color, movement/reversal, character edit, illumination-only, multiple changes, a real addition under illumination, and camera translation. Pair-level decisions covered 9/11 and were correct on all 9 covered pairs (selective accuracy 1.0 at 0.818 coverage); illumination and camera translation were sent to review. On the same masks, raw RGB localization scored 0.723 macro / 0.148 micro IoU and the photometric-residual masks scored 0.905 / 0.913. Both scored 0.000 IoU on camera translation. Manual claim review found 4/11 correct, 3/11 partial, and 4/11 incorrect descriptions. These are synthetic diagnostics, not benchmark results. See `reports/final_research_report.md` and the linked raw-baseline and end-to-end metrics.
