# Fine-grained image difference research

An independent research workspace for understanding what changed between two images, where it changed, and whether the evidence supports the conclusion. This workspace was designed from the written research brief and a fresh literature review.

## Current status

- Formal task definition, research review, candidate selection, dataset audit, and experiment protocol are documented in `research/`.
- A transparent pixel-change baseline and separate pair/localization metrics are implemented in `src/` and `scripts/`.
- The baseline and metric integrity checks have been run on deterministic synthetic examples, including an illumination nuisance failure case. These are diagnostics, not benchmark evidence.
- An end-to-end pipeline now emits ordered-pair descriptions, raw and photometric-residual masks, component boxes, visual artifacts, OCR cross-checks, review decisions, and JSON manifests.
- The 11-pair constructed diagnostic ran with cached Qwen3-VL and EasyOCR weights. The final selective pair decision covered 9/11 pairs and was correct on those covered examples; it abstained on the illumination-only and camera-translation cases. Manual claim review found 4 fully correct, 3 partially correct, and 4 incorrect descriptions. This is a small synthetic check, not benchmark evidence.
- The shared environment and caches were reused. No packages or weights were installed or downloaded. No public benchmark or specialist IDC checkpoint was downloaded. See `research/limitations.md`.

## Quick start

Use the existing Python environment from this host (or provide an environment with Pillow and NumPy for pixel-only runs). From this directory:

```bash
python3 scripts/compare_pair.py before.png after.png --output results/predictions/pair.json --overlay results/visualizations/overlay.png
python3 -m unittest discover -s tests -v
```

The optional Qwen3-VL diagnostic uses the already cached model and the existing shared environment (when available):

```bash
/home/vsu/Image_Diff/Paired_image_difference_Problem/.venv/bin/python scripts/run_qwen3vl_diagnostic.py
```

### End-to-end pair and manifest runs

The vision-language model and OCR weights load in local-only mode. The commands below do not download weights:

```bash
python scripts/verify_environment.py
python scripts/verify_data.py data/manifests/qwen3vl_synthetic_diagnostic.json
python scripts/run_dataset.py data/manifests/qwen3vl_synthetic_diagnostic.json \
  --output-dir results/runs/my_run --captioner qwen3vl --ocr easyocr
python scripts/validate_results.py results/runs/my_run/predictions_manifest.json
python scripts/evaluate_masks.py results/runs/my_run/predictions_manifest.json \
  --output results/metrics/my_run.json
```

Use `scripts/run_difference_pipeline.py BEFORE AFTER --output-dir OUT` for one pair. Add `--captioner qwen3vl` and/or `--ocr easyocr` to enable cached models. Images must have equal dimensions; resizing or registration must be done explicitly and recorded. The pipeline preserves raw and photometrically compensated maps, estimates global translation as a review cue, and never treats model confidence or claim-to-region associations as validated.

The raw comparison script reports pixel variation and connected regions only. The new pipeline adds general-model descriptions, OCR evidence, nuisance cues and abstention; it does not automatically align or warp images. Thresholds require validation data before real use.

## Package map

- `research/problem_definition.md`: tasks, structured output, and proposed success criteria.
- `research/literature_review.md`: primary-source review and evidence boundaries.
- `research/model_selection.md`: candidate comparison and execution decision.
- `research/dataset_audit.md`: annotation, licensing, leakage, and benchmark gaps.
- `research/experiment_plan.md`: staged, falsifiable protocol.
- `research/limitations.md`: completed work and remaining dependencies.
- `src/`: pixel and photometric localization, translation cue, Qwen/EasyOCR adapters, analysis pipeline, metrics and manifest integrity utilities.
- `scripts/`: pair and dataset inference, environment/data/result validation, evaluation and visualization.
- `tests/`: deterministic metric and baseline checks.
- `results/runs/qwen3vl_end_to_end/`: full pair outputs, masks, overlays, run metadata and prediction manifest.
- `results/metrics/qwen3vl_end_to_end_synthetic.json`: pair/selective and localization metrics. `qwen3vl_end_to_end_manual_review.json` records the bounded claim review.
- `results/visualizations/end_to_end_failure_gallery.png`: selected real outputs, masks and failure notes from the constructed diagnostic.

## Result interpretation

Do not interpret a pixel-change mask as semantic localization. Do not treat caption overlap metrics as proof of factual correctness. Any reported scientific result must identify whether it is published, locally reproduced, newly measured, or a hypothesis.
