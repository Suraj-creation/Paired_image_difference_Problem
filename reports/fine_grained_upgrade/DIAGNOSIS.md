# Why the previous scores were weak, and what the new run establishes

## Root-cause assessment

The previous real-data run is useful evidence, but it is too small to isolate individual causes: it evaluated five LEVIR-CD pairs and five SYSU-CD pairs. Its full pipeline had pixel-mask F1 of 0.108 on LEVIR-CD and 0.234 on SYSU-CD, below the unregistered RGB/Otsu baseline at 0.374 and 0.406. The model-enabled path also took roughly 69–109 seconds per pair. The no-registration ablation moved in opposite directions across the two datasets (LEVIR 0.089 vs. 0.108; SYSU 0.259 vs. 0.234), so registration alone does not explain the low scores. The SYSU pair-level F1 of 1.0 is not informative about false alarms because all five sampled rows were changed. See [`../real_dataset_evaluation/README.md`](../real_dataset_evaluation/README.md).

The most likely drivers are:

1. **Evaluation sample and metric mismatch.** Five pairs per dataset produce unstable aggregate pixel scores. The earlier SYSU sample had no unchanged pairs, making pair detection look perfect even with poor pixel localization. It also did not cover the nuisance and document/product cases in the problem statement.
2. **Model/domain mismatch.** AnyChange is a remote-sensing change model; DINOv2 is a semantic representation that is designed to be tolerant to appearance details; Qwen3-VL reasons in language and is not a pixel-mask estimator. Passing a component self-test proves that it loads and returns output, not that its output helps this task. The system did not train/fine-tune these models on the evaluated fine-grained domain.
3. **Pipeline fusion and thresholding.** The previous pipeline combined alignment, residual maps, learned proposals, OCR, VLM hypotheses, and heuristic evidence decisions. More branches add failure modes and do not improve a pixel mask unless they are calibrated on representative held-out labels. The previous report has no component-specific ablation that isolates AnyChange, DINOv2, or Qwen3-VL, so their individual contribution cannot be claimed.
4. **Baseline advantage under fixed alignment.** LEVIR/SYSU image pairs are already co-registered for change detection. A direct residual baseline can therefore match the dataset’s pixel-mask objective more closely than a generalized pipeline that spends capacity on uncertain registration and semantic inference. This does not make Otsu a robust general solution: it has no evidence reconciliation or abstention and will react to capture nuisances.

## What the new evaluation adds

The upgraded notebook uses the complete DiffSpot test split, a deterministic 20% calibration set, and a disjoint 80% holdout. It chooses an RGB residual threshold from calibration pair F1, then chooses an 8×8 spatial review gate on calibration positives and synthetic nuisance variants of calibration no-change screenshots. The gate is frozen before holdout evaluation.

On 3,520 holdout pairs, raw residual presence achieved pair precision/recall/F1 of 1.000/1.000/1.000, with 400/400 exact no-change controls correctly rejected. That is strong evidence for the simple detector on this controlled same-viewport benchmark, not proof against natural capture changes. The calibrated review policy auto-accepted 64.2% of true changes with 100% precision on the clean holdout, sending 31.7% of all holdout pairs to review. The target-box center hit rate was 99.8%; 99.3% of residual pixels fell inside the annotated target element. Mean residual-box IoU was 0.221 because the target box marks the full UI element while the predicted box tightly encloses only changed pixels; it should be read as a diagnostic, not as segmentation IoU.

In the held-out synthetic nuisance stress, all 100 examples per transform produced raw residual candidates. The review policy sent all brightness, shadow, and translation variants to review, and 99% of JPEG recompressions to review; one JPEG case was automatically marked localized. That remaining false claim is reported rather than tuned away using holdout data.

Qwen3-VL 4B was separately evaluated on 30 changed crop pairs and 10 no-change controls. It returned `changed=yes` for 5/30 changed pairs and for 0/10 controls (precision 1.0, recall 0.167, F1 0.286). This is too low-recall to use as the detector or a required confirmation gate. Its crop descriptions are retained in the CSV for qualitative review; the smoke-reviewed text example also showed an incorrect transcription, so semantic text/state accuracy is not claimed.

## What remains unproven

- DiffSpot uses rendered web interfaces and exact same-size screenshots. It does not validate physical viewpoint changes, reflections, product defects, document photography, or satellite imagery.
- Its no-change controls are exact duplicates. The nuisance stress variants are synthetic and do not replace real nuisance pairs.
- It provides a target element box rather than a per-pixel change mask. This run intentionally reports pair-level and box-grounding metrics, not pixel F1/IoU or Dice.
- The residual detector can find and locate visible changes, but it does not infer semantic change type. OCR or a VLM can be added as an explanation-only stage, with the pixel evidence and review decision retained as the authority.
- A broad genuine change can be sent to review because it resembles exposure, compression, or displacement. That is an intentional uncertainty outcome, but future work should improve the distinction using paired nuisance data and domain-specific calibration.

## Research basis

- [DiffSpot paper](https://arxiv.org/abs/2605.29615) introduces same-viewport controlled CSS edits, target-element annotations, difficulty tiers, and no-difference controls; it also reports that current VLMs miss many fine-grained changes.
- [DiffSpot dataset card](https://huggingface.co/datasets/tencent/DiffSpot) documents the public data schema and MIT license metadata.
- [Spot-the-Diff (EMNLP 2018)](https://aclanthology.org/D18-1436/) supplies human descriptions for real surveillance image pairs, but not dense masks suitable for pixel localization metrics.
- [Learning to Measure Change](https://arxiv.org/abs/1810.09111) studies illumination, shadows, and viewpoint as nuisance variation in scene change detection, motivating a separate nuisance-aware evaluation rather than assuming screenshot residual scores transfer to scenes.
