# Dataset and annotation audit

## Candidate assets

| Dataset | Main evidence | Grounding / nuisance value | Audit decision |
|---|---|---|---|
| OmniDiff | 15,598 pairs, 324 scenes, real and rendered subsets, 12 change types, detailed human captions (per ICCV 2025 paper and official repo). | Strong breadth and detailed semantic text. Available release description reviewed does not establish pixel masks. | Preferred modern caption set; inspect official split/provenance and exact annotation schema before metrics. |
| CLEVR-Change | Nearly 80K pairs, five semantic scene-change categories, with illumination/viewpoint distractor conditions (DUDA paper). | Controlled semantic and nuisance factors; synthetic scenes; supports precise generated scene changes. | Strong controlled stress test; report apart from natural imagery. |
| CLEVR-DC | Controlled rendered paired-image change captions, supported by MEDIC configurations. | Useful composition/change reasoning; exact mask compatibility to be checked per release. | Secondary controlled benchmark. |
| Spot-the-Diff | Surveillance-derived image pairs and multiple captions; released repository includes resized images and difference images. | Real imagery, but original paper assumes a change and does not provide change-region ground truth. Source license/use terms inherit VIRAT for images. | Caption evaluation only unless independently annotated; leakage group by video/source. |
| IEdit / Image Editing Request | Editing-request pairs used by CLIP4IDC, OneDiff, M³Diff, MEDIC. | Natural edited-image changes; semantic target can be underspecified and spatial masks not assumed. | Use for caption comparison after auditing data rights/splits. |
| Birds-to-Words | Fine-grained bird pair descriptions, included in OneDiff evaluation/training materials. | Useful fine-grained attributes; narrow domain. | Category-specific diagnostic only. |
| Spot-the-Shift | 2026 preprint: human-verified long-term driving pairs, natural captions plus spatial masks and human-validated caption evaluation. | Best apparent fit for grounded structural change; narrow driving domain. | Verify public data/code, mask schema, split IDs, annotation terms, and image access before adopting. |

## Annotation policy

Do not treat difference images, attention maps, or model heatmaps as ground truth. For each selected set, record whether the annotation represents (a) semantic change presence, (b) caption, (c) pixel mask, (d) box, (e) points, or (f) nuisance labels. A missing field is not a negative label.

## Leakage and split audit requirements

Before tuning, create a manifest containing pair ID, before/after paths, cryptographic hashes, scene/source/video identity, annotation IDs, official split, and image dimensions. Detect exact duplicates and near duplicates; group all pairs from a video, scene, source object, or edited original before splitting. Keep final test captions/masks inaccessible during prompt, threshold, and model selection. If source IDs are absent, report leakage risk and do not imply independent generalization.

## Access, rights, and fit

The M³Diff repo says its code is Apache-2.0, while data derived from third-party benchmarks, annotations, and model weights remain under their own terms. Spot-the-Diff explicitly inherits VIRAT image terms. A dataset cannot be considered usable solely because a download link exists. Keep dataset, annotation, base-model, and fine-tuned-weight licenses in separate records. No images or dataset files are included in this research workspace.
