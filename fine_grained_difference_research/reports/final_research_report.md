# First-principles image difference research: initial report

**Status date:** 2026-10-16  
**Scope:** independent research workspace, end-to-end inference pipeline, and controlled synthetic diagnostic  
**Conclusion:** an end-to-end research pipeline is implemented and run using existing cached models. It produces structured claims, separate localization evidence, OCR cues, conservative pair decisions, and reproducible metrics. No specialist image-difference model was reproduced and no public benchmark was evaluated.

## Question and task decomposition

Reliable image comparison requires separate evidence for (1) whether a meaningful change exists, (2) where it is, (3) which entity/property changed and in what direction, and (4) whether an apparent difference is only a nuisance such as lighting. Caption plausibility alone cannot establish spatial grounding or factual support. The structured output and proposed success criteria are in [`problem_definition.md`](../research/problem_definition.md).

## Literature findings

The strongest current first candidate is M³Diff: its ICCV 2025 paper introduces OmniDiff and a multi-scale differential-perception module. The official release describes 15,598 human-captioned pairs across 324 real/rendered scenes and 12 change types, with captions averaging about 60 words. The official release also provides evaluation instructions for several older IDC benchmarks. Its code is Apache-2.0, but datasets, annotations, base model, and weights have separate terms. Its documented training run uses eight A100 40GB GPUs, so faithful training is beyond this machine.

OneDiff offers a Siamese encoder and Visual Delta Module, DUDA remains a relevant distractor-robust historical baseline, CLIP4IDC tests adapted paired visual representations, and MEDIC tests whether type-specialized reasoning helps. Spot-the-Shift (September 2026 preprint) appears to be the best-fit grounded evaluation candidate because it pairs captions with spatial masks, but access and licensing were not confirmed. Full source registry and evidence boundaries: [`literature_review.md`](../research/literature_review.md).

## Environment and execution

The host has an NVIDIA RTX 4000 Ada Generation with 20 GB VRAM. The shared environment contains PyTorch 2.6.0+cu124, Transformers 4.57.6, OpenCV, SciPy, and EasyOCR. Cached Qwen3-VL and EasyOCR weights were reused; no packages or weights were downloaded and no duplicate environment was created. The end-to-end pipeline combines raw RGB difference evidence, a robust global-photometric residual view, connected regions, a global translation cue, optional Qwen3-VL structured claims, and optional local-only OCR. It writes pair-level JSON, masks, overlays, manifests, and run metadata. Different image dimensions are rejected instead of silently resized; no automatic warping is performed.

Thirteen deterministic unit tests pass. The end-to-end run covered 11 constructed pairs. The conservative decision layer covered 9/11 pairs and was correct on all 9 covered cases (81.8% coverage), abstaining on global illumination and camera translation. Localization macro IoU was 0.905, but the masks are from simple constructed scenes with exact rendered pixel differences; the camera-translation nuisance still had IoU 0.0. Researcher manual review of language claims found 4 fully correct, 3 partially correct, and 4 incorrect descriptions. Failures included a hallucinated rectangle movement, an incorrect one-character text-edit description, false semantic color claims under illumination, and a camera-shift false detection. EasyOCR returned `EXITI` before and `EXIT` after, so it did not recover the intended exact text. These results are synthetic diagnostics, not benchmark or independent human-study evidence. See [`qwen3vl_end_to_end_synthetic.json`](../results/metrics/qwen3vl_end_to_end_synthetic.json), [`qwen3vl_end_to_end_manual_review.json`](../results/metrics/qwen3vl_end_to_end_manual_review.json), and [`end_to_end_failure_gallery.png`](../results/visualizations/end_to_end_failure_gallery.png).

## Decision and next research step

Do not propose a new architecture yet. The next step is to obtain legally usable natural image pairs with compatible caption and spatial annotations, then evaluate the pipeline against a reproducible image-difference specialist. Registration and claim-to-region grounding need dedicated experiments. Use CLEVR-Change/DUDA for controlled nuisance testing and investigate Spot-the-Shift access and terms. Introduce MEDIC or other complexity only if error analysis identifies a type-specific failure and an experiment can isolate the gain.

## Limitations

No benchmark dataset or specialist model checkpoint is bundled; no real-image predictions, public mask benchmark scores, independent human study, statistical comparison, or calibrated uncertainty results exist. Qwen used 8.44 GiB peak VRAM and averaged 4.91 seconds per pair in this run. The M³Diff specialist checkpoint is about 16.2 GB and requires a custom LLaVA-based loading path; its inference fit on the 20 GB GPU has not been established, so it was not downloaded. The target domain and relative costs of misses versus false alarms remain unspecified. See [`limitations.md`](../research/limitations.md) and [`experiment_plan.md`](../research/experiment_plan.md).

## References

- [OmniDiff / M³Diff, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OmniDiff_A_Comprehensive_Benchmark_for_Fine-grained_Image_Difference_Captioning_ICCV_2025_paper.html) and [official implementation](https://github.com/Yuan-Liu-01/M3Diff).
- [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and [official Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).
- [Official M³Diff checkpoint card](https://huggingface.co/IVC-liuyuan/M3Diff).
- [OneDiff, ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Hu_OneDiff_A_Generalist_Model_for_Image_Difference_Captioning_ACCV_2024_paper.html).
- [Robust Change Captioning / DUDA, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html).
- [CLIP4IDC, AACL-IJCNLP 2022](https://aclanthology.org/2022.aacl-short.5/).
- [MEDIC official repository](https://github.com/VisualAIKHU/MEDIC).
- [Spot-the-Shift preprint](https://arxiv.org/abs/2609.10356).
