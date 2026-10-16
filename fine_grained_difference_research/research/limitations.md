# Limitations and status

## Completed

- Reviewed primary-source IDC papers and official releases current to 2026-10-16.
- Defined separate recognition, localization, semantic, nuisance, and uncertainty objectives.
- Audited likely benchmark fit and rights questions at the release-description level.
- Implemented a transparent raw RGB difference baseline and pair/mask metric functions.
- Ran deterministic synthetic checks for exact mask scoring and baseline mechanics, plus a nuisance stress case showing the raw-difference baseline's global-brightness failure.
- Implemented an end-to-end single-pair and manifest pipeline with raw and robust global-photometric residual maps, connected-region boxes, phase-correlation translation cues, optional cached Qwen3-VL description, optional local-only EasyOCR comparison, uncertainty/review decisions, artifact validation, and selective pair/mask evaluation.
- Ran 13 unit tests and an 11-pair constructed diagnostic on the RTX 4000 Ada. The run used 8.44 GiB peak Qwen VRAM and averaged 4.90 seconds of generation per pair. On the synthetic labels, the system covered 9/11 pair decisions with 100% accuracy on covered cases and abstained on two nuisance cases. Localization macro IoU was 0.905; camera translation was a clear false-positive localization failure. Manual review of semantic claims: 4 correct, 3 partial, 4 incorrect.
- Evaluated the raw RGB reference on the same 11 pairs: macro/micro IoU were 0.723/0.148, compared with 0.905/0.913 for the photometric-residual masks. Both approaches scored 0.000 IoU on camera translation. The result report documents the pair-level and case-specific comparison.

## Not completed

- No official dataset or specialist model checkpoint was downloaded. No packages or model weights were installed for the diagnostic; an existing shared virtual environment and cached general-purpose checkpoint were reused.
- No public benchmark, real-world held-out sample, independent human study, calibrated confidence estimate, or fine-tuning result exists. All 11 current evaluation pairs are constructed examples, and masks are exact rendered differences or hand-authored nuisance exclusions.
- Spot-the-Shift public release, exact mask access, and license were not verified from its arXiv landing page; it remains a candidate preprint benchmark.
- No deployment domain or application costs were supplied, so target operating thresholds cannot be fixed.
- The pipeline does not perform image warping or registration. It estimates a global translation cue, but camera movement still produces false-positive regions. The affine photometric compensator is a secondary view and may suppress real broad changes. Qwen claims are not grounded to regions automatically; EasyOCR is unreliable on the one-character text case. Confidence remains uncalibrated, and human review remains required for every pair.

## Resource and reproducibility constraint

The host provides an NVIDIA RTX 4000 Ada with 20 GB VRAM. The shared environment already has PyTorch 2.6.0+cu124, Transformers 4.57.6, OpenCV, SciPy, and EasyOCR; the Hugging Face and EasyOCR caches already contain the models used. They were reused without creating a second environment, installing packages, or downloading weights. The GPU is not exposed through the default sandbox, so the approved host execution path was used. The Qwen model's peak allocated memory in the final run was 8.44 GiB. M³Diff's official checkpoint is about 16.2 GB, uses a custom LLaVA-based loader, and its authors report training on eight NVIDIA A100 40GB GPUs. Its inference fit and dependencies on this card remain unverified; it was not fetched to avoid filling storage before compatibility and dataset access are established.
