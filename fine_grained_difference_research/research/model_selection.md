# Model selection decision

## Shortlist

| Candidate | Keep for investigation? | Why / gate before use |
|---|---|---|
| M³Diff | **Yes: first modern baseline** | Direct fine-grained IDC target, official weights/data and evaluation path exist. Gate on hardware, base model/data access, license compatibility, and a benchmark with adequate spatial labels. |
| DUDA | **Yes: controlled historical baseline** | Specifically models distractors and attention localization; CLEVR-Change gives controlled nuisance conditions. Gate on dependencies and a fair reproduction protocol. |
| MEDIC + DIRL | **Yes: specialization experiment only** | Public code, configurations, reported comparisons. Gate on rerunnable setup and category annotations; not a standalone grounded-mask solution. |
| OneDiff | **Conditional comparator** | Relevant Siamese/Visual Delta design. Verify official weights, data access, license, and inference path before spending implementation time. |
| CLIP4IDC | **Conditional comparator** | Relevant CLIP adaptation hypothesis, likely smaller than 7B MLLM; first verify executable code/checkpoint and exact splits. |

## Decision

Do not train a new architecture now. First reproduce M³Diff inference on a caption benchmark and evaluate a spatially grounded benchmark separately if its data are released with usable terms. Pair this with DUDA/CLEVR-Change for nuisance controls. Test MEDIC only if per-type analysis demonstrates a specific failure pattern and its training/data can be reproduced. Select the smallest set that answers these questions.

## Local hardware update and diagnostic

The host has an NVIDIA RTX 4000 Ada Generation with 20 GB VRAM. Rather than create another environment or fetch another checkpoint, the diagnostic reused the existing Python environment and cached `Qwen/Qwen3-VL-4B-Instruct` weights. No packages or weights were downloaded. On eight constructed pairs, the general-purpose VLM was manually judged correct on six (0.75): it handled no-change, small addition/removal, color change, and movement in both image orders. It misread a one-character sign edit and reported global brightening as a background color change. Mean generation time was 0.699 seconds per pair and peak allocated VRAM was 8.39 GiB. This small synthetic check is qualitative, not a benchmark or evidence of specialist performance. See `results/predictions/qwen3vl_zero_shot_synthetic.json` and `results/metrics/qwen3vl_synthetic_manual_review.json`.

M³Diff remains the first specialist candidate. Its official checkpoint is about 16.2 GB and requires the project's custom LLaVA-based loading path; authors report training on eight A100 40GB GPUs. A 20 GB card may or may not support its inference path with adequate activation headroom. Do not fetch that large checkpoint until the inference precision, runtime dependencies, and a legally usable evaluation set are confirmed. This storage and compatibility gate is not evidence against M³Diff.

## Fair-comparison rules

- Keep official splits where valid, audit grouping/leakage, and freeze prompts/preprocessing.
- Compare models on identical image pairs, annotation versions, and caption metrics.
- Report caption scores, semantic correctness, and spatial metrics separately.
- Include null-change pairs and hard nuisance subsets; report false alarms and missed changes.
- Record checkpoint ID/hash, code commit, software, hardware, runtime, and any deviation from paper.
- Treat published leaderboards as candidate discovery, not proof of superiority for a new domain.
