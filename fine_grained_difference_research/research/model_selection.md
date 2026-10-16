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

## Executed local diagnostic

The host has an NVIDIA RTX 4000 Ada Generation with 20 GB VRAM. The final 11-pair run reused the shared Python environment and cached `Qwen/Qwen3-VL-4B-Instruct` and EasyOCR weights; no package or model download was needed. The raw RGB pixel baseline and photometric-residual masks were evaluated on the same constructed pairs. Macro/micro IoU were 0.723/0.148 for raw RGB and 0.905/0.913 for the residual. These scores are controlled synthetic diagnostics, not benchmark evidence. The pair decision covered 9/11 at 100% selective accuracy; manual description review found 4 correct, 3 partial, and 4 incorrect cases. Peak allocated VRAM was 8.44 GiB and mean Qwen generation latency was 4.91 seconds per pair. See the final submission report and its linked run metrics.

M³Diff/OmniDiff remains the first specialist reproduction candidate, but its approximately 16.2 GB checkpoint and custom LLaVA-based loader have unverified inference fit on this GPU. Benchmark data access and terms were not confirmed. The checkpoint was not downloaded; this is a compatibility and evaluation gate, not evidence against the method.

## Fair-comparison rules

- Keep official splits where valid, audit grouping/leakage, and freeze prompts/preprocessing.
- Compare models on identical image pairs, annotation versions, and caption metrics.
- Report caption scores, semantic correctness, and spatial metrics separately.
- Include null-change pairs and hard nuisance subsets; report false alarms and missed changes.
- Record checkpoint ID/hash, code commit, software, hardware, runtime, and any deviation from paper.
- Treat published leaderboards as candidate discovery, not proof of superiority for a new domain.
