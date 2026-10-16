# Literature review: paired-image difference understanding

Review date: 2026-10-16. Sources below are primary papers or official project releases unless explicitly called out. Published claims are not local reproductions.

## Task framing

Image difference captioning (IDC) joins two problems: localizing visual evidence and expressing semantic differences. This distinction is explicit in the OmniDiff paper and is central to this project. Caption similarity cannot establish that the model selected the correct region or preserved before/after direction.

## Direct comparison candidates

| Work | Evidence-backed contribution | Fit and key limitation |
|---|---|---|
| **M³Diff / OmniDiff** (ICCV 2025) | OmniDiff contains 15,598 annotated pairs from 324 indoor/outdoor real and rendered scenes, 12 change types, and captions averaging about 60 words. M³Diff extends LLaVA-OneVision-7B with a multi-scale differential perception module using SigLIP features at multiple depths. Official repo provides evaluation for OmniDiff, Spot-the-Diff, IEdit, CLEVR-Change, CLEVR-DC. | Best first reproduction candidate because both model and benchmark target fine-grained IDC and detailed captions. However, the model is an 8B-class MLLM and captioning does not itself provide ground-truth mask accuracy. Official training setup reports 8 A100 40GB GPUs, batch 256, ~26 h; test-time hardware still needs measurement. Dataset images and weights have separate terms. |
| **OneDiff** (ACCV 2024) | Siamese image encoder plus Visual Delta Module; two-phase coupled-sample and multi-task training using DiffCap. Evaluates on Spot-the-Diff, Image-Editing-Request, Birds-to-Words. | Strong direct IDC comparator. Paper/repository/checkpoint reproduction and licensing need validation before selection; published caption scores are not grounded-mask evidence. |
| **DUDA** (ICCV 2019) | Dual Dynamic Attention localizes before/after evidence and Dynamic Speaker generates descriptions; CLEVR-Change was built with illumination/viewpoint distractors and five semantic change types. | Essential historical nuisance-robust baseline and useful controlled data. Older architecture and synthetic domain limit relevance to unconstrained images; localization was attention-based, not a general mask predictor. |
| **MEDIC** (ECCV 2026) | Two-stage change/no-change and change-type router, then type-specialized key-value memory experts; official release currently integrates with DIRL. Repository reports results on CLEVR-DC, CLEVR-Change, Spot-the-Diff, and IEdit. | Tests whether change-type specialization helps. The release depends on legacy feature/captioning pipelines and supplied benchmark settings. Its reported caption metric gains do not show mask grounding; results need independent reproduction. |
| **CLIP4IDC** (AACL-IJCNLP 2022) | Adapts a CLIP visual encoder to represent/align image-pair differences rather than independently captioning each image. Reports CLEVR-Change, Spot-the-Diff, IEdit evaluations. | Useful representation-learning comparator, notably for domain-gap and paired-feature hypotheses. Requires a fair reproduction and does not directly solve fine-grained spatial masks. |
| **Spot-the-Shift** (arXiv preprint, 2026-09) | Human-verified long-term driving change benchmark with captions and spatial masks; paper says current MLLMs struggle with fine-grained multi-image spatial capability and proposes a human-validated evaluation protocol. | Most directly aligned grounded benchmark for structural real-world changes. It is a preprint; dataset/code availability, terms, and splits were not verified from the landing page during this review. Treat as a candidate, not assumed-accessible data. |

## Research takeaways

1. Specialist IDC systems explicitly model paired features (DUDA attention, CLIP4IDC adaptation, OneDiff delta module, M³Diff multi-scale differential perception). Generic single-image caption quality is not a sufficient selection criterion.
2. Annotation depth varies: older Spot-the-Diff supplies multiple captions but no spatial ground truth; CLEVR-Change enables controlled distractor studies; OmniDiff adds breadth and longer descriptions but the paper/repo describe caption annotations, not comprehensive pixel masks; Spot-the-Shift is promising for paired caption-plus-mask evaluation.
3. MEDIC reports CIDEr improvements for DIRL+MEDIC (e.g. 84.1 to 99.8 on CLEVR-DC and 40.9 to 45.5 on Spot-the-Diff), but these are author-reported, change-caption metrics. They do not validate localization.
4. No method can be chosen for a target deployment domain until that domain is specified and its change policy and false-positive/miss costs are defined.

## Research gaps and falsifiable questions

- Does multi-scale pairwise feature comparison improve tiny-change localization or merely richer captions?
- Are errors concentrated by type enough to justify specialist routing, after accounting for router mistakes?
- Can language claims be grounded to masks at useful precision/recall on real scenes?
- How do models behave on unchanged pairs and nuisance-only changes? The benchmark mixture and split policy matter.
- Does performance generalize across scene/source identities rather than repeated frames or scene-specific cues?

## Primary-source registry

- [OmniDiff / M³Diff, ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_OmniDiff_A_Comprehensive_Benchmark_for_Fine-grained_Image_Difference_Captioning_ICCV_2025_paper.html)
- [Official M³Diff code and run instructions](https://github.com/Yuan-Liu-01/M3Diff)
- [Official M³Diff model card](https://huggingface.co/IVC-liuyuan/M3Diff)
- [OneDiff, ACCV 2024 paper](https://openaccess.thecvf.com/content/ACCV2024/html/Hu_OneDiff_A_Generalist_Model_for_Image_Difference_Captioning_ACCV_2024_paper.html)
- [DUDA, ICCV 2019 paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Park_Robust_Change_Captioning_ICCV_2019_paper.html)
- [Official DUDA repository](https://github.com/Seth-Park/RobustChangeCaptioning)
- [MEDIC official repository and reported tables](https://github.com/VisualAIKHU/MEDIC)
- [CLIP4IDC, AACL-IJCNLP 2022 paper](https://aclanthology.org/2022.aacl-short.5/)
- [Spot-the-Diff official dataset repository](https://github.com/harsh19/spot-the-diff)
- [Spot-the-Shift arXiv record](https://arxiv.org/abs/2609.10356)

## Evidence boundary

The descriptions above summarize claims in the cited papers/releases. No result in this document is a locally reproduced score. Repository instructions and model cards are mutable; record commit/revision and checkpoint hashes in any future experiment log.
