# Problem definition

## Objective

Given ordered images `(before, after)`, infer a set of meaningful changes, spatial evidence for each change, a directionally correct account of the before/after states, and uncertainty. The target function is:

`(I_before, I_after) -> (C, L, S, U)`

where `C` is the set of semantic changes, `L` their spatial support, `S` structured state descriptions, and `U` uncertainty/evidence limits. A single model need not produce all components.

## Four separately measured tasks

1. **Pair recognition:** meaningful change present, absent, or indeterminate.
2. **Localization:** which pixels/regions/objects support each change. Pixel masks, boxes, and points are different annotation levels and must not share mislabeled metrics.
3. **Change understanding:** entity, property, type, and before-to-after direction.
4. **Nuisance discrimination:** distinguish semantic changes from illumination, shadows, compression, camera/viewpoint shift, and other non-target variation where the image evidence permits.

## Structured change record

Each item should contain `change_id`, `type`, `entity`, `before_state`, `after_state`, `changed_property`, `location_before`, `location_after`, `mask_or_box`, `evidence`, `confidence`, and `review_required`. Null spatial values are permitted where labels are unavailable. `type` is drawn from addition, removal, text edit, color/material, shape/size, movement, orientation/pose, state, layout/structure, other, no-change, or uncertain. Multi-change pairs contain multiple records.

## Operational definitions

- **Meaningful change:** a change in depicted content/state that the target application intends to report. Pure pixel variation is not sufficient.
- **Grounded statement:** every entity, property, direction, and spatial claim is supported by visible evidence in the ordered pair.
- **Uncertain:** the pair does not make a unique conclusion reliable (e.g. occlusion, extreme image quality, unresolved alignment); uncertainty is an allowed outcome, not a semantic class of convenience.
- **Nuisance:** visible variation that should not count as a target semantic change under the declared application policy.

## Proposed acceptance criteria (to be ratified for a target domain)

Report separate held-out results for pair precision/recall/specificity; per-image and pixel localization; change-type, entity, property, and direction accuracy; multi-change completeness; unsupported-claim rate; nuisance-only false-positive rate; uncertainty calibration/coverage; and runtime/memory. A deployable threshold must be selected on validation data and frozen before final evaluation. Numerical thresholds cannot be responsibly fixed until the application cost of misses versus false alarms is known.

Minimum scientific claim: the proposed system must improve a relevant primary metric over the strongest reproducible specialist baseline on a leakage-resistant held-out set, with uncertainty intervals and no material regression in false claims or grounding. Caption metrics such as CIDEr/SPICE are secondary and do not establish grounding or truthfulness.
