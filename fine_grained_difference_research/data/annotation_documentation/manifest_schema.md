# Pair manifest schema

Store one JSON object per pair in a JSON list. Paths are relative to the manifest file.

```json
{
  "pair_id": "stable unique identifier",
  "before": "images/before.png",
  "after": "images/after.png",
  "split": "train | validation | test",
  "group_id": "scene/video/source identity used to prevent leakage",
  "change_present": true,
  "change_records": [],
  "truth_mask": "masks/pair.png",
  "annotation_source": "dataset/version/annotator protocol"
}
```

For unlabeled fields, omit rather than implying a negative. A no-change example should explicitly set `change_present` to false. Each record may follow the structure in `research/problem_definition.md`. All coordinates use original image pixels; masks must align with the after-image unless a benchmark specifies another convention. Keep official IDs and split names unchanged in a separate provenance record.
