"""Validate pair-manifest structure and report exact image duplicates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_manifest(manifest_path: str | Path) -> dict:
    path = Path(manifest_path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest root must be a JSON list")
    ids, hashes, missing, invalid, mask_mismatches = set(), {}, [], [], []
    for row in rows:
        if not isinstance(row, dict):
            invalid.append({"row": len(ids), "reason": "row must be a JSON object"})
            continue
        for field in ("pair_id", "before", "after", "split", "group_id"):
            if field not in row:
                raise ValueError(f"missing required field {field!r}")
        if row["pair_id"] in ids:
            raise ValueError(f"duplicate pair_id {row['pair_id']!r}")
        ids.add(row["pair_id"])
        image_sizes = {}
        for field in ("before", "after", "truth_mask"):
            if field == "truth_mask" and not row.get(field):
                continue
            image = (path.parent / row[field]).resolve()
            if not image.is_file():
                missing.append(str(image))
                continue
            if field in ("before", "after"):
                digest = sha256_file(image)
                hashes.setdefault(digest, []).append({"pair_id": row["pair_id"], "field": field, "path": str(image)})
            try:
                from PIL import Image
                with Image.open(image) as opened:
                    image_sizes[field] = opened.size
            except Exception as exc:
                invalid.append({"pair_id": row["pair_id"], "field": field,
                                "path": str(image), "reason": f"unreadable image: {exc}"})
                continue
        if image_sizes.get("before") and image_sizes.get("after") and image_sizes["before"] != image_sizes["after"]:
            invalid.append({"pair_id": row["pair_id"], "reason": "before/after dimensions differ",
                            "before_size": image_sizes["before"], "after_size": image_sizes["after"]})
        if image_sizes.get("after") and image_sizes.get("truth_mask") and image_sizes["after"] != image_sizes["truth_mask"]:
            mask_mismatches.append({"pair_id": row["pair_id"], "after_size": image_sizes["after"],
                                    "mask_size": image_sizes["truth_mask"]})
        if "change_present" in row and not isinstance(row["change_present"], bool):
            invalid.append({"pair_id": row["pair_id"], "reason": "change_present must be a boolean"})
    duplicates = [entries for entries in hashes.values() if len(entries) > 1]
    return {"pairs": len(rows), "missing_files": missing, "exact_duplicate_groups": duplicates,
            "group_split_conflicts": _group_split_conflicts([r for r in rows if isinstance(r, dict)]),
            "invalid_rows": invalid, "truth_mask_dimension_mismatches": mask_mismatches}


def _group_split_conflicts(rows: list[dict]) -> list[dict[str, list[str]]]:
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), set()).add(str(row["split"]))
    return [{"group_id": group, "splits": sorted(splits)} for group, splits in groups.items() if len(splits) > 1]
