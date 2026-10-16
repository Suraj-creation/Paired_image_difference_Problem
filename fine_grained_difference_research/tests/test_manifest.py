import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utilities.manifest import audit_manifest


class ManifestAuditTests(unittest.TestCase):
    def test_exact_duplicates_and_group_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.bin").write_bytes(b"same image bytes")
            (root / "b.bin").write_bytes(b"same image bytes")
            rows = [
                {"pair_id": "p1", "before": "a.bin", "after": "b.bin", "split": "train", "group_id": "scene"},
                {"pair_id": "p2", "before": "b.bin", "after": "a.bin", "split": "test", "group_id": "scene"},
            ]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            audit = audit_manifest(manifest)
            self.assertEqual(audit["pairs"], 2)
            self.assertEqual(len(audit["exact_duplicate_groups"]), 1)
            self.assertEqual(audit["group_split_conflicts"], [{"group_id": "scene", "splits": ["test", "train"]}])
            self.assertEqual(audit["missing_files"], [])


if __name__ == "__main__":
    unittest.main()
