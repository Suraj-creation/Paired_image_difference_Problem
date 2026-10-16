#!/usr/bin/env python3
"""Audit manifest paths, exact duplicate image content, and group split leakage."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utilities.manifest import audit_manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
result = audit_manifest(args.manifest)
serialized = json.dumps(result, indent=2) + "\n"
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
print(serialized, end="")
