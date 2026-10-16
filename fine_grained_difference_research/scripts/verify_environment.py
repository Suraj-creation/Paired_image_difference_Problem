#!/usr/bin/env python3
"""Record installed research dependencies, CUDA visibility, and cached Qwen weights."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path


PACKAGES = ("numpy", "Pillow", "torch", "transformers", "opencv-python", "scipy", "huggingface-hub")


def main() -> None:
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "module_importability": {name: bool(importlib.util.find_spec(name))
                                  for name in ("numpy", "PIL", "torch", "transformers", "cv2", "scipy")},
        "cuda": {"available": False, "device_name": None, "device_total_memory_gib": None},
        "qwen3_vl_cached": False,
        "qwen3_vl_cache_path": None,
        "network_or_install_attempted": False,
    }
    if importlib.util.find_spec("torch"):
        import torch
        result["cuda"]["available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["cuda"].update({"device_name": props.name,
                                   "device_total_memory_gib": round(props.total_memory / 1024**3, 2)})
    cache = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
    snapshots = sorted(cache.glob("*")) if cache.is_dir() else []
    if snapshots:
        result["qwen3_vl_cached"] = True
        result["qwen3_vl_cache_path"] = str(snapshots[-1])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

