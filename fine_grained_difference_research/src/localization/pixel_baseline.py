"""Transparent raw pixel-change baseline. This is not semantic change understanding."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def difference_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Mean absolute channel difference on [0, 1], requiring equal RGB shapes."""
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("before and after must be equally sized HxWx3 arrays")
    delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
    return delta.mean(axis=2, dtype=np.float32) / 255.0


def connected_regions(mask: np.ndarray, min_area: int = 1) -> list[dict[str, int]]:
    """8-connected component boxes with area; coordinates are half-open [x0,x1)."""
    if mask.ndim != 2 or min_area < 1:
        raise ValueError("mask must be 2-D and min_area must be >= 1")
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    regions = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        q = deque([(int(y), int(x))])
        seen[y, x] = True
        area = 0
        x0 = x1 = int(x)
        y0 = y1 = int(y)
        while q:
            cy, cx = q.popleft()
            area += 1
            x0, x1 = min(x0, cx), max(x1, cx)
            y0, y1 = min(y0, cy), max(y1, cy)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        if area >= min_area:
            regions.append({"x0": x0, "y0": y0, "x1": x1 + 1, "y1": y1 + 1, "area": area})
    regions.sort(key=lambda region: (-region["area"], region["y0"], region["x0"]))
    return regions


def filter_small_components(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    """Return a mask containing only complete 8-connected components >= min_area."""
    if mask.ndim != 2 or min_area < 1:
        raise ValueError("mask must be 2-D and min_area must be >= 1")
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    kept = np.zeros(mask.shape, dtype=bool)
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        q = deque([(int(y), int(x))])
        seen[y, x] = True
        component = [(int(y), int(x))]
        while q:
            cy, cx = q.popleft()
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
                        component.append((ny, nx))
        if len(component) >= min_area:
            yy, xx = zip(*component)
            kept[np.asarray(yy), np.asarray(xx)] = True
    return kept


def overlay_image(after: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Blend candidate pixels in red over the after image for inspection."""
    if mask.shape != after.shape[:2]:
        raise ValueError("mask dimensions must match image")
    base = after.astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    blended = np.where(mask[..., None], 0.45 * base + 0.55 * red, base)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
