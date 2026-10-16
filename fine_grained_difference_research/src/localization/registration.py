"""Estimate global translation as a nuisance cue without altering input images."""

from __future__ import annotations

import numpy as np


def estimate_global_translation(before: np.ndarray, after: np.ndarray) -> dict:
    """Estimate phase-correlation shift in pixels; never warp or suppress evidence."""
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("before and after must be equally sized HxWx3 arrays")
    try:
        import cv2
    except ImportError:
        return {"available": False, "candidate": False,
                "interpretation": "OpenCV is unavailable; no translation estimate was computed."}
    a = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(a, b)
    candidate = bool(np.hypot(dx, dy) >= 1.5 and response >= 0.01)
    return {
        "available": True,
        "estimated_shift_xy_px": [float(dx), float(dy)],
        "phase_correlation_response": float(response),
        "candidate": candidate,
        "interpretation": "possible global camera/view translation; estimate is not a registration and may be affected by scene changes",
    }

