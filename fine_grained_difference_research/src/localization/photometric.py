"""Robust photometric compensation as a second localization view.

The raw difference is always retained. A fitted global per-channel affine
transform can suppress broad exposure/color shifts, but can also hide a genuine
global change; its output is therefore evidence, never a final semantic label.
"""

from __future__ import annotations

import numpy as np


def fit_global_affine(before: np.ndarray, after: np.ndarray, *, max_samples: int = 100_000,
                      iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Fit after ~= gain * before + bias independently per RGB channel."""
    _check_pair(before, after)
    height, width, _ = before.shape
    stride = max(1, int(np.ceil(np.sqrt((height * width) / max_samples))))
    x = before[::stride, ::stride].reshape(-1, 3).astype(np.float64) / 255.0
    y = after[::stride, ::stride].reshape(-1, 3).astype(np.float64) / 255.0
    gains = np.ones(3, dtype=np.float64)
    biases = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        xc, yc = x[:, channel], y[:, channel]
        keep = np.ones(len(xc), dtype=bool)
        gain, bias = 1.0, float(np.median(yc - xc))
        for _ in range(iterations):
            if int(keep.sum()) < 8:
                break
            # A flat or narrowly distributed channel cannot identify gain and
            # bias separately. Use a robust offset rather than an unstable OLS fit.
            if np.ptp(xc[keep]) < 0.025:
                gain, bias = 1.0, float(np.median(yc[keep] - xc[keep]))
                break
            design = np.column_stack((xc[keep], np.ones(int(keep.sum()))))
            gain, bias = np.linalg.lstsq(design, yc[keep], rcond=None)[0]
            residual = np.abs(yc - (gain * xc + bias))
            center = np.median(residual[keep])
            mad = np.median(np.abs(residual[keep] - center))
            cutoff = max(center + 3.5 * 1.4826 * mad, 0.015)
            new_keep = residual <= cutoff
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
        huber_gain, huber_bias = np.clip(gain, 0.25, 4.0), np.clip(bias, -0.5, 0.5)
        huber_residual = np.abs(yc - np.clip(huber_gain * xc + huber_bias, 0.0, 1.0))

        # Theil-Sen over per-intensity median correspondences handles clipped
        # exposure transforms that make least-squares residuals look like local edits.
        levels, inverse = np.unique(np.round(xc * 255).astype(np.uint8), return_inverse=True)
        level_x = levels.astype(np.float64) / 255.0
        level_y = np.asarray([np.median(yc[inverse == index]) for index in range(len(levels))])
        if len(level_x) >= 3:
            upper_i, upper_j = np.triu_indices(len(level_x), k=1)
            dx = level_x[upper_j] - level_x[upper_i]
            usable = dx >= 0.04
            slopes = ((level_y[upper_j[usable]] - level_y[upper_i[usable]]) / dx[usable])
        else:
            slopes = np.asarray([], dtype=np.float64)
        if len(slopes):
            robust_gain = float(np.median(slopes))
            robust_bias = float(np.median(level_y - robust_gain * level_x))
        else:
            robust_gain, robust_bias = 1.0, float(np.median(yc - xc))
        robust_gain, robust_bias = np.clip(robust_gain, 0.25, 4.0), np.clip(robust_bias, -0.5, 0.5)
        robust_residual = np.abs(yc - np.clip(robust_gain * xc + robust_bias, 0.0, 1.0))
        if np.quantile(robust_residual, 0.90) < np.quantile(huber_residual, 0.90):
            gains[channel], biases[channel] = robust_gain, robust_bias
        else:
            gains[channel], biases[channel] = huber_gain, huber_bias
    return gains.astype(np.float32), biases.astype(np.float32)


def compensate_global_photometric_change(before: np.ndarray, after: np.ndarray
                                         ) -> tuple[np.ndarray, dict[str, list[float] | float]]:
    """Return absolute residual after fitting one robust global RGB transform."""
    gains, biases = fit_global_affine(before, after)
    x = before.astype(np.float32) / 255.0
    y = after.astype(np.float32) / 255.0
    prediction = np.clip(x * gains.reshape(1, 1, 3) + biases.reshape(1, 1, 3), 0.0, 1.0)
    residual = np.abs(y - prediction).mean(axis=2)
    info = {
        "gain_rgb": gains.tolist(),
        "bias_rgb_normalized": biases.tolist(),
        "transform_strength": float(np.mean(np.abs(gains - 1.0)) + np.mean(np.abs(biases))),
        "interpretation": "global affine RGB fit; nuisance hypothesis only, may suppress real global changes",
    }
    return residual.astype(np.float32), info


def adaptive_threshold_map(score: np.ndarray, *, floor: float = 0.035,
                           mad_multiplier: float = 4.0, ceiling: float = 0.25
                           ) -> tuple[np.ndarray, float]:
    """Threshold residuals using robust global background statistics."""
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("score must be a finite 2-D array")
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median)))
    threshold = float(np.clip(median + mad_multiplier * 1.4826 * mad, floor, ceiling))
    return score >= threshold, threshold


def _check_pair(before: np.ndarray, after: np.ndarray) -> None:
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("before and after must be equally sized HxWx3 arrays")

