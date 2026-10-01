"""Evaluator-only monocular gauge alignment; never an Agent policy input."""

from __future__ import annotations

import numpy as np


def fit_similarity(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a proper, positive-scale similarity on calibration poses only.

    A line or stationary camera does not determine all orientation parameters.
    Such calibration evidence is rejected, rather than returning an arbitrary
    rotation and reporting its error on the same fitted trajectory.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 3
    ):
        raise ValueError("similarity requires paired 3D points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("similarity points must be finite")
    first, second = source - source.mean(axis=0), target - target.mean(axis=0)
    if min(np.linalg.matrix_rank(first), np.linalg.matrix_rank(second)) < 2:
        raise ValueError("similarity gauge is unobservable on collinear poses")
    u, singular, vt = np.linalg.svd(second.T @ first / len(source))
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = u @ np.diag(signs) @ vt
    variance = np.mean(np.sum(first**2, axis=1))
    scale = float(singular @ signs / variance)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("similarity scale is not positive")
    translation = target.mean(axis=0) - scale * rotation @ source.mean(axis=0)
    return scale, rotation, translation
