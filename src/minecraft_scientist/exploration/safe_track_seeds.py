"""First-image-only corner detection inside unchanged trackable support bounds.

Different detector namespace, not retrospective filtering of failed old IDs.
All quality, spacing, budget, crop and patch/window settings retain defaults.
"""

import numpy as np

from .correspondence_refinement import _crop_gray, _odd
from .temporal_tracks import TemporalTrackConfig
from .visual_geometry import _frame, _opencv


def safe_seeds(first):
    settings = TemporalTrackConfig()
    image, cv2 = _frame(first), _opencv()
    gray, offset = _crop_gray(image, cv2)
    if settings.median_filter:
        gray = cv2.medianBlur(gray, 3)
    scale = image.shape[1] / 224
    margin = (
        max(
            _odd(settings.reference_window_pixels * scale),
            _odd(settings.reference_patch_pixels * scale),
        )
        / 2
        + 1
    )
    rows, columns = np.indices(gray.shape)
    mask = (
        (columns >= margin)
        & (rows >= margin)
        & (columns < gray.shape[1] - margin)
        & (rows < gray.shape[0] - margin)
    ).astype(np.uint8) * 255
    detected = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=settings.feature_budget,
        qualityLevel=settings.corner_quality,
        minDistance=settings.reference_feature_spacing * scale,
        mask=mask,
    )
    seeds = (
        np.empty((0, 2), dtype=np.float32)
        if detected is None
        else detected.reshape(-1, 2)
    )
    return seeds + offset.astype(np.float32)
