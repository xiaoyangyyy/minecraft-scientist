"""First-image patch ambiguity proxy, not correspondence or navigation proof.

Compare only separated original safe-seed patches in ONE public image. This is
not exhaustive texture uniqueness, depth, static-world identity or a guarantee
of future visibility. No observed survivor or evaluator score enters ranking.
"""

from dataclasses import dataclass

import numpy as np

from minecraft_scientist.exploration.correspondence_refinement import _crop_gray, _odd
from minecraft_scientist.exploration.safe_track_seeds import safe_seeds
from minecraft_scientist.exploration.temporal_tracks import TemporalTrackConfig
from minecraft_scientist.exploration.visual_geometry import _cells, _frame, _opencv


@dataclass(frozen=True, slots=True)
class ViewQualityProfile:
    maximum_separated_patch_correlation: float = 0.95
    reference_competitor_separation: int = 14
    minimum_distinctive_seeds: int = 32
    minimum_distinctive_cells: int = 4

    def __post_init__(self):
        if (
            type(self.maximum_separated_patch_correlation) is not float
            or self.maximum_separated_patch_correlation != 0.95
            or type(self.reference_competitor_separation) is not int
            or self.reference_competitor_separation != 14
            or type(self.minimum_distinctive_seeds) is not int
            or self.minimum_distinctive_seeds != 32
            or type(self.minimum_distinctive_cells) is not int
            or self.minimum_distinctive_cells != 4
        ):
            raise ValueError("public viewpoint proxy has exact registered parameters")


def patch_quality(gray, points, *, scale=1.0, profile=None):
    """Bounded first-frame-only comparison of already detected safe corners."""
    profile = ViewQualityProfile() if profile is None else profile
    if type(profile) is not ViewQualityProfile:
        raise ValueError("requires the exact viewpoint proxy profile")
    settings, cv2 = TemporalTrackConfig(), _opencv()
    gray, points = np.asarray(gray), np.asarray(points, dtype=np.float32)
    if points.size == 0:
        points = points.reshape(0, 2)
    if (
        gray.ndim != 2
        or gray.dtype != np.uint8
        or min(gray.shape) < 32
        or points.ndim != 2
        or points.shape[1] != 2
        or len(points) > settings.feature_budget
        or not np.isfinite(points).all()
        or type(scale) not in (int, float)
        or not np.isfinite(scale)
        or not 1 <= scale <= 2
    ):
        raise ValueError("requires bounded unchanged first-image patches and seeds")
    patch = _odd(settings.reference_patch_pixels * scale)
    margin = max(patch, _odd(settings.reference_window_pixels * scale)) / 2 + 1
    if len(points) and (
        np.any(points < margin)
        or np.any(points[:, 0] >= gray.shape[1] - margin)
        or np.any(points[:, 1] >= gray.shape[0] - margin)
    ):
        raise ValueError("safe patch seeds cannot leave the original support bounds")
    patches = np.asarray(
        [
            cv2.getRectSubPix(gray, (patch, patch), tuple(map(float, point))).ravel()
            for point in points
        ],
        dtype=np.float64,
    ).reshape(len(points), patch * patch)
    std = patches.std(axis=1) if len(points) else np.empty(0)
    valid = std >= settings.minimum_patch_std
    centered = patches - patches.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(
        centered, norms, out=np.zeros_like(centered), where=norms > 0
    )
    correlations = np.clip(normalized @ normalized.T, -1.0, 1.0)
    separated = np.linalg.norm(points[:, None] - points[None, :], axis=2) >= (
        profile.reference_competitor_separation * scale
    )
    competitors = separated & valid[:, None] & valid[None, :]
    records = []
    for identity in range(len(points)):
        candidates = np.flatnonzero(competitors[identity])
        alternative = (
            int(candidates[np.argmax(correlations[identity, candidates])])
            if len(candidates)
            else None
        )
        maximum = (
            float(correlations[identity, alternative])
            if alternative is not None
            else None
        )
        records.append(
            {
                "seed_id": identity,
                "patch_std": float(std[identity]),
                "textured": bool(valid[identity]),
                "separated_competitor_count": len(candidates),
                "most_similar_competitor_id": alternative,
                "maximum_separated_patch_correlation": maximum,
                "distinctive_proxy": bool(
                    valid[identity]
                    and maximum is not None
                    and maximum < profile.maximum_separated_patch_correlation
                ),
            }
        )
    return records


def score_view(first, *, profile=None):
    profile = ViewQualityProfile() if profile is None else profile
    first, cv2 = _frame(first), _opencv()
    if first.shape not in ((224, 224, 3), (448, 448, 3)):
        raise ValueError("viewpoint proxy requires the registered square RGB sensor")
    seeds = safe_seeds(first)
    gray, offset = _crop_gray(first, cv2)
    gray = cv2.medianBlur(gray, 3)
    records = patch_quality(
        gray, seeds - offset, scale=first.shape[1] / 224, profile=profile
    )
    ids = [record["seed_id"] for record in records if record["distinctive_proxy"]]
    cells = _cells(seeds[ids], *first.shape[:2]) if ids else 0
    return {
        "registration": "first-image-separated-safe-patch-view-quality-v1",
        "seed_pixels": seeds.tolist(),
        "seed_count": len(seeds),
        "patches": records,
        "distinctive_seed_ids": ids,
        "distinctive_seed_count": len(ids),
        "distinctive_cells": cells,
        "eligible_proxy": len(ids) >= profile.minimum_distinctive_seeds
        and cells >= profile.minimum_distinctive_cells,
        "future_correspondence_supported": False,
        "navigation_accepted": False,
    }


def rank_views(images, ordinals):
    if (
        type(images) is not list
        or len(images) != 12
        or type(ordinals) is not list
        or len(ordinals) != 12
        or any(type(slot) is not int for slot in ordinals)
        or ordinals != list(range(15, 49, 3))
    ):
        raise ValueError("requires exactly the twelve completed startup survey images")
    views = [
        score_view(image) | {"absolute_frame": slot}
        for image, slot in zip(images, ordinals)
    ]
    eligible = [view for view in views if view["eligible_proxy"]]
    ordered = sorted(
        eligible,
        key=lambda view: (
            -view["distinctive_seed_count"],
            -view["distinctive_cells"],
            -view["seed_count"],
            view["absolute_frame"],
        ),
    )
    return {
        "views": views,
        "ranked_frame_ordinals": [view["absolute_frame"] for view in ordered],
        "selected_frame": ordered[0]["absolute_frame"] if ordered else None,
        "executed_reorientation": False,
        "navigation_accepted": False,
        "scope": "First-image patch proxy on twelve already consumed startup images; no future survivor, private pose, control execution or map acceptance.",
    }
