from dataclasses import FrozenInstanceError

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

import public_view_quality as quality


def texture(size=224):
    gray = cv2.GaussianBlur(
        np.random.default_rng(72).integers(0, 256, (size, size), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_separated_identical_patch_is_ambiguous_not_a_new_identity():
    gray = np.zeros((100, 100), dtype=np.uint8)
    patch = np.random.default_rng(71).integers(0, 256, (7, 7), dtype=np.uint8)
    gray[27:34, 27:34] = patch
    gray[67:74, 67:74] = patch
    rows = quality.patch_quality(gray, [[30, 30], [70, 70]])
    assert [row["seed_id"] for row in rows] == [0, 1]
    assert all(
        row["maximum_separated_patch_correlation"] == pytest.approx(1.0) for row in rows
    )
    assert all(not row["distinctive_proxy"] for row in rows)
    assert [row["most_similar_competitor_id"] for row in rows] == [1, 0]


def test_no_separated_competitor_does_not_invent_uniqueness():
    gray = texture()[:, :, 0]
    rows = quality.patch_quality(gray, [[40, 40], [45, 40]])
    assert all(row["separated_competitor_count"] == 0 for row in rows)
    assert all(row["maximum_separated_patch_correlation"] is None for row in rows)
    assert all(not row["distinctive_proxy"] for row in rows)


@pytest.mark.parametrize("size", [224, 448])
def test_view_score_is_deterministic_first_image_only_and_preserves_all_seeds(size):
    first = texture(size)
    before = first.copy()
    score = quality.score_view(first)
    assert score == quality.score_view(first.copy())
    assert np.array_equal(first, before)
    assert np.array_equal(np.asarray(score["seed_pixels"]), quality.safe_seeds(first))
    assert len(score["patches"]) == score["seed_count"]
    assert 32 <= score["distinctive_seed_count"] <= score["seed_count"] <= 800
    assert score["eligible_proxy"] and score["distinctive_cells"] >= 4
    assert (
        not score["future_correspondence_supported"]
        and not score["navigation_accepted"]
    )


def test_untextured_empty_seeds_are_not_a_usable_view():
    score = quality.score_view(np.zeros((224, 224, 3), dtype=np.uint8))
    assert score["seed_count"] == score["distinctive_seed_count"] == 0
    assert score["patches"] == [] and not score["eligible_proxy"]


def test_tie_breaks_by_original_ordinal_and_no_control_is_executed():
    first = texture()
    result = quality.rank_views(
        [first.copy() for _ in range(12)], list(range(15, 49, 3))
    )
    assert result["ranked_frame_ordinals"] == list(range(15, 49, 3))
    assert result["selected_frame"] == 15
    assert not result["executed_reorientation"] and not result["navigation_accepted"]


def test_no_eligible_survey_abstains():
    result = quality.rank_views(
        [np.zeros((224, 224, 3), dtype=np.uint8)] * 12, list(range(15, 49, 3))
    )
    assert result["selected_frame"] is None and result["ranked_frame_ordinals"] == []


def test_outside_public_tracking_roi_does_not_influence_the_view_score():
    first = texture()
    score = quality.score_view(first)
    changed = first.copy()
    changed[: int(224 * 0.2)] = 0
    changed[int(224 * 0.76) :] = 255
    changed[:, : int(224 * 0.18)] = 0
    changed[:, int(224 * 0.82) :] = 255
    assert quality.score_view(changed) == score


@pytest.mark.parametrize("profile", [False, 0, {}, "default"])
def test_untyped_profiles_cannot_silently_select_defaults(profile):
    with pytest.raises(ValueError, match="exact viewpoint"):
        quality.score_view(texture(), profile=profile)


@pytest.mark.parametrize("change", ["count", "bool", "future", "order"])
def test_no_future_or_retimed_survey(change):
    ordinals = list(range(15, 49, 3))
    frames = [texture()] * 12
    if change == "count":
        frames.pop()
    elif change == "bool":
        ordinals[0] = True
    elif change == "future":
        ordinals[-1] = 51
    else:
        ordinals.reverse()
    with pytest.raises(ValueError, match="twelve completed"):
        quality.rank_views(frames, ordinals)


@pytest.mark.parametrize(
    "values",
    [
        {"maximum_separated_patch_correlation": 0.96},
        {"reference_competitor_separation": 14.0},
        {"minimum_distinctive_cells": True},
        {"minimum_distinctive_seeds": 31},
    ],
)
def test_proxy_cannot_relax_its_registration(values):
    with pytest.raises(ValueError, match="exact registered"):
        quality.ViewQualityProfile(**values)
    with pytest.raises(FrozenInstanceError):
        quality.ViewQualityProfile().minimum_distinctive_cells = 3


@pytest.mark.parametrize(
    "points", [[[0, 0]], [[50, float("nan")]], [[40, 40, 40]], [[40, 40]] * 801]
)
def test_malformed_or_unsafe_seeds_reject(points):
    with pytest.raises(ValueError):
        quality.patch_quality(texture()[:, :, 0], points)
