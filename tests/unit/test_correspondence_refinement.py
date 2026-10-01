import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.correspondence_refinement import (
    CorrespondenceRefinementConfig,
    RefinedCorrespondences,
    refine_correspondences,
)
from minecraft_scientist.exploration.dense_correspondence import DenseCorrespondences


def _fixture():
    rng = np.random.default_rng(17)
    gray = cv2.GaussianBlur(
        rng.integers(0, 256, (224, 224), dtype=np.uint8), (5, 5), 0.8
    )
    first = np.repeat(gray[:, :, None], 3, axis=2)
    displacement = np.array((2.4, -1.2))
    second = cv2.warpAffine(
        first,
        np.array(((1.0, 0, displacement[0]), (0, 1, displacement[1]))),
        (224, 224),
    )
    points = np.array(
        [(x, y) for x in range(65, 165, 15) for y in range(65, 155, 15)], dtype=float
    )
    guesses = points + displacement + rng.uniform(-0.7, 0.7, points.shape)
    seeds = DenseCorrespondences(
        points0=tuple(map(tuple, points)),
        points1=tuple(map(tuple, guesses)),
        confidence=(0.9,) * len(points),
        image_shape0=(224, 224),
        image_shape1=(224, 224),
        raw_match_count=len(points),
    )
    return first, second, seeds, displacement


def test_bidirectional_pixel_refinement_improves_seed_precision_and_preserves_provenance():
    first, second, seeds, displacement = _fixture()
    before = first.copy(), second.copy()
    result = refine_correspondences(first, second, seeds)
    assert len(result.retained_source_indices) >= 30
    refined = result.correspondences
    np.testing.assert_allclose(
        np.array(refined.points1) - np.array(refined.points0),
        np.broadcast_to(displacement, (len(refined.points0), 2)),
        atol=0.15,
    )
    assert (
        tuple(seeds.points0[index] for index in result.retained_source_indices)
        == refined.points0
    )
    assert (
        tuple(seeds.confidence[index] for index in result.retained_source_indices)
        == refined.confidence
    )
    np.testing.assert_array_equal(first, before[0])
    np.testing.assert_array_equal(second, before[1])


def test_refinement_cannot_observe_hud_or_other_pixels_outside_world_roi():
    first, second, seeds, _ = _fixture()
    rng = np.random.default_rng(18)
    changed0, changed1 = [
        rng.integers(0, 256, first.shape, dtype=np.uint8) for _ in range(2)
    ]
    changed0[44:170, 40:183], changed1[44:170, 40:183] = (
        first[44:170, 40:183],
        second[44:170, 40:183],
    )
    assert refine_correspondences(first, second, seeds) == refine_correspondences(
        changed0, changed1, seeds
    )


def test_flat_or_out_of_roi_seeds_cannot_supply_correspondences():
    first, _, seeds, _ = _fixture()
    flat = np.zeros_like(first)
    assert len(refine_correspondences(flat, flat, seeds).correspondences.points0) == 0
    outside = seeds.model_copy(
        update={
            "points0": ((10.0, 10.0),),
            "points1": ((12.0, 10.0),),
            "confidence": (0.9,),
        }
    )
    result = refine_correspondences(first, first, outside)
    assert result.retained_source_indices == () and result.cycle_consistent_count == 0


def test_empty_seeds_and_dimension_errors_are_explicit():
    first, second, seeds, _ = _fixture()
    empty = seeds.model_copy(update={"points0": (), "points1": (), "confidence": ()})
    assert refine_correspondences(first, second, empty).seed_count == 0
    with pytest.raises(ValueError, match="same-size"):
        refine_correspondences(first[:100], second, seeds)


def test_seed_budget_and_nonfinite_configuration_are_rejected():
    for values in (
        {"pyramid_levels": 5},
        {"reference_cycle_error": float("nan")},
        {"xpos": 1},
    ):
        with pytest.raises(ValidationError):
            CorrespondenceRefinementConfig(**values)
    first, second, seeds, _ = _fixture()
    result = refine_correspondences(first, second, seeds)
    values = result.model_dump()
    with pytest.raises(ValidationError):
        RefinedCorrespondences.model_validate(values | {"cycle_consistent_count": 0})
    with pytest.raises(ValidationError):
        RefinedCorrespondences.model_validate(
            values | {"retained_source_indices": (seeds.raw_match_count,)}
        )


def test_failed_forward_solver_does_not_trigger_a_reverse_solve(monkeypatch):
    first, second, seeds, _ = _fixture()
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        return None, None, None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", failed)
    result = refine_correspondences(first, second, seeds)
    assert result.retained_source_indices == () and len(calls) == 1


def test_nonfinite_failed_forward_tracks_are_not_sent_to_reverse_solver(monkeypatch):
    first, second, seeds, _ = _fixture()
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        count = len(args[2])
        return (
            np.full((count, 1, 2), np.nan, dtype=np.float32),
            np.zeros((count, 1), dtype=np.uint8),
            None,
        )

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", failed)
    result = refine_correspondences(first, second, seeds)
    assert result.retained_source_indices == () and len(calls) == 1
