import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.stationary_context import (
    StationaryContextConfig,
    StationarySeeds,
    stationary_background_seeds,
)


def _scene():
    rng = np.random.default_rng(211)
    gray = cv2.GaussianBlur(
        rng.integers(0, 256, (224, 224), dtype=np.uint8), (5, 5), 0.8
    )
    return np.repeat(gray[:, :, None], 3, axis=2)


def test_static_context_preserves_last_real_pixels_and_is_deterministic():
    image = _scene()
    frames = [image.copy() for _ in range(3)]
    result = stationary_background_seeds(frames)
    assert len(result.pixels) > 100 and result.context_frames == 3
    assert max(result.maximum_displacements) < 0.01
    assert result == stationary_background_seeds(frames)
    assert all(value == 0 for value in result.patch_range_mean)
    assert len(set(result.seed_ids)) == len(result.seed_ids)
    for frame in frames:
        np.testing.assert_array_equal(frame, image)


def test_moving_foreground_cannot_be_promoted_as_stationary_background():
    image = _scene()
    frames = [image.copy() for _ in range(3)]
    for index, frame in enumerate(frames):
        frame[75:125, 65 + index * 15 : 100 + index * 15] = 255
    result = stationary_background_seeds(frames)
    assert len(result.pixels) > 50
    assert all(not (62 <= x <= 133 and 72 <= y <= 128) for x, y in result.pixels)


def test_whole_view_drift_is_not_assumed_to_be_stationary_from_wait_intent():
    image = _scene()
    frames = [
        cv2.warpAffine(image, np.array(((1, 0, index * 3.0), (0, 1, 0))), (224, 224))
        for index in range(3)
    ]
    result = stationary_background_seeds(frames)
    assert result.pixels == ()


def test_outside_roi_cannot_enter_temporal_filtering_or_detection():
    image = _scene()
    altered = [
        np.random.default_rng(index).integers(0, 256, image.shape, dtype=np.uint8)
        for index in range(3)
    ]
    for frame in altered:
        frame[44:170, 40:183] = image[44:170, 40:183]
    assert stationary_background_seeds([image] * 3) == stationary_background_seeds(
        altered
    )


def test_flat_context_and_invalid_input_fail_closed():
    flat = np.zeros((224, 224, 3), dtype=np.uint8)
    assert stationary_background_seeds([flat] * 3).pixels == ()
    for frames in ([flat] * 2, [flat, flat, flat[:100]], [flat.astype(float)] * 3):
        with pytest.raises(ValueError):
            stationary_background_seeds(frames)
    for changed in (
        {"yaw": 0},
        {"maximum_patch_range": float("nan")},
        {"context_frames": True},
        {"temporal": {"maximum_frames": 3}},
    ):
        with pytest.raises(ValidationError):
            StationaryContextConfig.model_validate(changed)


def test_background_provenance_schema_rejects_misaligned_or_executable_claims():
    values = stationary_background_seeds([_scene()] * 3).model_dump()
    for changed in (
        {"seed_ids": (0, 0)},
        {"context_consistent_count": 0},
        {"maximum_displacements": ()},
        {"xpos": 0},
    ):
        with pytest.raises(ValidationError):
            StationarySeeds.model_validate(values | changed)
