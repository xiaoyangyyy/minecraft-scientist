import numpy as np
import pytest

from minecraft_scientist.exploration.perception import RGBEdgeSceneEncoder
from minecraft_scientist.exploration.resolution import ResolutionNormalizedSceneEncoder


def test_integer_resolution_change_keeps_reference_feature_scale():
    image = np.random.default_rng(10).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    enlarged = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)
    original = enlarged.copy()
    base = RGBEdgeSceneEncoder(grid_size=4)
    encoder = ResolutionNormalizedSceneEncoder(base, size=(64, 64))
    assert encoder.dimensions == base.dimensions
    np.testing.assert_array_equal(encoder.normalize(enlarged), image)
    assert encoder.encode(enlarged) == base.encode(image)
    np.testing.assert_array_equal(enlarged, original)


def test_noninteger_resolution_has_explicit_pillow_boundary():
    pytest.importorskip("PIL")
    encoder = ResolutionNormalizedSceneEncoder(
        RGBEdgeSceneEncoder(grid_size=4), size=(64, 48)
    )
    assert encoder.normalize(np.zeros((100, 90, 3), dtype=np.uint8)).shape == (
        48,
        64,
        3,
    )


@pytest.mark.parametrize("size", [(0, 64), (-1, 64), (True, 64), (64.0, 64), (64,)])
def test_invalid_reference_resolution_is_rejected(size):
    with pytest.raises(ValueError):
        ResolutionNormalizedSceneEncoder(RGBEdgeSceneEncoder(), size=size)


def test_invalid_pixels_are_rejected_and_equal_size_result_is_not_an_alias():
    encoder = ResolutionNormalizedSceneEncoder(
        RGBEdgeSceneEncoder(grid_size=4), size=(64, 64)
    )
    for frame in (
        np.zeros((64, 64)),
        np.full((64, 64, 3), np.nan),
        np.zeros((0, 64, 3)),
    ):
        with pytest.raises(ValueError):
            encoder.normalize(frame)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    normalized = encoder.normalize(frame)
    normalized[0, 0] = 255
    assert not frame.any()
