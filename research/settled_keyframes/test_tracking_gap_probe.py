import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tracking_gap_probe import compare


def scene(count=5):
    image = cv2.GaussianBlur(
        np.random.default_rng(113).integers(0, 256, (224, 224), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    return [
        cv2.warpAffine(
            rgb, np.array(((1.0, 0, index * 1.2), (0, 1, -index * 0.3))), (224, 224)
        )
        for index in range(count)
    ]


def test_direct_and_continuous_share_anchor_but_not_observation_counts():
    frames = scene()
    old = [frame.copy() for frame in frames]
    result = compare(frames)
    assert result["direct"]["observations"] == 2
    assert result["continuous"]["observations"] == 5
    assert result["direct"]["seed_count"] == result["continuous"]["seed_count"]
    assert result["shared_anchor_id_count"] > 100
    assert result["shared_endpoint_disagreement_95_pixels"] < 0.5
    assert not result["is_ground_truth_error"]
    assert all(np.array_equal(a, b) for a, b in zip(frames, old, strict=True))


def test_continuous_tracking_cannot_skip_a_failed_middle_image():
    frames = scene()
    frames[2] = np.zeros_like(frames[2])
    result = compare(frames)
    assert result["direct"]["surviving_count"] > 100
    assert result["continuous"]["surviving_count"] == 0
    assert result["shared_endpoint_disagreement_95_pixels"] is None


@pytest.mark.parametrize("count", [1, 129])
def test_complete_raw_interval_is_bounded_not_subsampled(count):
    with pytest.raises(ValueError, match="complete bounded"):
        compare([np.zeros((224, 224, 3), dtype=np.uint8)] * count)
