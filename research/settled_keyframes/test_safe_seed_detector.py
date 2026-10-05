import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from safe_seed_detector import safe_seeds

from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    diagnose_frame_chain,
)


@pytest.mark.parametrize("size", [224, 448])
def test_first_image_seeds_all_have_original_patch_and_window_support(size):
    gray = cv2.GaussianBlur(
        np.random.default_rng(23).integers(0, 256, (size, size), dtype=np.uint8),
        (5, 5),
        0.8,
    )
    first = np.repeat(gray[:, :, None], 3, axis=2)
    original = first.copy()
    seeds = safe_seeds(first)
    assert 32 < len(seeds) <= TemporalTrackConfig().feature_budget
    diagnosis = diagnose_frame_chain([first, first.copy()], seed_pixels=seeds)
    assert diagnosis.tracks.seed_count == len(seeds)
    assert diagnosis.tracks.surviving_counts[0] == len(seeds)
    assert not any(event.reason == "seed_roi" for event in diagnosis.rejections)
    assert np.array_equal(first, original)
    assert np.array_equal(seeds, safe_seeds(first.copy()))


def test_future_corruption_still_fails_without_renaming_failed_identities():
    first = np.random.default_rng(24).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    seeds = safe_seeds(first)
    diagnosis = diagnose_frame_chain([first, np.zeros_like(first)], seed_pixels=seeds)
    assert not diagnosis.tracks.track_ids
    rejected = [
        identity for event in diagnosis.rejections for identity in event.track_ids
    ]
    assert len(rejected) == len(set(rejected)) == len(seeds)
    assert set(rejected) == set(range(len(seeds)))
    assert np.array_equal(seeds, safe_seeds(first))


def test_untextured_image_cannot_invent_seeds():
    first = np.zeros((448, 448, 3), dtype=np.uint8)
    seeds = safe_seeds(first)
    assert seeds.shape == (0, 2)
    diagnosis = diagnose_frame_chain([first, first], seed_pixels=seeds)
    assert diagnosis.tracks.seed_count == 0 and not diagnosis.tracks.track_ids
