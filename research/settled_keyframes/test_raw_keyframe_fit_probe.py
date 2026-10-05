import numpy as np
import pytest

pytest.importorskip("cv2")

from raw_keyframe_fit_probe import sample_raw_tracks
from test_tracking_gap_probe import scene
from tracking_gap_probe import compare

from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


def test_sampling_preserves_original_identity_and_every_raw_gate():
    raw = TemporalTracks.model_validate(compare(scene(7))["continuous"]["tracks"])
    sampled = sample_raw_tracks(raw, [0, 2, 6])
    assert sampled.track_ids == raw.track_ids
    assert sampled.seed_count == raw.seed_count
    assert sampled.pixels == (raw.pixels[0], raw.pixels[2], raw.pixels[6])
    assert sampled.surviving_counts == tuple(
        raw.surviving_counts[slot] for slot in [0, 2, 6]
    )
    assert sampled.minimum_patch_correlations == raw.minimum_patch_correlations
    assert sampled.chain_cycle_errors == raw.chain_cycle_errors


def test_failed_raw_middle_cannot_be_rescued_by_sampling():
    frames = scene(7)
    frames[3] = np.zeros_like(frames[3])
    raw = TemporalTracks.model_validate(compare(frames)["continuous"]["tracks"])
    assert sample_raw_tracks(raw, [0, 2, 6]).track_ids == ()


@pytest.mark.parametrize(
    "slots", [[0, 1], [0, 1, 1, 6], [0, 3, 2, 6], [0, 2, 5], [1, 2, 6], [0, True, 6]]
)
def test_projection_cannot_change_clock_or_drop_prefix_endpoints(slots):
    raw = TemporalTracks.model_validate(compare(scene(7))["continuous"]["tracks"])
    with pytest.raises(ValueError, match="full-prefix"):
        sample_raw_tracks(raw, slots)
