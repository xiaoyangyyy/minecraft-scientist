import numpy as np
import pytest
from pydantic import ValidationError

cv2 = pytest.importorskip("cv2", reason="requires optional geometry extra")

from minecraft_scientist.exploration.temporal_tracks import (
    TemporalTrackConfig,
    TemporalTracks,
    track_frame_chain,
)


def _images(count=5):
    rng = np.random.default_rng(113)
    image = cv2.GaussianBlur(
        rng.integers(0, 256, (224, 224), dtype=np.uint8), (5, 5), 0.8
    )
    rgb = np.repeat(image[:, :, None], 3, axis=2)
    shift = np.array((1.2, -0.3))
    frames = [
        cv2.warpAffine(
            rgb,
            np.array(((1.0, 0, index * shift[0]), (0, 1, index * shift[1]))),
            (224, 224),
        )
        for index in range(count)
    ]
    return frames, shift


def test_temporal_chain_preserves_identities_and_accumulated_pixels():
    frames, shift = _images()
    originals = [frame.copy() for frame in frames]
    tracks = track_frame_chain(frames)
    assert len(tracks.track_ids) > 100 and tracks.frame_count == 5
    pixels = np.array(tracks.pixels)
    for index in range(1, 5):
        np.testing.assert_allclose(
            pixels[index] - pixels[0],
            np.broadcast_to(index * shift, pixels[0].shape),
            atol=0.3,
        )
    assert np.max(tracks.chain_cycle_errors) <= 1
    assert all(
        left >= right
        for left, right in zip(tracks.surviving_counts, tracks.surviving_counts[1:])
    )
    assert len(set(tracks.track_ids)) == len(tracks.track_ids)
    pairs = tracks.endpoint_correspondences()
    assert pairs.points0 == tracks.pixels[0] and pairs.points1 == tracks.pixels[-1]
    for first, second in zip(frames, originals, strict=True):
        np.testing.assert_array_equal(first, second)


def test_temporal_pyramids_exclude_outside_hud_pixels_before_filtering():
    frames, _ = _images(3)
    changed = [
        np.random.default_rng(index).integers(0, 256, frames[0].shape, dtype=np.uint8)
        for index in range(3)
    ]
    for original, altered in zip(frames, changed, strict=True):
        altered[44:170, 40:183] = original[44:170, 40:183]
    assert track_frame_chain(frames) == track_frame_chain(changed)


def test_failed_middle_frame_cannot_be_skipped_or_reseed_an_old_identity():
    frames, _ = _images(4)
    frames[2] = np.zeros_like(frames[2])
    tracks = track_frame_chain(frames)
    assert tracks.track_ids == () and tracks.pixels == ((), (), (), ())
    assert tracks.surviving_counts[0] > 0 and tracks.surviving_counts[-2:] == (0, 0)


def test_explicit_seed_order_is_preserved_in_track_provenance():
    frames, _ = _images(3)
    seeds = ((80.0, 80.0), (110.0, 110.0), (10.0, 10.0), (140.0, 130.0))
    tracks = track_frame_chain(frames, seed_pixels=seeds)
    assert 2 not in tracks.track_ids
    assert tuple(seeds[index] for index in tracks.track_ids) == tracks.pixels[0]
    assert len(tracks.track_ids) >= 2


def test_empty_or_flat_images_do_not_create_tracks():
    frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * 3
    assert track_frame_chain(frames).seed_count == 0
    assert track_frame_chain(frames, seed_pixels=()).pixels == ((), (), ())


def test_chain_is_deterministic_and_bounded():
    frames, _ = _images(3)
    config = TemporalTrackConfig(feature_budget=64)
    first = track_frame_chain(frames, config=config)
    assert first == track_frame_chain(frames, config=config)
    assert first.seed_count <= 64
    with pytest.raises(ValueError, match="bounded"):
        track_frame_chain(frames * 11)
    with pytest.raises(ValueError, match="identical"):
        track_frame_chain((frames[0], frames[1][:100]))
    with pytest.raises(ValueError, match="finite"):
        track_frame_chain(frames, seed_pixels=((float("nan"), 1),))
    with pytest.raises(ValueError, match="budget"):
        track_frame_chain(frames, config=config, seed_pixels=((90.0, 90.0),) * 65)


def test_schemas_reject_privileged_nonfinite_and_inconsistent_track_data():
    with pytest.raises(ValidationError):
        TemporalTrackConfig(yaw=1)
    with pytest.raises(ValidationError):
        TemporalTrackConfig(reference_cycle_error=float("nan"))
    frames, _ = _images(3)
    tracks = track_frame_chain(frames).model_dump()
    for replacement in (
        {"surviving_counts": (1, 2, 1)},
        {"track_ids": (0, 0)},
        {"pixels": ((),)},
        {"chain_cycle_errors": (float("nan"),)},
        {"xpos": 1},
    ):
        with pytest.raises(ValidationError):
            TemporalTracks.model_validate(tracks | replacement)


def test_invalid_forward_output_is_never_fed_into_reverse_tracking(monkeypatch):
    frames, _ = _images(3)
    calls = []

    def invalid(*args, **kwargs):
        calls.append(1)
        count = len(args[2])
        return (
            np.full((count, 1, 2), np.nan, dtype=np.float32),
            np.zeros((count, 1), dtype=np.uint8),
            None,
        )

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", invalid)
    tracks = track_frame_chain(frames)
    assert len(calls) == 1 and tracks.track_ids == ()
