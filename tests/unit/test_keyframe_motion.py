import numpy as np
import pytest

from minecraft_scientist.exploration import keyframe_motion as policy
from minecraft_scientist.exploration.keyframe_motion import (
    KeyframeMotionConfig,
    KeyframeSelection,
)
from minecraft_scientist.exploration.motion_stability import (
    MotionSensitivityTrial,
    StableMotionEvidence,
)
from minecraft_scientist.exploration.multiview_motion import (
    IntermediateFrameCheck,
    MultiViewMotionEvidence,
)
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks
from minecraft_scientist.exploration.two_view_motion import TwoViewMotionEvidence


def _tracks(count, frames):
    # Policy-only fixture, not image-motion accuracy evidence.
    pixels = tuple(
        (60 + index % 8 * 10, 60 + index // 8 * 10) for index in range(count)
    )
    return TemporalTracks(
        image_shape=(224, 224),
        frame_count=frames,
        seed_count=count,
        surviving_counts=(count,) * frames,
        track_ids=tuple(range(count)),
        pixels=(pixels,) * frames,
        minimum_patch_correlations=(1.0,) * count,
        chain_cycle_errors=(0.0,) * count,
    )


def _motion(count, *, supported):
    candidate = TwoViewMotionEvidence(
        supported=supported,
        reason="policy fixture, not physical evidence",
        matches=count,
        relative_rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)) if supported else None,
        unit_translation=(-1, 0, 0) if supported else None,
    )
    if not supported:
        return MultiViewMotionEvidence(
            supported=False, reason="fixture rejected", candidate=candidate
        )
    sensitivity = StableMotionEvidence(
        supported=True,
        reason="policy fixture",
        candidate=candidate,
        trials=(
            MotionSensitivityTrial(
                supported=True,
                reason="fixture",
                direction_deviation_degrees=0,
                rotation_deviation_degrees=0,
            ),
        )
        * 16,
        supported_fraction=1,
        direction_deviation_95_degrees=0,
        rotation_deviation_95_degrees=0,
        relative_rotation=candidate.relative_rotation,
        unit_translation=candidate.unit_translation,
    )
    return MultiViewMotionEvidence(
        supported=True,
        reason="policy fixture",
        candidate=candidate,
        frame_checks=(
            IntermediateFrameCheck(frame_slot=1, supported=True, reason="fixture"),
        ),
        sensitivity=sensitivity,
        relative_rotation=candidate.relative_rotation,
        unit_translation=candidate.unit_translation,
    )


@pytest.mark.parametrize("offsets", [(), (2, 2), (4, 2), (1,), (32,), (2.0,), (True,)])
def test_invalid_prefix_budgets_are_not_coerced(offsets):
    with pytest.raises(ValueError):
        KeyframeMotionConfig(candidate_offsets=offsets)


def test_fixed_offsets_always_retain_terminal_baseline():
    config = KeyframeMotionConfig()
    assert config.offsets(0) == (0,)
    assert config.offsets(1) == (1,)
    assert config.offsets(3) == (2, 3)
    assert config.offsets(15) == (2, 4, 6, 8, 12, 15)
    for invalid in (-1, 32, True):
        with pytest.raises(ValueError):
            config.offsets(invalid)


def test_each_fit_sees_only_its_prefix_with_fixed_seed_and_first_support_latched(
    monkeypatch,
):
    observed = []

    def tracker(frames, **kwargs):
        observed.append([int(frame[0, 0, 0]) for frame in frames])
        return _tracks(64, len(frames))

    def estimator(tracks, camera, *, config, seed):
        assert seed == 7 and config == KeyframeMotionConfig().multiview
        return _motion(64, supported=tracks.frame_count >= 5)

    monkeypatch.setattr(policy, "track_frame_chain", tracker)
    monkeypatch.setattr(policy, "estimate_multiview_motion", estimator)
    frames = [np.full((32, 32, 3), index, dtype=np.uint8) for index in range(8)]
    before = [frame.copy() for frame in frames]
    result = policy.infer_keyframe_prefixes(frames, np.eye(3), seed=7)
    assert observed == [list(range(length)) for length in (3, 5, 7, 8)]
    assert result.selected_offset == 4
    assert result.selected().motion.supported
    assert result.attempts[-1].end_offset == 7 and result.attempts[-1].motion.supported
    for first, second in zip(frames, before, strict=True):
        np.testing.assert_array_equal(first, second)


def test_later_rejection_cannot_overwrite_selected_local_pose(monkeypatch):
    monkeypatch.setattr(
        policy, "track_frame_chain", lambda frames, **kw: _tracks(64, len(frames))
    )
    monkeypatch.setattr(
        policy,
        "estimate_multiview_motion",
        lambda tracks, *args, **kw: _motion(64, supported=tracks.frame_count == 3),
    )
    result = policy.infer_keyframe_prefixes([np.zeros((32, 32, 3))] * 6, np.eye(3))
    assert result.selected_offset == 2
    assert not result.attempts[-1].motion.supported
    assert result.selected().motion.supported
    with pytest.raises(ValueError, match="first supported"):
        KeyframeSelection.model_validate(result.model_dump() | {"selected_offset": 5})


def test_all_rejections_and_zero_length_control_remain_explicit(monkeypatch):
    calls = []

    def tracker(frames, **kwargs):
        calls.append(len(frames))
        return _tracks(0, len(frames))

    monkeypatch.setattr(policy, "track_frame_chain", tracker)
    monkeypatch.setattr(
        policy,
        "estimate_multiview_motion",
        lambda *args, **kw: _motion(0, supported=False),
    )
    result = policy.infer_keyframe_prefixes([np.zeros((32, 32, 3))], np.eye(3))
    assert calls == [2] and result.selected() is None
    assert result.terminal_offset == 0 and result.attempts[0].end_offset == 0
    with pytest.raises(ValueError, match="terminal baseline"):
        KeyframeSelection.model_validate(result.model_dump() | {"attempts": ()})


def test_invalid_inputs_stop_before_tracking(monkeypatch):
    monkeypatch.setattr(
        policy,
        "track_frame_chain",
        lambda *args, **kw: pytest.fail("invalid budget entered tracking"),
    )
    for seed in (-1, True, 2**31 - 16):
        with pytest.raises(ValueError, match="seed"):
            policy.infer_keyframe_prefixes(
                [np.zeros((32, 32, 3))], np.eye(3), seed=seed
            )
    for count in (0, 33):
        with pytest.raises(ValueError, match="bounded"):
            policy.infer_keyframe_prefixes([np.zeros((32, 32, 3))] * count, np.eye(3))


def test_real_identity_images_reject_every_candidate():
    pytest.importorskip("cv2", reason="requires optional geometry extra")
    image = np.random.default_rng(73).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    camera = np.array(((90, 0, 111.5), (0, 160, 111.5), (0, 0, 1)))
    result = policy.infer_keyframe_prefixes([image] * 8, camera)
    assert result.selected_offset is None and len(result.attempts) == 4
    assert all(not attempt.motion.supported for attempt in result.attempts)
