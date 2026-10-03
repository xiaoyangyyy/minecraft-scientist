import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from minecraft_scientist.exploration import online_landmark_probe as online
from minecraft_scientist.exploration import visibility_acquisition as visibility
from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.temporal_tracks import TemporalTracks


def tracks(tick, ids=tuple(range(400))):
    seeds = np.array(
        [
            (65 + i % 8 * 12, 148 + i // 8 * 2)
            if i < 48
            else (55 + i % 20 * 5, 55 + i // 20 * 4)
            for i in range(400)
        ]
    )
    pixels = []
    for frame in range(tick + 1):
        moved = seeds.copy()
        moved[:48, 1] += frame * 4
        pixels.append(tuple(map(tuple, moved[list(ids)])))
    return TemporalTracks(
        image_shape=(224, 224),
        frame_count=tick + 1,
        seed_count=400,
        surviving_counts=(400,) * tick + (len(ids),),
        track_ids=ids,
        pixels=tuple(pixels),
        minimum_patch_correlations=(1,) * len(ids),
        chain_cycle_errors=(0,) * len(ids),
    )


def test_public_calibration_uses_temporal_track_edge_coordinates(monkeypatch):
    captured = []
    monkeypatch.setattr(
        online, "_opencv", lambda: SimpleNamespace(__version__="test-only")
    )
    monkeypatch.setattr(
        online,
        "_features",
        lambda image, config, cv2: (
            np.array(((20.0, 30.0), (40.0, 50.0))),
            np.ones((2, 32), dtype=np.uint8),
        ),
    )

    def calibrate(features, width, config, cv2):
        captured.extend(features)
        return 90.0, 24, 0.01

    monkeypatch.setattr(online, "_calibrate_focal", calibrate)
    result = online.calibrate_reference_camera(
        [np.zeros((224, 224, 3), dtype=np.uint8)] * 12, (640, 360)
    )
    np.testing.assert_allclose(captured[0][0], ((20.5, 30.5), (40.5, 50.5)))
    np.testing.assert_allclose(
        result["camera_intrinsics"], ((90, 0, 112), (0, 160, 112), (0, 0, 1))
    )
    assert result["camera_resize"]["principal_point"] == "pixel_edge_center"


def candidate(tick=2, *, current=None, supported=True):
    from test_keyframe_motion import _motion

    current = current or tracks(tick)
    motion = _motion(len(current.track_ids), supported=supported)
    if supported:
        motion = motion.model_copy(
            update={
                "training_track_ids": tuple(range(36)),
                "validation_track_ids": tuple(range(36, 48)),
            }
        )
    return KeyframeAttempt(end_offset=tick, tracks=current, motion=motion)


def controller():
    return visibility.VisibilityController(
        (18, 9),
        policy=visibility.VisibilityPolicy(
            landmark_protection="first_supported_map_v1"
        ),
    )


def test_map_specific_edge_warning_switches_while_generic_points_are_safe():
    control = controller()
    control.observe(tracks(1))
    fitted = candidate()
    decision = control.observe(tracks(2), landmark=fitted)
    assert decision.projected_edge_fraction < control.policy.projected_edge_trigger
    assert decision.retention_fraction == 1
    assert decision.landmarks.projected_edge_fraction == 1
    assert decision.instruction == "switch" and decision.endpoint_tick == 2
    assert decision.landmarks.training_ids == tuple(range(36))
    assert decision.landmarks.validation_ids == tuple(range(36, 48))
    assert (
        decision.landmarks.candidate_sha256
        == hashlib.sha256(fitted.model_dump_json().encode("utf-8")).hexdigest()
    )


def test_map_support_loss_aborts_despite_many_generic_survivors():
    control = controller()
    control.observe(tracks(1))
    control.observe(tracks(2), landmark=candidate())
    decision = control.observe(tracks(3, tuple(range(28, 400))))
    assert len(decision.retained_ids) == 372 and decision.phase_retention_fraction > 0.9
    assert len(decision.landmarks.training_ids) == 8
    assert decision.instruction == "abort" and not decision.holdout_ticks
    with pytest.raises(ValueError, match="terminal"):
        control.observe(tracks(4))


def test_lost_protected_id_cannot_return_or_move_to_other_partition():
    control = controller()
    control.observe(tracks(1))
    switched = control.observe(tracks(2), landmark=candidate())
    control.observe(tracks(3, tuple(range(1, 400))))
    decision = control.observe(tracks(4))
    assert decision.landmarks.candidate_sha256 == switched.landmarks.candidate_sha256
    assert decision.landmarks.training_ids == tuple(range(1, 36))
    assert decision.landmarks.validation_ids == tuple(range(36, 48))
    assert decision.landmarks.initial_training_count == 36
    assert decision.instruction == "finish" and decision.holdout_ticks == (3, 4)


@pytest.mark.parametrize(
    "change", ["unsupported", "stale", "pixels", "invented", "empty"]
)
def test_invalid_map_attachment_cannot_mutate_controller(change):
    control = controller()
    first = control.observe(tracks(1))
    fitted = candidate()
    if change == "unsupported":
        fitted = candidate(supported=False)
    elif change == "stale":
        fitted = candidate(3)
    elif change == "pixels":
        pixels = np.asarray(fitted.tracks.pixels) + 0.01
        fitted = fitted.model_copy(
            update={
                "tracks": fitted.tracks.model_copy(
                    update={
                        "pixels": tuple(tuple(map(tuple, frame)) for frame in pixels)
                    }
                )
            }
        )
    elif change == "invented":
        fitted = fitted.model_copy(
            update={
                "motion": fitted.motion.model_copy(
                    update={"training_track_ids": tuple(range(500, 536))}
                )
            }
        )
    else:
        fitted = fitted.model_copy(
            update={
                "motion": fitted.motion.model_copy(update={"training_track_ids": ()})
            }
        )
    with pytest.raises(ValueError):
        control.observe(tracks(2), landmark=fitted)
    assert control.decisions == (first,) and not control.has_landmark
    assert control.observe(tracks(2), landmark=candidate()).instruction == "switch"


def test_frozen_candidate_cannot_be_replaced_by_another_supported_map():
    control = controller()
    control.observe(tracks(1))
    first = control.observe(tracks(2), landmark=candidate())
    with pytest.raises(ValueError, match="frozen once"):
        control.observe(tracks(3), landmark=candidate(3))
    assert control.decisions[-1] == first
    assert (
        control.observe(tracks(3)).landmarks.candidate_sha256
        == first.landmarks.candidate_sha256
    )


@pytest.mark.parametrize(
    "update",
    [
        {"training_ids": (1, 1)},
        {"validation_ids": (0,)},
        {"initial_training_count": 12},
        {"candidate_sha256": "fake"},
        {"fit_tick": 1},
        {"projected_edge_fraction": float("nan")},
    ],
)
def test_landmark_visibility_schema_rejects_forged_partitions(update):
    control = controller()
    control.observe(tracks(1))
    status = control.observe(tracks(2), landmark=candidate()).landmarks
    with pytest.raises(ValidationError):
        visibility.LandmarkVisibility.model_validate(status.model_dump() | update)


def test_online_fitter_excludes_ids_lost_in_previous_prefix(monkeypatch):
    from test_keyframe_motion import _motion

    seen = []

    def fit(current, camera, *, config, seed):
        seen.append(current)
        assert current.track_ids == tuple(range(1, 400)) and seed == 17
        return _motion(len(current.track_ids), supported=False)

    monkeypatch.setattr(online, "estimate_multiview_motion", fit)
    result = online.fit_online_landmarks(
        tracks(2), tuple(range(1, 400)), np.eye(3), seed=17
    )
    assert result.end_offset == 2 and result.tracks == seen[0]
    assert not result.motion.supported


def test_probe_fits_current_prefix_once_and_freezes_before_next_control(monkeypatch):
    actions, fits = [], []
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    monkeypatch.setattr(
        visibility, "track_frame_chain", lambda frames, config: tracks(len(frames) - 1)
    )

    def execute(handle):
        actions.append(handle)
        return visibility.PublicProbeStep(image, handle, True)

    def observe(current, retained):
        assert actions == [18, 18] and current.frame_count == 3
        assert retained == tuple(range(400))
        fits.append(current)
        return candidate(current=current)

    result = visibility.acquire_visibility_probe(
        image, execute, (18, 9), policy=controller().policy, landmark_observer=observe
    )
    assert actions == [18, 18, 9, 9] and len(fits) == 1
    assert result.terminal == "finished" and len(result.landmark_attempts) == 1
    assert result.landmark_attempts[0].tracks == fits[0]


def test_observer_requires_explicit_policy_before_any_control():
    with pytest.raises(ValueError, match="explicit protection"):
        visibility.acquire_visibility_probe(
            np.zeros((224, 224, 3), dtype=np.uint8),
            lambda handle: pytest.fail("unconfigured observer executed a control"),
            (0, 1),
            landmark_observer=lambda *args: candidate(),
        )


def test_public_calibration_rejects_wrong_reference_count_before_opencv():
    with pytest.raises(ValueError, match="twelve"):
        online.calibrate_reference_camera([], (640, 360))


def test_actual_online_geometry_retains_disjoint_map_partitions():
    pytest.importorskip("cv2")
    from test_multiview_motion import CAMERA, _tracks

    result = online.fit_online_landmarks(_tracks(), None, CAMERA)
    assert result.motion.supported
    assert set(result.motion.training_track_ids).isdisjoint(
        result.motion.validation_track_ids
    )
    assert (
        len(result.motion.training_track_ids) + len(result.motion.validation_track_ids)
        >= 32
    )
