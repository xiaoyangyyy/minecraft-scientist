import numpy as np
import pytest
from pydantic import ValidationError
from test_keyframe_motion import _motion
from test_online_landmark_probe import tracks

from minecraft_scientist.exploration import second_leg_shadow as shadow
from minecraft_scientist.exploration.keyframe_motion import KeyframeAttempt
from minecraft_scientist.exploration.pose_direction_search import (
    DirectionSearchEvidence,
    MapValidation,
)
from minecraft_scientist.exploration.visibility_acquisition import PublicProbeStep


def images(count=5):
    return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(count)]


@pytest.fixture
def policy_fitter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        shadow, "track_frame_chain", lambda frames, config: tracks(len(frames) - 1)
    )

    def fit(current, retained, camera, *, seed):
        calls.append(("fit", current.frame_count - 1))
        filtered = shadow.retain_tracks(current, retained)
        motion = _motion(len(filtered.track_ids), supported=True).model_copy(
            update={
                "training_track_ids": filtered.track_ids[:36],
                "validation_track_ids": filtered.track_ids[36:48],
            }
        )
        return KeyframeAttempt(
            end_offset=current.frame_count - 1, tracks=filtered, motion=motion
        )

    def search(candidate, camera, *, seed, config):
        calls.append(("search", None if candidate is None else candidate.end_offset))
        # Policy-only fixture, not a successful physical/reference replay.
        return DirectionSearchEvidence(
            config=config,
            ambiguity_detected=False,
            reason="policy fixture",
            training_track_ids=()
            if candidate is None
            else candidate.motion.training_track_ids,
            validation_track_ids=()
            if candidate is None
            else candidate.motion.validation_track_ids,
            reference=None
            if candidate is None
            else MapValidation(
                supported=False,
                reason="policy fixture",
                training_tracks=36,
                validation_tracks=12,
            ),
        )

    monkeypatch.setattr(shadow, "fit_online_landmarks", fit)
    monkeypatch.setattr(shadow, "inspect_direction_search", search)
    return calls


def test_fit_and_complete_search_precede_every_new_control_and_never_replace_map(
    policy_fitter,
):
    def execute(handle):
        policy_fitter.append(("execute", handle))
        return PublicProbeStep(images(1)[0], handle, True)

    result = shadow.acquire_second_leg_shadow(
        images(), execute, (0, 1), np.eye(3), seed=1002
    )
    assert (
        policy_fitter
        == [("fit", 2), ("fit", 3), ("fit", 4), ("search", 2)] + [("execute", 2)] * 4
    )
    assert result["terminal"] == "completed" and result["failure"] is None
    assert result["freeze"]["selection"]["selected_offset"] == 2
    assert result["plan"]["action_start_frame"] == 4
    assert result["plan"]["target_frames"] == [6, 8]
    assert (
        result["freeze_sha256"]
        == shadow.SecondLegFreeze.model_validate(result["freeze"]).digest()
    )
    assert len(result["prefix_tracks"]) == len(result["steps"]) == 4


def test_no_candidate_is_retained_without_fabricating_map_validation(
    policy_fitter, monkeypatch
):
    def unsupported(current, retained, camera, *, seed):
        return KeyframeAttempt(
            end_offset=current.frame_count - 1,
            tracks=current,
            motion=_motion(len(current.track_ids), supported=False),
        )

    monkeypatch.setattr(shadow, "fit_online_landmarks", unsupported)
    result = shadow.acquire_second_leg_shadow(
        images(),
        lambda handle: PublicProbeStep(images(1)[0], handle, True),
        (0, 1),
        np.eye(3),
        seed=1002,
    )
    assert result["terminal"] == "completed"
    assert result["freeze"]["selection"]["selected_offset"] is None
    assert not result["plan"]["available"] and not result["plan"]["target_frames"]
    assert len(result["freeze"]["selection"]["attempts"]) == 3


def test_map_support_loss_aborts_even_with_hundreds_of_generic_survivors(
    policy_fitter, monkeypatch
):
    monkeypatch.setattr(
        shadow,
        "track_frame_chain",
        lambda frames, config: tracks(
            len(frames) - 1,
            tuple(range(28, 400)) if len(frames) > 5 else tuple(range(400)),
        ),
    )
    result = shadow.acquire_second_leg_shadow(
        images(),
        lambda handle: PublicProbeStep(images(1)[0], handle, True),
        (0, 1),
        np.eye(3),
        seed=1002,
    )
    assert result["terminal"] == "support_aborted" and result["failure"] is None
    assert len(result["steps"]) == 1 and len(result["retained_ids"][0]) == 372
    assert not result["plan"]["available"] and not result["plan"]["target_frames"]


def test_lost_identity_never_returns_during_shadow_or_enters_fitted_map(
    policy_fitter, monkeypatch
):
    monkeypatch.setattr(
        shadow,
        "track_frame_chain",
        lambda frames, config: tracks(
            len(frames) - 1,
            tuple(range(1, 400)) if len(frames) == 2 else tuple(range(400)),
        ),
    )
    result = shadow.acquire_second_leg_shadow(
        images(),
        lambda handle: PublicProbeStep(images(1)[0], handle, True),
        (0, 1),
        np.eye(3),
        seed=1002,
    )
    assert result["terminal"] == "completed"
    assert 0 not in result["freeze"]["selection"]["attempts"][0]["tracks"]["track_ids"]
    assert all(0 not in retained for retained in result["retained_ids"])


@pytest.mark.parametrize(
    "receipt",
    (
        PublicProbeStep(images(1)[0], 1, True),
        PublicProbeStep(images(1)[0], 2, False),
        PublicProbeStep(images(1)[0], 2, True, ticks=2),
        PublicProbeStep(images(1)[0], 2, 1),
        {"frame": images(1)[0], "yaw": 0},
    ),
)
def test_failed_receipt_is_retained_but_never_tracked_or_retried(
    policy_fitter, receipt
):
    result = shadow.acquire_second_leg_shadow(
        images(), lambda handle: receipt, (0, 1), np.eye(3), seed=1002
    )
    assert result["terminal"] == "control_failed" and result["failure"] is not None
    assert len(result["steps"]) == 1 and result["prefix_tracks"] == []
    assert not result["plan"]["available"]


def test_future_pixels_cannot_change_frozen_prefix(policy_fitter, monkeypatch):
    def changed(frames, config):
        current = tracks(len(frames) - 1)
        if len(frames) > 5:
            pixels = np.asarray(current.pixels).copy()
            pixels[0, 0, 0] += 0.1
            current = current.model_copy(
                update={"pixels": tuple(tuple(map(tuple, frame)) for frame in pixels)}
            )
        return current

    monkeypatch.setattr(shadow, "track_frame_chain", changed)
    result = shadow.acquire_second_leg_shadow(
        images(),
        lambda handle: PublicProbeStep(images(1)[0], handle, True),
        (0, 1),
        np.eye(3),
        seed=1002,
    )
    assert (
        result["terminal"] == "tracking_failed"
        and "prefix" in result["failure"]["message"]
    )
    assert len(result["steps"]) == 1 and result["prefix_tracks"] == []


@pytest.mark.parametrize(
    "handles,count",
    (((0, 0), 4), ((0, 1), 2), ((0, 4), 4), ((True, 1), 4), ((0, 1), True)),
)
def test_invalid_public_handles_fail_before_any_control(handles, count):
    with pytest.raises(ValueError):
        shadow.choose_shadow_handle(handles, count)


def test_policy_seed_schedule_and_handle_rule_are_fixed():
    policy = shadow.SecondLegShadowPolicy()
    assert [policy.seed(number) for number in range(8)] == list(range(1002, 1038, 5))
    assert [
        shadow.choose_shadow_handle(pair, 4)
        for pair in ((0, 1), (0, 2), (0, 3), (1, 0))
    ] == [2, 1, 1, 2]
    for invalid in (-1, 8, True):
        with pytest.raises(ValueError):
            policy.seed(invalid)
    for fields in ({"ticks": 3}, {"seed_base": 0}, {"pose": 0}):
        with pytest.raises(ValidationError):
            shadow.SecondLegShadowPolicy(**fields)


def test_actual_synthetic_geometry_freezes_first_supported_prefix(monkeypatch):
    pytest.importorskip("cv2", reason="actual geometry needs optional extra")
    from test_multiview_motion import CAMERA, _tracks

    full = _tracks(frame_count=5)

    def prefix(frames, config):
        return full.model_copy(
            update={
                "frame_count": len(frames),
                "pixels": full.pixels[: len(frames)],
                "surviving_counts": (200,) * len(frames),
            }
        )

    monkeypatch.setattr(shadow, "track_frame_chain", prefix)
    frozen = shadow.prepare_second_leg(images(), CAMERA, seed=7)
    assert frozen.selection.selected() is not None
    assert frozen.search.reference_matches_original_checks
    assert len(frozen.search.trials) == 16
    assert (
        frozen.digest() == shadow.prepare_second_leg(images(), CAMERA, seed=7).digest()
    )
