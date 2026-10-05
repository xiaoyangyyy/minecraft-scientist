import copy
import hashlib

import numpy as np
import pytest
from pydantic import ValidationError

pytest.importorskip("cv2", reason="optional geometry")

from test_joint_multiview_map import CAMERA, fixture_tracks, prefix

from minecraft_scientist.exploration import stable_keyframe_map as stable
from minecraft_scientist.exploration.settled_map_control import PublicControlStep


@pytest.fixture
def public_scene(monkeypatch):
    tracks, _, _ = fixture_tracks(frames=8, noise=0.02)
    first = np.zeros((224, 224, 3), dtype=np.uint8)
    calls = []

    def execute(kind, handle):
        calls.append((kind, handle))
        image = np.full_like(first, len(calls))
        return PublicControlStep(image, kind, handle, True, 1)

    def flow(first, last, settings):
        novel = int(last[0, 0, 0]) - int(first[0, 0, 0]) > 1
        return {
            "tracks": {"track_ids": list(range(40)), "seed_count": 40},
            "cells": 4,
            "stable": not novel,
            "median_flow_pixels": 4 if novel else 0,
            "percentile95_flow_pixels": 5 if novel else 0,
        }

    monkeypatch.setattr(stable, "_visible_flow", flow)
    monkeypatch.setattr(
        stable,
        "track_frame_chain",
        lambda frames, **kwargs: prefix(tracks, len(frames)),
    )
    return first, execute, calls


def test_past_freezes_before_distinct_control_with_real_synthetic_geometry(
    public_scene, monkeypatch
):
    first, execute, calls = public_scene
    fitted_at = []
    fit = stable.fit_joint_map

    def recording(*args, **kwargs):
        fitted_at.append(len(calls))
        return fit(*args, **kwargs)

    monkeypatch.setattr(stable, "fit_joint_map", recording)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13
    )
    assert result["terminal"] == "validated"
    assert len(result["attempts"]) == 1
    frozen = result["frozen"]
    assert len(frozen["maps"]) == 4
    assert frozen["index"]["frame_ordinals"] == [56, 66, 76, 86, 96, 106]
    assert fitted_at == [58] * 4
    assert result["future"]["absolute_frame"] == 116
    assert len(calls) == len(result["steps"]) == 68
    assert result["future"]["handle"] != calls[48][1]  # last acquisition MOVE
    assert result["motion"]["frame_ordinals"] == [106, 116]
    assert stable._digest(frozen) == result["frozen_sha256"]
    assert not result["navigation_accepted"] and not result["metric_scale_known"]
    assert not frozen["index"]["continuous_raw_identity"]
    assert np.array_equal(first, np.zeros_like(first))


def test_first_eligible_stability_waits_are_not_shortened(public_scene):
    first, execute, _ = public_scene
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13
    )
    assert all(
        last - first == 8
        for first, last in (record["interval"] for record in result["settles"])
    )
    assert all(len(record["stable_tail"]) == 3 for record in result["settles"])


def test_stalled_images_are_retained_not_used_as_novel_keyframes(
    public_scene, monkeypatch
):
    first, execute, calls = public_scene
    monkeypatch.setattr(
        stable,
        "_visible_flow",
        lambda *args: {
            "tracks": {"track_ids": list(range(40)), "seed_count": 40},
            "cells": 4,
            "stable": True,
            "median_flow_pixels": 0,
            "percentile95_flow_pixels": 0,
        },
    )
    result = stable.acquire_stable_keyframe_map(first, execute, CAMERA, anchor_frame=48)
    assert result["terminal"] == "map_unavailable"
    assert result["frozen"] is None and result["attempts"] == []
    assert len(result["samples"]) == 12
    assert sum(sample["selected"] for sample in result["samples"]) == 1
    assert len(calls) == len(result["steps"]) == 118
    assert len({handle for kind, handle in calls if kind == "move"}) == 4


def test_unstable_initial_prefix_exhausts_exact_wait_budget(public_scene, monkeypatch):
    first, execute, calls = public_scene
    monkeypatch.setattr(stable, "_visible_flow", lambda *args: {"stable": False})
    result = stable.acquire_stable_keyframe_map(first, execute, CAMERA, anchor_frame=48)
    assert result["terminal"] == "initial_wait_exhausted"
    assert len(calls) == 16
    assert not result["samples"] and not result["attempts"]


@pytest.mark.parametrize("bad", ["kind", "handle", "ticks", "compliance", "sensor"])
def test_bad_receipt_retains_actual_action_but_cannot_supply_evidence(
    public_scene, bad
):
    first, _, _ = public_scene
    receipt = PublicControlStep(first, "wait", None, True, 1)
    fields = (
        vars(receipt)
        | {
            "kind": {"executed_kind": "move"},
            "handle": {"executed_handle": 0},
            "ticks": {"ticks": True},
            "compliance": {"compliant": False},
            "sensor": {"frame": np.zeros((448, 448, 3), dtype=np.uint8)},
        }[bad]
    )
    result = stable.acquire_stable_keyframe_map(
        first, lambda *args: PublicControlStep(**fields), CAMERA, anchor_frame=48
    )
    assert result["terminal"] == "execution_failed"
    assert result["failure"] is not None
    assert len(result["steps"]) == 1 and result["samples"] == []


def test_future_failure_cannot_refit_or_accept_passing_subset(
    public_scene, monkeypatch
):
    first, execute, _ = public_scene
    validate = stable.validate_joint_future

    def counterexample(model, *args, **kwargs):
        evidence = validate(model, *args, **kwargs)
        if model.supported and model.bootstrap_slot == 3:
            return evidence | {"status": "inconsistent"}
        return evidence

    monkeypatch.setattr(stable, "validate_joint_future", counterexample)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13
    )
    assert result["terminal"] == "rejected" and result["motion"] is None
    assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert len(result["attempts"]) == 1
    assert len(result["future"]["candidates"]) == len(result["frozen"]["maps"])


@pytest.mark.parametrize("bad", ["duplicate", "unordered", "hash", "continuous"])
def test_sample_clock_cannot_disguise_sparse_slots_as_raw_ticks(bad):
    tracks, _, _ = fixture_tracks(frames=3)
    data = {
        "namespace": "sampled",
        "frame_ordinals": (8, 18, 28),
        "rgb_sha256": (hashlib.sha256(b"rgb").hexdigest(),) * 3,
        "tracks": tracks,
    }
    data = copy.deepcopy(data)
    if bad == "duplicate":
        data["frame_ordinals"] = (8, 8, 28)
    elif bad == "unordered":
        data["frame_ordinals"] = (18, 8, 28)
    elif bad == "hash":
        data["rgb_sha256"] = ("bogus",) * 3
    else:
        data["continuous_raw_identity"] = True
    with pytest.raises(ValidationError):
        stable.KeyframeIndex.model_validate(data)


def test_geometry_and_stability_gates_cannot_be_relaxed():
    with pytest.raises(ValidationError, match="registered geometry"):
        stable.StableKeyframePolicy(fitting={"iterations": 1})
    assert stable.StableKeyframePolicy().maximum_episode_actions == 288


def test_micro_acquisition_keeps_larger_unseen_future_control(
    public_scene, monkeypatch
):
    first, execute, calls = public_scene
    fitted_at = []
    fit = stable.fit_joint_map

    def recording(*args, **kwargs):
        fitted_at.append(len(calls))
        return fit(*args, **kwargs)

    monkeypatch.setattr(stable, "fit_joint_map", recording)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13, micro_controls=True
    )
    assert result["terminal"] == "validated"
    assert (
        result["micro_control_registration"]
        == "one-tick-acquisition-two-tick-future-v1"
    )
    assert result["policy"] == stable.StableMicroKeyframePolicy().model_dump(
        mode="json"
    )
    assert result["frozen"]["index"]["frame_ordinals"] == [56, 65, 74, 83, 92, 101]
    assert fitted_at == [53] * 4
    assert result["future"]["absolute_frame"] == 111
    assert len(calls) == len(result["steps"]) == 63
    assert calls[53:55] == [("move", result["future"]["handle"])] * 2
    assert sum(kind == "move" for kind, _ in calls[:53]) == 5
    assert result["future"]["handle"] != calls[44][1]
    assert result["motion"]["frame_ordinals"] == [101, 111]
    assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert not result["navigation_accepted"] and not result["metric_scale_known"]


@pytest.mark.parametrize("flag", [1, "true", None])
def test_micro_registration_is_explicit_boolean(public_scene, flag):
    first, execute, calls = public_scene
    with pytest.raises(ValueError, match="explicit boolean"):
        stable.acquire_stable_keyframe_map(
            first, execute, CAMERA, anchor_frame=48, micro_controls=flag
        )
    assert calls == []


def test_micro_policy_cannot_disguise_default_protocol(public_scene):
    first, execute, calls = public_scene
    with pytest.raises(ValidationError):
        stable.acquire_stable_keyframe_map(
            first,
            execute,
            CAMERA,
            anchor_frame=48,
            policy=stable.StableMicroKeyframePolicy(),
        )
    assert calls == []
    with pytest.raises(ValidationError):
        stable.StableMicroKeyframePolicy(validation_move_ticks=1)
    with pytest.raises(ValidationError, match="registered geometry"):
        stable.StableMicroKeyframePolicy(fitting={"iterations": 1})


@pytest.mark.parametrize("micro,expected_steps", [(False, 58), (True, 53)])
def test_partial_bootstrap_failure_retains_completed_models_and_actions(
    public_scene, monkeypatch, micro, expected_steps
):
    first, execute, calls = public_scene
    fit = stable.fit_joint_map

    def fail_second(*args, **kwargs):
        if kwargs["bootstrap_slot"] == 3:
            raise np.linalg.LinAlgError("injected singular bootstrap")
        return fit(*args, **kwargs)

    monkeypatch.setattr(stable, "fit_joint_map", fail_second)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13, micro_controls=micro
    )
    assert result["terminal"] == "geometry_failed"
    assert result["failure"]["stage"] == "past_fitting"
    assert result["failure"]["type"] == "LinAlgError"
    assert len(calls) == len(result["steps"]) == expected_steps
    assert sum(sample["selected"] for sample in result["samples"]) == 6
    attempt = result["attempts"][0]
    assert attempt["failed_bootstrap_slot"] == 3
    assert len(attempt["proposal"]["maps"]) == 1
    assert attempt["proposal"]["maps"][0]["bootstrap_slot"] == 2
    assert attempt["sha256"] == stable._digest(attempt["proposal"])
    assert result["frozen"] is None and result["future"] is None
    assert result["motion"] is None and not result["navigation_accepted"]


@pytest.mark.parametrize("micro,expected_steps", [(False, 18), (True, 17)])
def test_novelty_failure_retains_both_completed_settles(
    public_scene, monkeypatch, micro, expected_steps
):
    first, execute, calls = public_scene
    flow = stable._visible_flow

    def fail_novelty(first, last, settings):
        if int(last[0, 0, 0]) - int(first[0, 0, 0]) > 1:
            raise ValueError("injected endpoint correspondence failure")
        return flow(first, last, settings)

    monkeypatch.setattr(stable, "_visible_flow", fail_novelty)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, micro_controls=micro
    )
    assert result["terminal"] == "geometry_failed"
    assert result["failure"]["stage"] == "novelty"
    assert len(calls) == len(result["steps"]) == expected_steps
    assert len(result["settles"]) == 2
    assert len(result["samples"]) == 1 and result["attempts"] == []


@pytest.mark.parametrize("stage", ["past_fitting", "future_validation"])
def test_track_failure_keeps_selected_frames_and_frozen_parent(
    public_scene, monkeypatch, stage
):
    first, execute, calls = public_scene
    track = stable.track_frame_chain

    def fail_tracks(frames, **kwargs):
        if len(frames) == (6 if stage == "past_fitting" else 7):
            raise np.linalg.LinAlgError("injected tracking failure")
        return track(frames, **kwargs)

    monkeypatch.setattr(stable, "track_frame_chain", fail_tracks)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13, micro_controls=True
    )
    assert result["terminal"] == "geometry_failed"
    assert result["failure"]["stage"] == stage
    assert len(calls) == len(result["steps"]) == (53 if stage == "past_fitting" else 63)
    assert sum(sample["selected"] for sample in result["samples"]) == 6
    if stage == "past_fitting":
        assert result["attempts"] == [] and result["frozen"] is None
    else:
        assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert result["future"] is None and result["motion"] is None


@pytest.mark.parametrize("micro,expected_steps", [(False, 68), (True, 63)])
def test_future_numerical_failure_keeps_prior_candidate_and_all_controls(
    public_scene, monkeypatch, micro, expected_steps
):
    first, execute, calls = public_scene
    validate = stable.validate_joint_future

    def fail_second(model, *args, **kwargs):
        if model.bootstrap_slot == 3:
            raise np.linalg.LinAlgError("injected singular future pose")
        return validate(model, *args, **kwargs)

    monkeypatch.setattr(stable, "validate_joint_future", fail_second)
    result = stable.acquire_stable_keyframe_map(
        first, execute, CAMERA, anchor_frame=48, seed=13, micro_controls=micro
    )
    assert result["terminal"] == "geometry_failed"
    assert result["failure"]["stage"] == "future_validation"
    assert len(calls) == len(result["steps"]) == expected_steps
    assert len(result["future"]["candidates"]) == 1
    assert result["future"]["candidates"][0]["bootstrap_slot"] == 2
    assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert result["motion"] is None and not result["navigation_accepted"]
