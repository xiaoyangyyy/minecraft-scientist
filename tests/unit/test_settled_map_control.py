import copy
import json

import numpy as np
import pytest
from pydantic import ValidationError
from test_active_causal_map import (
    runner as runner,  # noqa: PLC0414 -- pytest fixture re-export
)
from test_causal_map_window import CAMERA, tracks
from test_causal_map_window import (
    geometry as geometry,  # noqa: PLC0414 -- pytest fixture re-export
)

from minecraft_scientist.exploration import active_causal_map as active
from minecraft_scientist.exploration import causal_map_window as causal
from minecraft_scientist.exploration import settled_map_control as settled


@pytest.fixture
def session(runner, monkeypatch):
    image, acquire, _, trace = runner
    acquired = active.acquire_causal_map(image, acquire, CAMERA, anchor_frame=48)
    frames = [image.copy() for _ in range(acquired["evidence_frames"])]
    monkeypatch.setattr(
        settled, "track_frame_chain", lambda images, **kw: tracks(len(images))
    )
    original = causal.validate_joint_future
    shift = np.zeros(3)
    requests = []

    def future(*args, **kwargs):
        value = original(*args, **kwargs)
        value["poses"][0]["translation"] = (
            np.array(value["poses"][0]["translation"]) + shift
        ).tolist()
        return value

    def execute(kind, handle):
        requests.append((kind, handle))
        if kind == "move":
            shift[:] += ((0.1, 0, 0), (0, 0, 0.1), (-0.1, 0, 0))[handle]
        return settled.PublicControlStep(image.copy(), kind, handle, True)

    monkeypatch.setattr(settled, "validate_joint_future", future)
    return acquired, frames, execute, requests, trace


def test_settled_controls_wait_before_and_after_each_control_without_refitting(session):
    acquired, frames, execute, requests, trace = session
    saved = copy.deepcopy(acquired)
    before_fits = [item for item in trace if item[0] == "fit"]
    result = settled.run_settled_controls(acquired, frames, execute)
    expected = [("wait", None)] * 3
    for handle in (0, 1, 2):
        expected += [("move", handle)] * 4 + [("wait", None)] * 3
    assert requests == expected
    assert result["terminal"] == "completed" and result["failure"] is None
    assert [trial["start_frame"] for trial in result["trials"]] == [56, 63, 70]
    assert [trial["end_frame"] for trial in result["trials"]] == [63, 70, 77]
    assert all(trial["motion"]["supported"] for trial in result["trials"])
    assert result["frozen_maps_unchanged"] and acquired == saved
    assert [item for item in trace if item[0] == "fit"] == before_fits
    assert all(len(item["candidates"]) == 2 for item in result["prefixes"])
    assert result["pose_domain"] == "returned_image_camera; not current-player pose"
    assert not result["navigation_accepted"] and not result["metric_scale_known"]


def test_rejected_acquisition_never_dispatches_continuation(session):
    acquired, _, execute, requests, _ = session
    result = settled.run_settled_controls(
        acquired | {"terminal": "rejected"}, [], execute
    )
    assert result["terminal"] == "not_started" and not requests


@pytest.mark.parametrize("field", ["frame_count", "pixels", "future_pose"])
def test_forged_initial_checkpoint_fails_before_any_new_control(session, field):
    acquired, frames, execute, requests, _ = session
    if field == "frame_count":
        frames = frames[:-1]
    elif field == "pixels":
        acquired["snapshot"]["prefixes"][-1]["pixels"][0][0] = [1, 1]
    else:
        acquired["snapshot"]["decisions"][-1]["candidates"][0]["future"]["poses"][0][
            "translation"
        ][0] += 0.1
    with pytest.raises(ValueError):
        settled.run_settled_controls(acquired, frames, execute)
    assert not requests


@pytest.mark.parametrize(
    "field", ["kind", "handle", "ticks", "compliant", "receipt", "shape"]
)
def test_failed_controls_do_not_enter_future_pose_evidence(session, field):
    acquired, frames, _, _, _ = session

    def execute(kind, handle):
        if field == "receipt":
            return {"frame": frames[-1]}
        return settled.PublicControlStep(
            frames[-1][:100] if field == "shape" else frames[-1],
            "move" if field == "kind" else kind,
            0 if field == "handle" else handle,
            field != "compliant",
            np.int64(1) if field == "ticks" else 1,
        )

    result = settled.run_settled_controls(acquired, frames, execute)
    assert result["terminal"] == "control_failed" and result["failure"]
    assert result["prefixes"] == [] and len(result["steps"]) == 1
    assert result["evidence_frames"] == len(frames)
    json.dumps(result, allow_nan=False)


def test_public_instability_exhausts_wait_budget_without_starting_motion(
    session, monkeypatch
):
    acquired, frames, execute, requests, _ = session
    original = settled.validate_joint_future

    def drifting(candidate, past, current, **kwargs):
        result = original(candidate, past, current, **kwargs)
        result["poses"][0]["translation"][1] += (
            current.frame_count - len(frames)
        ) * 0.1
        return result

    monkeypatch.setattr(settled, "validate_joint_future", drifting)
    result = settled.run_settled_controls(acquired, frames, execute)
    assert requests == [("wait", None)] * 8
    assert result["terminal"] == "settle_exhausted" and not result["trials"]


def test_original_model_support_failure_stops_and_retains_failed_prefix(
    session, monkeypatch
):
    acquired, frames, execute, requests, _ = session
    original = settled.validate_joint_future

    def unsupported(candidate, past, current, **kwargs):
        result = original(candidate, past, current, **kwargs)
        if current.frame_count > len(frames) and candidate.bootstrap_slot == 2:
            result.update(status="unavailable", frame_checks=[], poses=[])
        return result

    monkeypatch.setattr(settled, "validate_joint_future", unsupported)
    result = settled.run_settled_controls(acquired, frames, execute)
    assert result["terminal"] == "support_aborted" and requests == [("wait", None)]
    assert (
        len(result["prefixes"]) == 1
        and len(result["prefixes"][0]["common_pose_slots"]) == 1
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"minimum_wait_ticks": 2},
        {"maximum_wait_ticks": 9},
        {"maximum_translation_ratio": 0.02},
        {"move_ticks": 5},
        {"temporal": {"maximum_frames": 128}},
        {"handles": [False, 1, 2]},
        {"handles": [0, 2, 1]},
        {"move_ticks": 4.0},
        {"stable_prefixes": 2.0},
    ],
)
def test_invalid_budgets_cannot_dispatch_controls(session, policy):
    acquired, frames, execute, requests, _ = session
    with pytest.raises(ValidationError):
        settled.run_settled_controls(acquired, frames, execute, policy=policy)
    assert not requests


def test_partial_move_retains_executed_endpoint_without_supported_prediction(
    session, monkeypatch
):
    acquired, frames, execute, requests, _ = session
    original = settled.validate_joint_future

    def unavailable(candidate, past, current, **kwargs):
        result = original(candidate, past, current, **kwargs)
        if current.frame_count >= len(frames) + 5:
            result.update(status="unavailable", frame_checks=[], poses=[])
        return result

    monkeypatch.setattr(settled, "validate_joint_future", unavailable)
    result = settled.run_settled_controls(acquired, frames, execute)
    assert requests == [("wait", None)] * 3 + [("move", 0)] * 2
    assert result["terminal"] == "support_aborted"
    trial = result["trials"][0]
    assert trial["end_frame"] == trial["start_frame"] + 2
    assert trial["move_end_frame"] is None and trial["motion"] is None
    assert trial["end_frame"] == result["prefixes"][-1]["absolute_frame"]
