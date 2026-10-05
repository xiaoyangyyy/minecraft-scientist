import copy
from itertools import groupby

import pytest
from pydantic import ValidationError

pytest.importorskip("cv2", reason="optional geometry")

from test_joint_multiview_map import CAMERA, fixture_tracks, prefix
from test_stable_keyframe_map import public_scene as public_scene  # noqa: PLC0414

from minecraft_scientist.exploration import stable_keyframe_map as stable


def acquire(first, execute, **kwargs):
    return stable.acquire_stable_keyframe_map(
        first,
        execute,
        CAMERA,
        anchor_frame=48,
        seed=13,
        micro_controls=True,
        responsive_controls=True,
        recovery_controls=True,
        **kwargs,
    )


def test_recovery_protocol_without_failure_keeps_original_geometry(public_scene):
    first, execute, calls = public_scene
    result = acquire(first, execute)
    assert result["terminal"] == "validated" and len(calls) == 63
    assert result["failed_wait_windows"] == result["recovery_wait_windows"] == []
    assert result["epoch_links"] == []
    assert len(result["epochs"]) == 1
    epoch = result["epochs"][0]
    assert epoch["status"] == "validated" and epoch["termination_frame"] == 111
    assert epoch["frame_ordinals"] == [56, 65, 74, 83, 92, 101]
    assert epoch["trial_ids"] == list(range(5)) and epoch["attempt_indices"] == [0]
    assert not epoch["global_gauge_aligned"]
    assert all(entry["recovery_windows"] == 0 for entry in result["settles"])
    assert result["policy"] == stable.StableRecoveringKeyframePolicy().model_dump(
        mode="json"
    )


def test_failed_initial_windows_remain_recorded_without_invented_map(
    public_scene, monkeypatch
):
    first, execute, calls = public_scene
    monkeypatch.setattr(stable, "_visible_flow", lambda *args: {"stable": False})
    result = acquire(first, execute)
    assert result["terminal"] == "initial_wait_exhausted" and len(calls) == 48
    assert result["epochs"] == result["attempts"] == result["samples"] == []
    assert [entry["interval"] for entry in result["failed_wait_windows"]] == [
        [48, 64],
        [64, 80],
        [80, 96],
    ]
    assert [entry["recovery_available"] for entry in result["failed_wait_windows"]] == [
        True,
        True,
        False,
    ]
    assert all(entry["epoch_id"] is None for entry in result["failed_wait_windows"])
    assert [entry["status"] for entry in result["recovery_wait_windows"]] == [
        "exhausted",
        "exhausted",
    ]
    assert result["future"] is None and result["motion"] is None


def test_recovered_initial_prefix_uses_first_eligible_tail(public_scene, monkeypatch):
    first, execute, _ = public_scene
    flow = stable._visible_flow

    def delayed(first, last, settings):
        value = flow(first, last, settings)
        if int(last[0, 0, 0]) <= 16:
            return value | {"stable": False}
        return value

    monkeypatch.setattr(stable, "_visible_flow", delayed)
    result = acquire(first, execute)
    assert result["terminal"] == "validated"
    assert result["settles"][0]["interval"] == [48, 67]
    assert result["settles"][0]["stable_tail"] == [[64, 65], [65, 66], [66, 67]]
    assert result["failed_wait_windows"][0]["interval"] == [48, 64]
    assert result["recovery_wait_windows"][0]["interval"] == [64, 67]
    assert result["epochs"][0]["anchor_frame"] == 67


def test_connection_loss_after_recovery_preserves_parent_and_resets_local_history(
    public_scene, monkeypatch
):
    first, execute, calls = public_scene
    flow, fit = stable._visible_flow, stable.fit_joint_map

    def falling(first, last, settings):
        a, b = int(first[0, 0, 0]), int(last[0, 0, 0])
        measured = flow(first, last, settings)
        if b - a == 1 and 55 <= b <= 70:
            return measured | {"stable": False}
        if a == 53 and b == 73:
            return measured | {
                "tracks": {"track_ids": [], "seed_count": 40},
                "cells": 0,
            }
        return measured

    def rejected_parent(*args, **kwargs):
        evidence = fit(*args, **kwargs)
        return (
            evidence.model_copy(update={"supported": False})
            if len(calls) == 53
            else evidence
        )

    monkeypatch.setattr(stable, "_visible_flow", falling)
    monkeypatch.setattr(stable, "fit_joint_map", rejected_parent)
    result = acquire(first, execute)
    assert result["terminal"] == "validated" and len(calls) == 128
    parent, child = result["epochs"]
    assert parent["status"] == "connection_lost_after_recovery"
    assert parent["trial_ids"] == list(range(6))
    assert parent["frame_ordinals"] == [56, 65, 74, 83, 92, 101]
    assert parent["attempt_indices"] == [0] and parent["termination_frame"] == 121
    assert child["anchor_frame"] == 121 and child["trial_ids"] == list(range(6, 11))
    assert child["frame_ordinals"] == [121, 130, 139, 148, 157, 166]
    assert child["attempt_indices"] == [1]
    link = result["epoch_links"][0]
    assert link["parent_sha256"] == stable._digest(parent)
    assert not link["supported"] and not link["metric_scale_known"]
    assert (
        result["attempts"][0]["proposal"]["index"]["namespace"] == parent["namespace"]
    )
    assert (
        result["frozen"]["index"]["namespace"]
        == child["namespace"]
        != parent["namespace"]
    )
    assert not any(
        model["supported"] for model in result["attempts"][0]["proposal"]["maps"]
    )
    assert all(
        item["sha256"] == stable._digest(item["proposal"])
        for item in result["attempts"]
    )
    choices = result["control_choices"]
    assert [item["observed_trials"] for item in choices] == list(range(6)) + list(
        range(5)
    )
    assert choices[6]["scoring_trial_ids"] == [] and choices[6]["epoch_id"] == 1
    assert choices[10]["scoring_trial_ids"] == [6, 7, 8, 9]
    # Changing the namespace cannot bypass the episode-wide consecutive-control cap.
    assert choices[9]["required_distinct_from"] == 0 and choices[9]["handle"] != 0
    assert (
        max(len(list(run)) for _, run in groupby(item["handle"] for item in choices))
        <= 4
    )
    future = result["future_control_choice"]
    assert future["epoch_id"] == 1 and future["scoring_trial_ids"] == list(range(6, 11))
    assert result["motion"]["frame_ordinals"] == [166, 176]
    lost = next(item for item in result["samples"] if item.get("trial") == 5)
    assert lost["selected"] is False and lost["correspondence_supported"] is False
    assert result["failed_wait_windows"][0]["interval"] == [102, 118]
    assert result["recovery_wait_windows"][0]["interval"] == [118, 121]
    assert result["failed_wait_windows"][0]["epoch_id"] == 0
    assert stable._digest(result["frozen"]) == result["frozen_sha256"]
    assert not result["navigation_accepted"]


@pytest.mark.parametrize("recovered", [False, True])
def test_stall_or_loss_without_recovery_does_not_open_a_new_epoch(
    public_scene, monkeypatch, recovered
):
    first, execute, calls = public_scene
    flow = stable._visible_flow

    def stalled(first, last, settings):
        a, b = int(first[0, 0, 0]), int(last[0, 0, 0])
        measured = flow(first, last, settings)
        if recovered and b - a == 1 and 10 <= b <= 25:
            return measured | {"stable": False}
        if b - a > 1:
            return measured | (
                {"median_flow_pixels": 0}
                if recovered
                else {"tracks": {"track_ids": [], "seed_count": 40}, "cells": 0}
            )
        return measured

    monkeypatch.setattr(stable, "_visible_flow", stalled)
    result = acquire(first, execute)
    assert result["terminal"] == "map_unavailable"
    assert len(result["epochs"]) == 1 and not result["epoch_links"]
    assert len(result["control_choices"]) == 11
    assert len(result["steps"]) == len(calls) < 240
    assert result["frozen"] is None and result["future_control_choice"] is None
    assert bool(result["recovery_wait_windows"]) == recovered


def test_future_wait_recovery_cannot_refit_frozen_past(public_scene, monkeypatch):
    first, execute, calls = public_scene
    flow, fit = stable._visible_flow, stable.fit_joint_map
    frozen_inputs, fitted_at = [], []

    def record(*args, **kwargs):
        frozen_inputs.append(copy.deepcopy(args[0].model_dump(mode="json")))
        fitted_at.append(len(calls))
        return fit(*args, **kwargs)

    def future_unstable(first, last, settings):
        a, b = int(first[0, 0, 0]), int(last[0, 0, 0])
        measured = flow(first, last, settings)
        if b - a == 1 and 56 <= b <= 71:
            return measured | {"stable": False}
        return measured

    monkeypatch.setattr(stable, "_visible_flow", future_unstable)
    monkeypatch.setattr(stable, "fit_joint_map", record)
    result = acquire(first, execute)
    assert result["terminal"] == "validated" and len(calls) == 74
    assert fitted_at == [53] * 4
    assert all(data == result["frozen"]["past"] for data in frozen_inputs)
    assert result["future"]["absolute_frame"] == 122
    assert result["settles"][-1]["interval"] == [103, 122]
    assert result["failed_wait_windows"][0]["phase"] == "future"
    assert (
        result["epochs"][0]["termination_frame"] == 122 and len(result["epochs"]) == 1
    )
    assert stable._digest(result["frozen"]) == result["frozen_sha256"]


def test_two_losses_cannot_create_third_epoch_or_extra_local_trials(
    public_scene, monkeypatch
):
    first, execute, calls = public_scene
    flow = stable._visible_flow

    def falling_twice(first, last, settings):
        a, b = int(first[0, 0, 0]), int(last[0, 0, 0])
        measured = flow(first, last, settings)
        if b - a == 1 and (10 <= b <= 25 or 30 <= b <= 45):
            return measured | {"stable": False}
        if (a, b) in ((8, 28), (28, 48)):
            return measured | {
                "tracks": {"track_ids": [], "seed_count": 40},
                "cells": 0,
            }
        if b - a > 1:
            return measured | {"median_flow_pixels": 0}
        return measured

    monkeypatch.setattr(stable, "_visible_flow", falling_twice)
    result = acquire(first, execute)
    assert result["terminal"] == "map_unavailable" and len(calls) == 138
    assert len(result["epochs"]) == 2 and len(result["epoch_links"]) == 1
    assert result["epochs"][0]["trial_ids"] == [0]
    assert result["epochs"][1]["trial_ids"] == list(range(1, 12))
    assert len(result["control_choices"]) == 12
    assert len(result["recovery_wait_windows"]) == 2
    assert result["epochs"][1]["frame_ordinals"] == [76]
    assert result["frozen"] is None and result["future_control_choice"] is None


def test_long_two_epoch_acquisition_stops_at_global_action_cap(
    public_scene, monkeypatch
):
    from minecraft_scientist.exploration.joint_multiview_map import JointMapEvidence

    first, execute, calls = public_scene
    flow = stable._visible_flow
    tracks, _, _ = fixture_tracks(frames=13, noise=0.02)

    def long_wait(first, last, settings):
        a, b = int(first[0, 0, 0]), int(last[0, 0, 0])
        measured = flow(first, last, settings)
        moves = [index + 1 for index, (kind, _) in enumerate(calls) if kind == "move"]
        if b - a == 1 and moves:
            required_unstable = 16 if len(moves) == 8 else 13
            return measured | {"stable": len(calls) - moves[-1] > required_unstable}
        if (a, b) == (127, 147):
            return measured | {
                "tracks": {"track_ids": [], "seed_count": 40},
                "cells": 0,
            }
        return measured

    def unsupported(tracks, camera, **kwargs):
        return JointMapEvidence(
            supported=False,
            reason="injected rejected geometry",
            config=kwargs["config"],
            seed=kwargs["seed"],
            camera=camera,
            bootstrap_slot=kwargs["bootstrap_slot"],
            fitting_frames=tracks.frame_count,
            input_sha256="0" * 64,
        )

    monkeypatch.setattr(stable, "_visible_flow", long_wait)
    monkeypatch.setattr(
        stable,
        "track_frame_chain",
        lambda frames, **kwargs: prefix(tracks, len(frames)),
    )
    monkeypatch.setattr(stable, "fit_joint_map", unsupported)
    result = acquire(first, execute)
    assert result["terminal"] == "budget_exhausted"
    assert len(calls) == len(result["steps"]) == 240
    assert result["steps"][-1]["absolute_frame"] == 288
    assert len(result["epochs"]) == 2 and len(result["epoch_links"]) == 1
    assert result["epochs"][0]["trial_ids"] == list(range(8))
    assert result["epochs"][1]["termination_frame"] == 288
    assert 11 < len(result["control_choices"]) <= 22
    assert all(len(epoch["trial_ids"]) <= 11 for epoch in result["epochs"])
    assert all(
        item["sha256"] == stable._digest(item["proposal"])
        for item in result["attempts"]
    )
    assert result["frozen"] is None and result["future"] is None


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_recovery_flag_must_be_explicit_boolean(public_scene, flag):
    first, execute, calls = public_scene
    with pytest.raises(ValueError, match="explicit boolean"):
        stable.acquire_stable_keyframe_map(
            first, execute, CAMERA, anchor_frame=48, recovery_controls=flag
        )
    assert calls == []


def test_recovery_registration_cannot_disguise_response_policy(public_scene):
    first, execute, calls = public_scene
    with pytest.raises(ValueError, match="responsive micro registration"):
        stable.acquire_stable_keyframe_map(
            first, execute, CAMERA, anchor_frame=48, recovery_controls=True
        )
    with pytest.raises(ValidationError):
        stable.acquire_stable_keyframe_map(
            first,
            execute,
            CAMERA,
            anchor_frame=48,
            micro_controls=True,
            responsive_controls=True,
            policy=stable.StableRecoveringKeyframePolicy(),
        )
    assert calls == []
    for field, value in (
        ("maximum_recovery_windows", 3),
        ("maximum_local_epochs", 3),
        ("maximum_total_acquisition_trials", 23),
        ("maximum_episode_actions", 289),
    ):
        with pytest.raises(ValidationError):
            stable.StableRecoveringKeyframePolicy(**{field: value})
    with pytest.raises(ValidationError, match="registered geometry"):
        stable.StableRecoveringKeyframePolicy(stability={"maximum_wait_ticks": 12})
