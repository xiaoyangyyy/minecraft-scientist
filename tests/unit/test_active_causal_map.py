import copy

import numpy as np
import pytest
from pydantic import ValidationError
from test_causal_map_window import CAMERA, tracks
from test_causal_map_window import (
    geometry as geometry,  # noqa: PLC0414 -- pytest fixture re-export
)

from minecraft_scientist.exploration import active_causal_map as active
from minecraft_scientist.exploration import causal_map_window as causal
from minecraft_scientist.exploration.visibility_acquisition import PublicProbeStep


@pytest.fixture
def runner(geometry, monkeypatch):
    trace, _ = geometry
    requests = []
    monkeypatch.setattr(
        active, "track_frame_chain", lambda frames, **kw: tracks(len(frames))
    )
    initial = np.zeros((224, 224, 3), dtype=np.uint8)

    def execute(handle):
        requests.append(handle)
        trace.append(("execute", handle))
        return PublicProbeStep(initial.copy(), handle, True)

    return initial, execute, requests, trace


def test_live_dispatch_latches_first_prefix_before_distinct_control(runner):
    initial, execute, requests, trace = runner
    result = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=48)
    assert requests == [0, 0, 0, 1, 1]
    assert result["terminal"] == "validated"
    assert result["fit_count"] == 2
    assert [item["absolute_frame"] for item in result["acquisition_attempts"]] == [51]
    assert trace.index(("fit", 3)) < trace.index(("execute", 1))
    assert all(kind != "fit" for kind, _ in trace[trace.index(("execute", 1)) :])
    assert result["controller_decision"]["absolute_frame"] == 53
    assert len(result["prefixes"]) == len(result["steps"]) == 5
    assert not result["navigation_accepted"] and not result["metric_scale_known"]
    past = result["acquisition_attempts"][0]["snapshot"]
    assert past["prefixes"] == past["decisions"] == []
    assert past["maps"] == result["snapshot"]["maps"]


def test_declared_control_switch_is_bounded_and_holdout_excludes_both(runner):
    initial, execute, requests, _ = runner
    result = active.acquire_causal_map(
        initial, execute, CAMERA, anchor_frame=0, policy={"minimum_ticks": 7}
    )
    assert requests == [0] * 6 + [2] + [1] * 2
    assert result["evidence_frames"] == 10
    assert result["fit_count"] == 6


def test_exhaustion_retains_all_prefixes_and_never_requests_future(runner, monkeypatch):
    initial, execute, requests, _ = runner
    original = causal.fit_joint_map

    def unsupported(*args, **kwargs):
        return original(*args, **kwargs).model_copy(update={"supported": False})

    monkeypatch.setattr(causal, "fit_joint_map", unsupported)
    result = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=48)
    assert requests == [0] * 6 + [2] * 4
    assert result["terminal"] == "exhausted"
    assert result["fit_count"] == 44
    assert len(result["acquisition_attempts"]) == 8
    assert result["snapshot"] is None and result["controller_decision"] is None
    assert all(not item["eligible"] for item in result["acquisition_attempts"])


def test_original_support_loss_aborts_before_replenishment(runner, monkeypatch):
    initial, execute, requests, _ = runner
    monkeypatch.setattr(
        active,
        "track_frame_chain",
        lambda frames, **kw: tracks(len(frames), ids=tuple(range(31))),
    )
    result = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=0)
    assert result["terminal"] == "support_aborted" and requests == [0]
    assert result["fit_count"] == 0


@pytest.mark.parametrize(
    "kind",
    ["handle", "ticks", "bool_handle", "compliance", "shape", "type", "exception"],
)
def test_failed_controls_never_enter_map_evidence(runner, kind):
    initial, _, _, trace = runner

    def execute(handle):
        if kind == "exception":
            raise RuntimeError("injected controller failure")
        if kind == "type":
            return {"frame": initial, "executed_handle": handle}
        return PublicProbeStep(
            initial if kind != "shape" else initial[:100],
            handle
            if kind not in ("handle", "bool_handle")
            else (2 if kind == "handle" else False),
            kind != "compliance",
            2 if kind == "ticks" else 1,
        )

    result = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=0)
    assert result["terminal"] == "control_failed"
    assert result["failure"] and len(result["steps"]) == 1
    assert result["evidence_frames"] == 1
    assert result["prefixes"] == result["acquisition_attempts"] == []
    assert not any(kind == "fit" for kind, _ in trace)


def test_future_failed_control_leaves_latched_past_immutable(runner):
    initial, execute_ok, _, _ = runner

    def execute(handle):
        return (
            execute_ok(handle)
            if handle == 0
            else PublicProbeStep(initial.copy(), 2, False)
        )

    result = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=0)
    assert result["terminal"] == "control_failed"
    assert result["evidence_frames"] == 4
    assert len(result["prefixes"]) == 3
    assert result["snapshot"] == result["acquisition_attempts"][0]["snapshot"]


def test_prefix_revision_becomes_tracking_failure_not_new_theory(runner, monkeypatch):
    initial, execute, requests, _ = runner

    def track(frames, **kwargs):
        return tracks(len(frames), offset=int(len(frames) > 2))

    monkeypatch.setattr(active, "track_frame_chain", track)
    result = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=0)
    assert result["terminal"] == "tracking_failed" and requests == [0, 0]
    assert len(result["prefixes"]) == 1 and result["fit_count"] == 0


@pytest.mark.parametrize(
    "policy",
    [
        {"handles": [0, 0]},
        {"handles": [False, 2]},
        {"handles": [-1, 2]},
        {"minimum_ticks": 10, "maximum_ticks": 8},
        {"maximum_ticks": 6},
        {"maximum_ticks": 12},
        {"window": {"minimum_models": 1}},
    ],
)
def test_invalid_policy_cannot_dispatch_actions(runner, policy):
    initial, execute, requests, _ = runner
    with pytest.raises(ValidationError):
        active.acquire_causal_map(
            initial, execute, CAMERA, anchor_frame=0, policy=policy
        )
    assert not requests


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"anchor_frame": -1},
        {"anchor_frame": True},
        {"camera": np.zeros((3, 3))},
    ],
)
def test_bad_session_arguments_are_rejected_before_actions(runner, kwargs):
    initial, execute, requests, _ = runner
    params = {"camera": CAMERA, "anchor_frame": 0} | kwargs
    with pytest.raises(ValueError):
        active.acquire_causal_map(initial, execute, **params)
    assert not requests


def test_replay_is_deterministic_and_snapshots_do_not_alias(runner):
    initial, execute, _, _ = runner
    first = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=48)
    second = active.acquire_causal_map(initial, execute, CAMERA, anchor_frame=48)
    assert first == second
    frozen = copy.deepcopy(first["acquisition_attempts"])
    first["snapshot"]["maps"].clear()
    assert first["acquisition_attempts"] == frozen
