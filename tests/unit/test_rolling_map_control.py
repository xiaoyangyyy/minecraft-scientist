import copy
import json

import numpy as np
import pytest
from pydantic import ValidationError
from test_causal_map_window import CAMERA, tracks
from test_causal_map_window import (
    geometry as geometry,  # noqa: PLC0414 -- fixture re-export
)

from minecraft_scientist.exploration import causal_map_window as causal
from minecraft_scientist.exploration import rolling_map_control as rolling
from minecraft_scientist.exploration.settled_map_control import PublicControlStep


@pytest.fixture
def session(geometry, monkeypatch):
    trace, _ = geometry
    frames = [np.full((224, 224, 3), index, dtype=np.uint8) for index in range(12)]
    parent = {
        "terminal": "map_unavailable",
        "failure": None,
        "seed": 2005,
        "acquisition": {
            "anchor_frame": 0,
            "evidence_frames": 12,
            "steps": [
                {"requested_handle": handle} for handle in [0] * 6 + [2] * 4 + [1] * 2
            ],
        },
        "settled": None,
    }
    requests = []

    def track(images, **kwargs):
        return tracks(len(images), offset=int(images[0][0, 0, 0]))

    def execute(kind, handle):
        requests.append((kind, handle))
        trace.append(("execute", handle))
        return PublicControlStep(
            np.full_like(frames[0], 11 + len(requests)), kind, handle, True
        )

    monkeypatch.setattr(rolling, "track_frame_chain", track)
    return parent, frames, execute, requests, trace


def test_rolling_windows_freeze_before_future_and_keep_explicit_lineage(session):
    parent, frames, execute, requests, trace = session
    before = copy.deepcopy(parent)
    result = rolling.maintain_public_maps(parent, frames, execute, CAMERA)
    assert result["terminal"] == "completed" and result["failure"] is None
    assert requests == [("move", 3)] * 2 + [("move", 1)] * 2
    assert parent == before and result["parent_sha256"] == rolling._digest(parent)
    first, second = result["epochs"]
    assert first["parent_sha256"] == result["parent_sha256"]
    assert second["parent_sha256"] == rolling._digest(first)
    assert [entry["frozen"]["anchor_frame"] for entry in result["epochs"]] == [0, 2]
    assert all(entry["terminal"] == "validated" for entry in result["epochs"])
    assert all(
        entry["frozen"]["snapshot"]["decisions"] == [] for entry in result["epochs"]
    )
    assert trace[:10] == [("fit", slot) for slot in range(2, 12)]
    assert trace[10] == ("execute", 3)
    assert sum(kind == "fit" for kind, _ in trace) == 20
    assert first["frozen"]["snapshot"]["maps"] == first["terminal_snapshot"]["maps"]
    assert not result["global_gauge_aligned"] and not result["navigation_accepted"]


def test_unobservable_initial_history_requests_least_used_nonsemantic_handle(
    session, monkeypatch
):
    parent, frames, execute, requests, _ = session
    fit = causal.fit_joint_map

    def unsupported_first(past, *args, **kwargs):
        model = fit(past, *args, **kwargs)
        return (
            model.model_copy(update={"supported": False})
            if past.pixels[0][0][0] == 40.5
            else model
        )

    monkeypatch.setattr(causal, "fit_joint_map", unsupported_first)
    result = rolling.maintain_public_maps(parent, frames, execute, CAMERA)
    assert result["terminal"] == "completed"
    assert requests[:3] == [("move", 3), ("move", 1), ("move", 1)]
    first = result["epochs"][0]
    assert [attempt["supported_maps"] for attempt in first["attempts"]] == [0, 10]
    assert (
        first["attempts"][0]["snapshot"]["past"]["pixels"]
        != first["attempts"][1]["snapshot"]["past"]["pixels"]
    )
    assert (
        first["attempts"][0]["track_namespace"]
        != first["attempts"][1]["track_namespace"]
    )


def test_gauge_bridge_freezes_before_second_epoch_future_and_never_promotes_navigation(
    session, monkeypatch
):
    parent, frames, execute, requests, _ = session
    seen = []

    def bridge(first, second, **kwargs):
        assert len(requests) == 2
        assert first["decisions"] == second["decisions"] == []
        assert first["prefixes"] == second["prefixes"] == []
        seen.append((first["anchor_frame"], second["anchor_frame"]))
        return {"supported": True, "proof": "unit fake; not geometric/runtime evidence"}

    monkeypatch.setattr(rolling, "register_map_gauges", bridge)
    result = rolling.maintain_public_maps(
        parent, frames, execute, CAMERA, gauge_policy={}
    )
    assert seen == [(0, 2)] and len(requests) == 4
    assert result["supported_local_gauge_links"] == 1
    item = result["gauge_bridges"][0]
    assert item["evidence_sha256"] == rolling._digest(item["evidence"])
    assert (item["parent_epoch"], item["child_epoch"]) == (0, 1)
    assert not result["global_gauge_aligned"] and not result["navigation_accepted"]


def test_rejected_gauge_pair_cannot_rewrite_already_accepted_motion(session):
    parent, frames, execute, requests, _ = session
    result = rolling.maintain_public_maps(
        parent, frames, execute, CAMERA, gauge_policy={}
    )
    assert result["terminal"] == "completed" and len(requests) == 4
    assert result["supported_local_gauge_links"] == 0
    assert not result["gauge_bridges"][0]["evidence"]["supported"]
    assert all(epoch["terminal"] == "validated" for epoch in result["epochs"])


def test_future_control_cannot_modify_frozen_gauge_record(session, monkeypatch):
    parent, frames, execute, requests, _ = session
    evidence = {"supported": True}
    monkeypatch.setattr(
        rolling, "register_map_gauges", lambda *args, **kwargs: evidence
    )

    def mutate(kind, handle):
        if len(requests) >= 2:
            evidence["supported"] = False
        return execute(kind, handle)

    with pytest.raises(ValueError, match="frozen gauge"):
        rolling.maintain_public_maps(parent, frames, mutate, CAMERA, gauge_policy={})


def test_exhaustion_retains_all_failed_fits_without_future_validation(
    session, monkeypatch
):
    parent, frames, execute, requests, _ = session
    fit = causal.fit_joint_map
    monkeypatch.setattr(
        causal,
        "fit_joint_map",
        lambda *args, **kwargs: fit(*args, **kwargs).model_copy(
            update={"supported": False}
        ),
    )
    result = rolling.maintain_public_maps(parent, frames, execute, CAMERA)
    assert result["terminal"] == "map_unavailable" and len(result["epochs"]) == 1
    assert requests == [("move", 3)] * 4
    entry = result["epochs"][0]
    assert len(entry["attempts"]) == 5 and entry["frozen"] is None
    assert entry["future_prefixes"] == [] and entry["decision"] is None


def test_global_action_cap_preserves_incomplete_validation_without_promotion(session):
    parent, frames, execute, requests, _ = session
    parent["acquisition"]["anchor_frame"] = 188
    result = rolling.maintain_public_maps(parent, frames, execute, CAMERA)
    assert result["terminal"] == "budget_exhausted" and len(requests) == 1
    entry = result["epochs"][0]
    assert entry["decision"]["instruction"] == "collect"
    assert result["steps"][0]["action_response_frame"] == 200


@pytest.mark.parametrize(
    "case", ["kind", "handle", "ticks", "compliant", "shape", "receipt"]
)
def test_failed_receipts_never_enter_rolling_future_evidence(session, case):
    parent, frames, _, _, _ = session

    def execute(kind, handle):
        if case == "receipt":
            return {"frame": frames[0]}
        return PublicControlStep(
            frames[0][:100] if case == "shape" else frames[0],
            "wait" if case == "kind" else kind,
            0 if case == "handle" else handle,
            case != "compliant",
            np.int64(1) if case == "ticks" else 1,
        )

    result = rolling.maintain_public_maps(parent, frames, execute, CAMERA)
    assert result["terminal"] == "control_failed" and result["failure"]
    assert not result["epochs"][0]["future_prefixes"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("case", ["anchor", "seed", "count", "camera", "failed_parent"])
def test_invalid_rolling_parent_refuses_before_dispatch(session, case):
    parent, frames, execute, requests, _ = session
    camera = CAMERA
    if case == "anchor":
        parent["acquisition"]["anchor_frame"] = -1
    elif case == "seed":
        parent["seed"] = True
    elif case == "count":
        frames = frames[:-1]
    elif case == "camera":
        camera = np.zeros((3, 3))
    else:
        parent["failure"] = {"type": "injected"}
    with pytest.raises(ValueError):
        rolling.maintain_public_maps(parent, frames, execute, camera)
    assert not requests


@pytest.mark.parametrize(
    "policy",
    [
        {"maximum_epochs": 3},
        {"history_frames": 13},
        {"window": {"maximum_direction_disagreement": 20}},
    ],
)
def test_rolling_policy_cannot_silently_relax_geometry(session, policy):
    parent, frames, execute, requests, _ = session
    with pytest.raises(ValidationError):
        rolling.maintain_public_maps(parent, frames, execute, CAMERA, policy=policy)
    assert not requests
